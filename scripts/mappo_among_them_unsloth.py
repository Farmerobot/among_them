#!/usr/bin/env python3
"""
MAPPO Self-Play Training for Among Them using Unsloth

Key components:
1. Self-play trajectory collection (all 5 players use same policy)
2. Centralized critic with global state view (History.to_text() + all player thoughts)
3. PPO with clipped surrogate objective
4. Value head pre-trained on existing games
5. Efficient training with Unsloth LoRA adapters
6. Optimizations: Unsloth's native cache with branching, word normalization
"""

# Environment setup BEFORE importing torch
import os
from typing import Callable
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Uncomment for debugging CUDA errors - forces synchronous execution
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# Help PyTorch reuse memory blocks more efficiently
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from pathlib import Path

# Note: Multiprocessing removed in favor of in-process batch inference

# Unsloth imports
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
import wandb

# Among Them imports
from among_them.models.player import Player
from among_them.utils.end_utils import get_end_game_reason
from among_them.game_engine import GameEngine
from among_them.game_config import GameConfig
from among_them.models.action import Action, ActionType
from among_them.models.history import History
from among_them.models.player_role import PlayerRole
from among_them.models.end_game import EndGameReason
from among_them.game_jsonencoder import game_object_hook, GameJSONEncoder


# ============================================================================
# MARK: CONFIGURATION
# ============================================================================

@dataclass
class MAPPOConfig:
    """Configuration for MAPPO training"""
    # Model configuration
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    max_seq_length: int = 60000 # 3k reasoning per turn with 11 turns avg per player. 20 turns = 60k
    max_reasoning_tokens: int = 3000 # arbitrary based on avg on ref-policy
    load_in_4bit: bool = True
    lora_rank: int = 16
    
    # Game configuration
    num_players: int = 5
    num_impostors: int = 1
    num_tasks: int = 2
    map_size: int = 0
    num_task_phase_actions_per_player: int = 8
    num_discuss_phase_actions_per_player: int = 3
    impostor_cooldown: int = 1
    
    # Training configuration
    num_policy_iterations: int = 400
    trajectories_per_iteration: int = 4
    inference_batch_size: int = 16  # Number of environments to batch during inference
    actor_lr: float = 1e-6
    critic_lr: float = 3e-6
    gradient_accumulation_steps: int = 4
    
    # PPO-specific
    ppo_epochs: int = 4
    ppo_minibatch_size: int = 128  # Turns per minibatch during PPO update (A100: 128-256)
    precompute_batch_size: int = 16  # DEPRECATED: use logprob_batch_size instead
    logprob_batch_size: int = 4  # Batch size for log-prob computation (small - full sequences are very long)
    value_batch_size: int = 4  # Batch size for value computation (small - global states are long)
    clip_epsilon: float = 0.2
    gamma: float = 1.0
    gae_lambda: float = 0.95
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    kl_penalty_coef: float = 0.1  # KL divergence penalty from reference policy
    
    # Value head pretraining
    pretrain_value_head: bool = True
    pretrain_epochs: int = 10
    pretrain_examples: int = 3
    pretrain_lr: float = 1e-4
    data_dir: str = "/content/among_them/data" # TODO change them
    
    # Output paths
    output_dir: str = "/content/drive/MyDrive/among_them/outputs/mappo_training"
    checkpoint_dir: str = "/content/drive/MyDrive/among_them/outputs/mappo_checkpoints"
    
    # Checkpoint loading
    load_model: Optional[str] = None  # Iteration number (e.g., "50"), "final", or path to checkpoint dir
    load_head: Optional[str] = None   # Iteration number (e.g., "50"), "final", or path to value_head.pt file
    resume_from: Optional[str] = None  # Resume training from checkpoint (iteration number or "latest")
    
    # Logging
    wandb_project: str = "among-them-mappo"
    wandb_run_name: str = "mappo-5players"
    log_every_n_iterations: int = 10
    save_every_n_iterations: int = 50
    save_trajectories: bool = True  # Save trajectories to disk for analysis
    
    # Debug
    debug: bool = True
    seed: int = 42
    load_trajectories_from_disk: Optional[int] = None  # Load last N trajectories from disk (debug only)


# ============================================================================
# TRAJECTORY DATA STRUCTURES
# MARK: DATA STRUCTURES
# ============================================================================

@dataclass
class TurnData:
    """Data for a single turn in a trajectory"""
    player_name: str
    player_role: PlayerRole
    
    # Local observation (what the actor sees)
    conversation: List[Dict[str, str]]
    actions: List[Action]
    chosen_action_idx: int
    chosen_action: Action
    reasoning: str
    
    # Policy outputs - GENERATIVE PPO
    # Stores raw per-token log-probs for entire generation (reasoning + action)
    generation_log_probs: List[float]
    
    # Global state for critic (centralized)
    global_state_repr: str
    
    # === Fields with default values must come after required fields ===
    
    # Action distribution for entropy computation (softmax over discrete actions)
    action_probs: Optional[List[float]] = None  # Probability for each action
    
    # Token statistics
    input_tokens: int = 0
    output_tokens: int = 0
    generation_time: float = 0.0  # Time to generate this turn in seconds
    
    # Value estimates (filled during training)
    value_estimate: Optional[float] = None
    advantage: Optional[float] = None
    returns: Optional[float] = None
    
    # Pre-computed tensors for PPO update (stored on CPU)
    precomputed_old_log_probs: Optional[Any] = None  # torch.Tensor on CPU
    precomputed_ref_log_probs: Optional[Any] = None  # torch.Tensor on CPU
    advantage_tensor: Optional[Any] = None  # torch.Tensor on device
    returns_tensor: Optional[Any] = None  # torch.Tensor on device


@dataclass
class Trajectory:
    """Complete game trajectory with rewards"""
    turns: List[TurnData]
    winner_role: PlayerRole
    end_reason: EndGameReason
    collection_time: float = 0.0  # Time to collect trajectory in seconds
    total_input_tokens: int = 0  # Total input tokens across all turns
    total_output_tokens: int = 0  # Total output tokens across all turns
    
    def get_reward_for_role(self, role: PlayerRole) -> float:
        """Get reward for a specific role with proper mapping"""
        # NO_ACTIONS_LEFT: Crewmates win by timeout but small reward to discourage stalling
        if self.end_reason == EndGameReason.NO_ACTIONS_LEFT:
            return 0.3 if role == PlayerRole.CREWMATE else -0.3
        
        # Normal win conditions
        if self.winner_role == role:
            return 1.0
        else:
            return -1.0
    
    def get_return_for_player(self, player_name: str) -> float:
        """Calculate return for a player"""
        player_role = None
        for turn in self.turns:
            if turn.player_name == player_name:
                player_role = turn.player_role
                break
        
        if player_role is None:
            raise ValueError(f"Player {player_name} not found in trajectory")
        
        return self.get_reward_for_role(player_role)


# ============================================================================
# VALUE HEAD (Centralized Critic)
# MARK: VALUE HEAD
# ============================================================================

class ValueHead(nn.Module):
    """Centralized critic that estimates state value from global state"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.value_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] or [batch_size, hidden_size]
        Returns:
            value: [batch_size, 1] value estimate
        """
        # Cast input to float32 to match float32 weights
        # This prevents both dtype mismatch errors and NaN from float16 overflow
        stable_hidden_states = hidden_states.to(torch.float32)

        if stable_hidden_states.dim() == 3:
            stable_hidden_states = stable_hidden_states[:, -1, :]
        
        value = self.value_proj(stable_hidden_states)
        return value

# ============================================================================
# SELF-PLAY GAME RUNNER
# MARK: SELF-PLAY RUNNER
# ============================================================================

class MAPPOActor:
    """
    Actor that collects self-play trajectories using the shared policy model.
    
    IMPORTANT: This actor uses the SAME model instance as the trainer.
    No separate model loading or syncing needed.
    """
    
    def __init__(self, model, tokenizer, value_head, config: MAPPOConfig, trainer=None):
        self.model = model  # Shared with trainer
        self.tokenizer = tokenizer
        self.value_head = value_head
        self.config = config
        self.trainer = trainer  # Reference to trainer for iteration tracking
        self.progress_callback: Callable[[int, int], None] = lambda x, y: None  # Optional callback for turn-by-turn progress updates
        
        # Set tokenizer padding for batch inference (do once, not every call)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.game_config = GameConfig(
            num_tasks=config.num_tasks,
            num_players=config.num_players,
            num_impostors=config.num_impostors,
            map_size=config.map_size,
            num_task_phase_actions_per_player=config.num_task_phase_actions_per_player,
            num_discuss_phase_actions_per_player=config.num_discuss_phase_actions_per_player,
            impostor_cooldown=config.impostor_cooldown
        )
    
    def _generate_action_with_policy( # MARK: .    GENERATE
        self,
        conversation: List[Dict[str, str]],
        actions: List[Action],
    ) -> Tuple[int, List[float], str, int, int]:
        """
        HYBRID GENERATIVE PPO:
        1. Generates reasoning with per-token log probs.
        2. If "None" action (SPEAK), generates response freely (generative).
        3. If discrete actions, ranks them (using word-norm for selection).
        4. Returns FULL per-token log_probs for (reasoning + action).
        
        Returns: (action_idx, generation_log_probs, reasoning, input_tokens, output_tokens)
        """
        self.model.eval()
        device = self.model.device
        
        # 1. Apply chat template and tokenize
        input_text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(device)
        input_token_count = inputs.input_ids.shape[1]
        
        # 2. Use Unsloth's native cache (start with None for prefill)
        past_key_values = None
        
        # Get </think> token
        think_close_token = self.tokenizer.encode("</think>", add_special_tokens=False)[0]
        
        # TIMING: Initialize timing dictionary
        timing = {
            "prefill": 0.0,
            "reasoning_loop": 0.0,
            "reasoning_tokens": 0,
            "prefix_append": 0.0,
            "kv_clone_total": 0.0,
            "kv_clone_count": 0,
            "action_generation": 0.0,
            "action_tokens_processed": 0,
            "model_forward_time": 0.0,
            "python_overhead": 0.0,
        }
        
        # 3. Generate reasoning
        with torch.no_grad():
            # Prefill prompt
            prompt_len = inputs.input_ids.shape[1]
            cache_position = torch.arange(prompt_len, device=device)
            
            # TIMING: Prefill
            t_prefill_start = time.time()
            # Pass None for prefill, capture Unsloth's cache
            out = self.model(
                inputs.input_ids,
                cache_position=cache_position,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = out.logits
            past_key_values = out.past_key_values  # Capture Unsloth's cache
            timing["prefill"] = time.time() - t_prefill_start
            
            # Sample tokens until </think> AND store log probs
            generated_reason_ids = []
            reasoning_log_probs = []  # Store reasoning log_probs
            cur_pos = torch.tensor([prompt_len], device=device)
            
            # Get generation config for sampling parameters
            gen_cfg = getattr(self.model, "generation_config", None)
            temperature = float(getattr(gen_cfg, "temperature", 1.0) or 1.0)
            top_p = float(getattr(gen_cfg, "top_p", 1.0) or 1.0)
            top_k = int(getattr(gen_cfg, "top_k", 0) or 0)
            
            # if self.config.debug:
            #     print("\n💭 Streaming reasoning tokens: ", end="", flush=True)
            
            # TIMING: Reasoning loop
            t_reasoning_start = time.time()
            for _ in range(self.config.max_reasoning_tokens):
                t_token_start = time.time()
                step_logits = logits[:, -1, :].squeeze(0)
                
                # Temperature scaling
                if temperature != 1.0:
                    step_logits = step_logits / max(temperature, 1e-6)
                
                # Top-k filtering
                if top_k and top_k > 0 and top_k < step_logits.numel():
                    kth_vals, _ = torch.topk(step_logits, top_k)
                    min_keep = kth_vals[-1]
                    step_logits = torch.where(
                        step_logits < min_keep,
                        torch.tensor(float('-inf'), device=device, dtype=step_logits.dtype),
                        step_logits
                    )
                
                # Top-p (nucleus) filtering
                if top_p and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(step_logits, descending=True)
                    sorted_probs = torch.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_mask = cumulative_probs > top_p
                    if sorted_mask.any():
                        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                        sorted_mask[..., 0] = False
                    sorted_logits = torch.where(
                        sorted_mask,
                        torch.tensor(float('-inf'), device=device, dtype=sorted_logits.dtype),
                        sorted_logits
                    )
                    step_logits = torch.full_like(step_logits, float('-inf'))
                    step_logits.scatter_(0, sorted_indices, sorted_logits)
                
                # Sample and get log prob
                step_log_probs = F.log_softmax(step_logits, dim=-1)
                probs = torch.softmax(step_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                token_id = int(next_token.item())
                
                if token_id == think_close_token:
                    # if self.config.debug:
                    #     print("</think>", flush=True)
                    break
                
                generated_reason_ids.append(token_id)
                reasoning_log_probs.append(step_log_probs[token_id].item())  # Save log_prob
                
                # Stream token if debug mode
                # if self.config.debug:
                #     token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                #     print(token_text, end="", flush=True)
                
                # Continue generation (unpack tuple from fast path)
                t_model_start = time.time()
                out_tuple = self.model(
                    next_token[:, None].to(dtype=torch.long),
                    past_key_values=past_key_values,
                    use_cache=True,
                    position_ids=cur_pos,
                )
                logits, past_key_values = out_tuple[0], out_tuple[1]
                timing["model_forward_time"] += time.time() - t_model_start
                
                # TIMING: Track python overhead (everything except model forward)
                timing["python_overhead"] += time.time() - t_token_start - (time.time() - t_model_start)
                cur_pos += 1
            
            # TIMING: End reasoning loop
            timing["reasoning_loop"] = time.time() - t_reasoning_start
            timing["reasoning_tokens"] = len(generated_reason_ids)
            
            # Print newline after streaming
            if self.config.debug:
                print()  # Newline after streaming
            
            # 4. CHECK: Generative (SPEAK) or Discrete (ranking) turn?
            is_generative_turn = actions[0].type == ActionType.SPEAK
            
            # 5. Append </think>\n\nAction: [Speak:] prefix AND store log probs
            end_think_token_ids = self.tokenizer.encode("\n</think>", add_special_tokens=False)
            if is_generative_turn:
                action_prefix_token_ids = self.tokenizer.encode("\n\nTell:", add_special_tokens=False)
            else:
                action_prefix_token_ids = self.tokenizer.encode("\n\nAction:", add_special_tokens=False)
            
            # if self.config.debug:
            #     print("🎯 Appending prefix: ", end="", flush=True)
            
            # TIMING: Prefix append
            t_prefix_start = time.time()
            prefix_ids = end_think_token_ids + action_prefix_token_ids
            prefix_log_probs = []  # Store log_probs for prefix
            for tid in prefix_ids:
                # if self.config.debug:
                #     token_text = self.tokenizer.decode([tid], skip_special_tokens=False)
                #     print(repr(token_text), end=" ", flush=True)
                tok = torch.tensor([[tid]], dtype=torch.long, device=device)
                out_tuple = self.model(
                    tok,
                    past_key_values=past_key_values,
                    use_cache=True,
                    position_ids=cur_pos,
                )
                current_logits = out_tuple[0][0, -1, :]
                
                # Get log_prob for this forced token
                step_log_probs = F.log_softmax(current_logits, dim=-1)
                prefix_log_probs.append(step_log_probs[tid].item())
                
                past_key_values = out_tuple[1]
                cur_pos += 1
            
            # if self.config.debug:
            #     print()  # Newline after prefix
            
            timing["prefix_append"] = time.time() - t_prefix_start
            
            last_logits = current_logits
            prefix_end_pos = cur_pos.item()
            
            # CRITICAL: Save the prefix cache before branching
            prefix_cache = past_key_values
            
            if is_generative_turn:
                # --- PATH A: GENERATIVE PPO (Speak Freely) ---
                if self.config.debug:
                    print("🗣️ [Generative Turn] Speaking freely...")
                
                generated_action_ids = []
                action_log_probs_list = []
                current_logits_branch = last_logits
                action_kv_cache = prefix_cache
                cache_position_act = torch.tensor([prefix_end_pos], device=device)
                
                # Generate up to 100 tokens for speech
                for _ in range(100):
                    step_logits = current_logits_branch
                    
                    # Apply sampling (temp, top_k, top_p)
                    if temperature != 1.0:
                        step_logits = step_logits / max(temperature, 1e-6)
                    
                    if top_k and top_k > 0 and top_k < step_logits.numel():
                        kth_vals, _ = torch.topk(step_logits, top_k)
                        min_keep = kth_vals[-1]
                        step_logits = torch.where(
                            step_logits < min_keep,
                            torch.tensor(float('-inf'), device=device, dtype=step_logits.dtype),
                            step_logits
                        )
                    
                    if top_p and top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(step_logits, descending=True)
                        sorted_probs = torch.softmax(sorted_logits, dim=-1)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                        sorted_mask = cumulative_probs > top_p
                        if sorted_mask.any():
                            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                            sorted_mask[..., 0] = False
                        sorted_logits = torch.where(
                            sorted_mask,
                            torch.tensor(float('-inf'), device=device, dtype=sorted_logits.dtype),
                            sorted_logits
                        )
                        step_logits = torch.full_like(step_logits, float('-inf'))
                        step_logits.scatter_(0, sorted_indices, sorted_logits)
                    
                    # Sample and get log prob
                    step_log_probs = F.log_softmax(step_logits, dim=-1)
                    probs = torch.softmax(step_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    token_id = int(next_token.item())
                    
                    if token_id == self.tokenizer.eos_token_id:
                        break
                    
                    generated_action_ids.append(token_id)
                    action_log_probs_list.append(step_log_probs[token_id].item())
                    
                    next_output_tuple = self.model(
                        next_token[:, None].to(dtype=torch.long),
                        past_key_values=action_kv_cache,
                        use_cache=True,
                        position_ids=cache_position_act,
                    )
                    current_logits_branch = next_output_tuple[0][0, -1, :]
                    action_kv_cache = next_output_tuple[1]
                    cache_position_act += 1
                
                generated_action_text = self.tokenizer.decode(generated_action_ids, skip_special_tokens=True)
                
                # Update the action object - MUST set target_message for SPEAK actions
                # because set_stories() reads from target_message to set command_perspective
                actions[0].target_message = generated_action_text
                actions[0].command_perspective = generated_action_text
                chosen_idx = 0
                chosen_per_token_log_probs = action_log_probs_list
            
            else:
                # --- PATH B: DISCRETE PPO (Ranking Actions) ---
                if self.config.debug:
                    print(f"📋 [Discrete Turn] Ranking {len(actions)} actions...")
                
                selection_scores = []
                all_per_token_log_probs = []
                
                for action in actions:
                    # CRITICAL: Clone the prefix cache for this branch
                    # This prevents subsequent actions from seeing KV state
                    # modified by previous actions
                    # Deep-clone the tuple of (Key, Value) tensor pairs for each layer
                    
                    # TIMING: KV cache cloning
                    t_clone_start = time.time()
                    action_kv_cache = tuple(
                        (k.clone(), v.clone()) for k, v in prefix_cache
                    )
                    clone_time = time.time() - t_clone_start
                    timing["kv_clone_total"] += clone_time
                    timing["kv_clone_count"] += 1
                    action_text = action.command_perspective
                    action_tokens = self.tokenizer(action_text, add_special_tokens=False, return_tensors="pt")
                    action_token_ids = action_tokens.input_ids[0].to(device)
                
                    # Print action header if debug mode
                    # if self.config.debug:
                    #     print(f"\n  📝 Action: {action_text[:60]}..." if len(action_text) > 60 else f"\n  📝 Action: {action_text}")
                    #     print(f"     Streaming tokens: ", end="", flush=True)
                    
                    num_words = len(action_text.split())
                    if len(action_token_ids) == 0:
                        selection_scores.append(float('-inf'))
                        all_per_token_log_probs.append([])
                        continue
                    
                    # Teacher forcing through action tokens
                    raw_token_log_probs = []
                    total_log_prob = 0.0
                    current_logits_branch = last_logits
                    cache_position_act = torch.tensor([prefix_end_pos], device=device)
                    
                    # TIMING: Action token processing
                    t_action_start = time.time()
                    for i, token_id in enumerate(action_token_ids):
                        t_action_token_start = time.time()
                        
                        # Calculate log_probs directly for numerical stability
                        step_log_probs = F.log_softmax(current_logits_branch, dim=-1)
                        log_prob = step_log_probs[token_id].item()
                        
                        # Stream action token if debug mode
                        # if self.config.debug:
                        #     token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                        #     print(f"{token_text}(log_p={log_prob:.3f}) ", end="", flush=True)
                        raw_token_log_probs.append(log_prob)
                        total_log_prob += log_prob
                        
                        if i < len(action_token_ids) - 1:
                            t_action_model_start = time.time()
                            next_output_tuple = self.model(
                                token_id.unsqueeze(0).unsqueeze(0),
                                past_key_values=action_kv_cache,
                                use_cache=True,
                                position_ids=cache_position_act,
                            )
                            current_logits_branch = next_output_tuple[0][0, -1, :]
                            action_kv_cache = next_output_tuple[1]
                            timing["model_forward_time"] += time.time() - t_action_model_start
                            timing["python_overhead"] += time.time() - t_action_token_start - (time.time() - t_action_model_start)
                            cache_position_act += 1
                        
                        timing["action_tokens_processed"] += 1
                    
                    # TIMING: End action processing
                    timing["action_generation"] += time.time() - t_action_start
                    
                    # 1. Calculate SELECTION score (word-normalized)
                    word_normalized_log_prob = total_log_prob / max(num_words, 1)
                    selection_scores.append(word_normalized_log_prob)
                    
                    # if self.config.debug:
                    #     print(f"→ word_norm_log_prob={word_normalized_log_prob:.4f}")
                    # 2. Store the TRAINING signal (raw per-token log probs)
                    all_per_token_log_probs.append(raw_token_log_probs)
                
                # Select action using word-normalized scores
                # Handle edge case: if all scores are -inf, use uniform distribution
                scores_tensor = torch.tensor(selection_scores, device=device)
                if torch.all(torch.isinf(scores_tensor)):
                    action_probs = torch.ones(len(actions), device=device) / len(actions)
                else:
                    action_probs = torch.softmax(scores_tensor, dim=-1)
                chosen_idx = int(torch.multinomial(action_probs, num_samples=1).item())
                chosen_per_token_log_probs = all_per_token_log_probs[chosen_idx]
            
            # Combine all log_probs for PPO update
            full_generation_log_probs = reasoning_log_probs + prefix_log_probs + chosen_per_token_log_probs
            output_token_count = len(full_generation_log_probs)
            
            # Extract reasoning text
            reasoning = self.tokenizer.decode(generated_reason_ids)
            
            if self.config.debug:
                print(f"\n{'='*60}")
                print("🎯 ACTION GENERATION")
                print(f"{'='*60}")
                print(f"Generated {len(generated_reason_ids)} reasoning tokens")
                print(f"Reasoning: {reasoning[:200]}..." if len(reasoning) > 200 else f"Reasoning: {reasoning}")
                if not is_generative_turn:
                    print("\nAction probabilities:")
                    for idx, (action, prob) in enumerate(zip(actions, action_probs)):
                        marker = "👉" if idx == chosen_idx else "  "
                        print(f"{marker} [{idx}] {action.command_perspective[:60]}: {prob.item():.4f}")
                print(f"\nTotal generation tokens: {len(full_generation_log_probs)}")
                print(f"Input tokens: {input_token_count}, Output tokens: {output_token_count}")
                
                # TIMING: Print detailed timing breakdown
                total_time = timing["prefill"] + timing["reasoning_loop"] + timing["prefix_append"] + timing["kv_clone_total"] + timing["action_generation"]
                print(f"\n{'─'*60}")
                print("⏱️  TIMING BREAKDOWN")
                print(f"{'─'*60}")
                print(f"Prefill:              {timing['prefill']*1000:8.2f}ms")
                print(f"Reasoning loop:       {timing['reasoning_loop']*1000:8.2f}ms ({timing['reasoning_tokens']} tokens)")
                if timing['reasoning_tokens'] > 0:
                    print(f"  ├─ ms/token:        {timing['reasoning_loop']*1000/timing['reasoning_tokens']:8.2f}ms")
                    print(f"  ├─ tok/s:           {timing['reasoning_tokens']/max(timing['reasoning_loop'],1e-6):8.1f}")
                print(f"Prefix append:        {timing['prefix_append']*1000:8.2f}ms")
                if not is_generative_turn:
                    print(f"KV cache cloning:     {timing['kv_clone_total']*1000:8.2f}ms ({timing['kv_clone_count']} clones)")
                    if timing['kv_clone_count'] > 0:
                        print(f"  ├─ ms/clone:        {timing['kv_clone_total']*1000/timing['kv_clone_count']:8.2f}ms")
                    print(f"Action generation:    {timing['action_generation']*1000:8.2f}ms ({timing['action_tokens_processed']} tokens)")
                    if timing['action_tokens_processed'] > 0:
                        print(f"  ├─ ms/token:        {timing['action_generation']*1000/timing['action_tokens_processed']:8.2f}ms")
                print(f"{'─'*60}")
                print(f"Model forward time:   {timing['model_forward_time']*1000:8.2f}ms")
                print(f"Python overhead:      {timing['python_overhead']*1000:8.2f}ms")
                print(f"Total time:           {total_time*1000:8.2f}ms")
                if output_token_count > 0:
                    print(f"Overall tok/s:        {output_token_count/max(total_time,1e-6):8.1f}")
                print(f"{'='*60}\n")
            
            # Return action_probs for proper entropy computation
            # For generative turns, there's only one "action" (the generated text)
            action_probs_list = action_probs.cpu().tolist() if not is_generative_turn else [1.0]
            
            return chosen_idx, full_generation_log_probs, reasoning, input_token_count, output_token_count, action_probs_list
    
    def generate_action_batch( # MARK: .    GENERATE BATCH
        self,
        batch_conversations: List[List[Dict[str, str]]],
        batch_actions_list: List[List[Action]],
    ) -> List[Tuple[int, List[float], str, int, int]]:
        """
        BATCH GENERATION for vectorized environment stepping.
        
        Generate actions for a batch of environments simultaneously.
        Uses left-padding for decoder-only models and handles variable-length reasoning.
        
        Args:
            batch_conversations: List of conversations, one per environment
            batch_actions_list: List of action lists, one per environment
            
        Returns:
            List of (action_idx, generation_log_probs, reasoning, input_tokens, output_tokens)
            for each environment in the batch
        """
        self.model.eval()
        device = self.model.device
        batch_size = len(batch_conversations)
        
        if batch_size == 0:
            return []
        
        # Timing
        t_batch_start = time.time()
        timing = {
            "tokenization": 0.0,
            "prefill": 0.0,
            "reasoning_loop": 0.0,
            "reasoning_tokens": 0,
            "action_selection": 0.0,
            "action_forward_passes": 0,
        }
        
        # 1. Apply chat template and tokenize with LEFT padding (already set in __init__)
        t_tok_start = time.time()
        input_texts = [
            self.tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=True
            ) for conv in batch_conversations
        ]
        
        # Tokenize as batch with left padding
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length
        ).to(device)
        
        input_lengths = (inputs.attention_mask == 1).sum(dim=1).tolist()
        timing["tokenization"] = time.time() - t_tok_start
        
        # 2. Prefill batch
        t_prefill_start = time.time()
        with torch.no_grad():
            prompt_len = inputs.input_ids.shape[1]
            cache_position = torch.arange(prompt_len, device=device).unsqueeze(0).expand(batch_size, -1)
            
            # Prefill all prompts
            out = self.model(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                cache_position=cache_position,
                past_key_values=None,
                use_cache=True,
            )
            logits = out.logits
            past_key_values = out.past_key_values
            timing["prefill"] = time.time() - t_prefill_start
            
            # Get </think> token
            think_close_token = self.tokenizer.encode("</think>", add_special_tokens=False)[0]
            
            # Get generation config for sampling
            gen_cfg = getattr(self.model, "generation_config", None)
            temperature = float(getattr(gen_cfg, "temperature", 1.0) or 1.0)
            top_p = float(getattr(gen_cfg, "top_p", 1.0) or 1.0)
            top_k = int(getattr(gen_cfg, "top_k", 0) or 0)
            
            # 3. Generate reasoning tokens for batch until all generate </think>
            # Track which sequences are still generating
            t_reasoning_start = time.time()
            active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            batch_reasoning_ids = [[] for _ in range(batch_size)]
            batch_reasoning_log_probs = [[] for _ in range(batch_size)]
            cur_pos = torch.tensor([prompt_len] * batch_size, device=device)
            
            for _ in range(self.config.max_reasoning_tokens):
                if not active_mask.any():
                    break
                
                # Get logits for the last position (works for both prefill and KV-cache steps)
                step_logits = logits[:, -1, :].clone()
                
                # Apply sampling (temperature, top_k, top_p)
                if temperature != 1.0:
                    step_logits = step_logits / max(temperature, 1e-6)
                
                if top_k and top_k > 0:
                    for i in range(batch_size):
                        if not active_mask[i]:
                            continue
                        if top_k < step_logits[i].numel():
                            kth_vals, _ = torch.topk(step_logits[i], top_k)
                            min_keep = kth_vals[-1]
                            step_logits[i] = torch.where(
                                step_logits[i] < min_keep,
                                torch.tensor(float('-inf'), device=device, dtype=step_logits.dtype),
                                step_logits[i]
                            )
                
                if top_p and top_p < 1.0:
                    for i in range(batch_size):
                        if not active_mask[i]:
                            continue
                        sorted_logits, sorted_indices = torch.sort(step_logits[i], descending=True)
                        sorted_probs = torch.softmax(sorted_logits, dim=-1)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                        sorted_mask = cumulative_probs > top_p
                        if sorted_mask.any():
                            sorted_mask[1:] = sorted_mask[:-1].clone()
                            sorted_mask[0] = False
                        sorted_logits = torch.where(
                            sorted_mask,
                            torch.tensor(float('-inf'), device=device, dtype=sorted_logits.dtype),
                            sorted_logits
                        )
                        step_logits[i] = torch.full_like(step_logits[i], float('-inf'))
                        step_logits[i].scatter_(0, sorted_indices, sorted_logits)
                
                # Sample tokens
                step_log_probs = F.log_softmax(step_logits, dim=-1)
                probs = torch.softmax(step_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                
                # Update only active sequences
                for i in range(batch_size):
                    if not active_mask[i]:
                        continue
                    
                    token_id = int(next_tokens[i].item())
                    
                    if token_id == think_close_token:
                        active_mask[i] = False
                        continue
                    
                    batch_reasoning_ids[i].append(token_id)
                    batch_reasoning_log_probs[i].append(step_log_probs[i, token_id].item())
                
                # Continue generation for active sequences
                # For inactive ones, we pad with eos tokens
                next_tokens_padded = torch.where(
                    active_mask,
                    next_tokens,
                    torch.tensor(self.tokenizer.eos_token_id, device=device)
                )
                
                out_tuple = self.model(
                    next_tokens_padded.unsqueeze(1),
                    past_key_values=past_key_values,
                    use_cache=True,
                    position_ids=cur_pos.unsqueeze(1),
                )
                logits = out_tuple[0]
                past_key_values = out_tuple[1]
                cur_pos += 1
            
            timing["reasoning_loop"] = time.time() - t_reasoning_start
            timing["reasoning_tokens"] = sum(len(ids) for ids in batch_reasoning_ids)
            
            # 4. Process action selection for each item in batch
            t_action_start = time.time()
            # Since different items may have different action types (generative vs discrete),
            # we need to handle them individually after reasoning
            
            results = []
            
            for batch_idx in range(batch_size):
                actions = batch_actions_list[batch_idx]
                reasoning_ids = batch_reasoning_ids[batch_idx]
                reasoning_log_probs = batch_reasoning_log_probs[batch_idx]
                
                # Decode reasoning
                reasoning = self.tokenizer.decode(reasoning_ids)
                
                # Append prefix: \n</think>\n\nAction: or \n</think>\n\nTell:
                is_generative_turn = actions[0].type == ActionType.SPEAK
                
                end_think_token_ids = self.tokenizer.encode("\n</think>", add_special_tokens=False)
                if is_generative_turn:
                    action_prefix_token_ids = self.tokenizer.encode("\n\nTell:", add_special_tokens=False)
                else:
                    action_prefix_token_ids = self.tokenizer.encode("\n\nAction:", add_special_tokens=False)
                
                prefix_ids = end_think_token_ids + action_prefix_token_ids
                prefix_log_probs = []
                
                # Build full sequence for fresh forward pass (avoids Unsloth KV cache batch size issues)
                # Get the original input tokens for this item (without left padding)
                orig_input_ids = inputs.input_ids[batch_idx]
                orig_mask = inputs.attention_mask[batch_idx]
                # Remove left padding
                non_pad_mask = orig_mask == 1
                orig_input_ids = orig_input_ids[non_pad_mask]
                
                # Build: input + reasoning + prefix
                reasoning_tensor = torch.tensor(reasoning_ids, dtype=torch.long, device=device)
                prefix_tensor = torch.tensor(prefix_ids, dtype=torch.long, device=device)
                full_seq = torch.cat([orig_input_ids, reasoning_tensor, prefix_tensor]).unsqueeze(0)
                
                # Fresh forward pass without KV cache
                with torch.no_grad():
                    out = self.model(full_seq, use_cache=True)
                    item_past_kv = out.past_key_values
                    last_logits = out.logits[0, -1, :]
                    timing["action_forward_passes"] += 1
                
                # Compute log probs for prefix tokens
                prefix_start = len(orig_input_ids) + len(reasoning_ids)
                for i, tid in enumerate(prefix_ids):
                    pos = prefix_start + i - 1  # -1 because we predict from previous position
                    if pos >= 0:
                        step_log_probs = F.log_softmax(out.logits[0, pos, :], dim=-1)
                        prefix_log_probs.append(step_log_probs[tid].item())
                
                prefix_end_pos = full_seq.shape[1]
                
                # Now handle generative vs discrete
                if is_generative_turn:
                    # Generative: generate freely
                    generated_action_ids = []
                    action_log_probs_list = []
                    action_kv_cache = item_past_kv
                    cache_position_act = torch.tensor([prefix_end_pos], device=device)
                    current_logits_branch = last_logits
                    
                    for _ in range(100):
                        step_logits = current_logits_branch
                        
                        if temperature != 1.0:
                            step_logits = step_logits / max(temperature, 1e-6)
                        
                        # Apply sampling...
                        step_log_probs = F.log_softmax(step_logits, dim=-1)
                        probs = torch.softmax(step_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        token_id = int(next_token.item())
                        
                        if token_id == self.tokenizer.eos_token_id:
                            break
                        
                        generated_action_ids.append(token_id)
                        action_log_probs_list.append(step_log_probs[token_id].item())
                        
                        next_output_tuple = self.model(
                            next_token[:, None].to(dtype=torch.long),
                            past_key_values=action_kv_cache,
                            use_cache=True,
                            position_ids=cache_position_act,
                        )
                        current_logits_branch = next_output_tuple[0][0, -1, :]
                        action_kv_cache = next_output_tuple[1]
                        cache_position_act += 1
                    
                    generated_action_text = self.tokenizer.decode(generated_action_ids, skip_special_tokens=True)
                    # MUST set target_message for SPEAK actions (set_stories() reads from it)
                    actions[0].target_message = generated_action_text
                    actions[0].command_perspective = generated_action_text
                    chosen_idx = 0
                    chosen_per_token_log_probs = action_log_probs_list
                    action_probs = [1.0]  # Generative turn = deterministic single action
                    generated_message = generated_action_text  # Return for caller to use
                
                else:
                    # Discrete: rank actions
                    selection_scores = []
                    all_per_token_log_probs = []
                    
                    for action in actions:
                        action_kv_cache = tuple(
                            (k.clone(), v.clone()) for k, v in item_past_kv
                        )
                        action_text = action.command_perspective
                        action_tokens = self.tokenizer(action_text, add_special_tokens=False, return_tensors="pt")
                        action_token_ids = action_tokens.input_ids[0].to(device)
                        
                        num_words = len(action_text.split())
                        if len(action_token_ids) == 0:
                            selection_scores.append(float('-inf'))
                            all_per_token_log_probs.append([])
                            continue
                        
                        raw_token_log_probs = []
                        total_log_prob = 0.0
                        current_logits_branch = last_logits
                        cache_position_act = torch.tensor([prefix_end_pos], device=device)
                        
                        for i, token_id in enumerate(action_token_ids):
                            step_log_probs = F.log_softmax(current_logits_branch, dim=-1)
                            log_prob = step_log_probs[token_id].item()
                            raw_token_log_probs.append(log_prob)
                            total_log_prob += log_prob
                            
                            if i < len(action_token_ids) - 1:
                                next_output_tuple = self.model(
                                    token_id.unsqueeze(0).unsqueeze(0),
                                    past_key_values=action_kv_cache,
                                    use_cache=True,
                                    position_ids=cache_position_act,
                                )
                                current_logits_branch = next_output_tuple[0][0, -1, :]
                                action_kv_cache = next_output_tuple[1]
                                cache_position_act += 1
                        
                        word_normalized_log_prob = total_log_prob / max(num_words, 1)
                        selection_scores.append(word_normalized_log_prob)
                        all_per_token_log_probs.append(raw_token_log_probs)
                    
                    # Handle edge case: if all scores are -inf, use uniform distribution
                    scores_tensor = torch.tensor(selection_scores, device=device)
                    if torch.all(torch.isinf(scores_tensor)):
                        action_probs_tensor = torch.ones(len(actions), device=device) / len(actions)
                    else:
                        action_probs_tensor = torch.softmax(scores_tensor, dim=-1)
                    chosen_idx = int(torch.multinomial(action_probs_tensor, num_samples=1).item())
                    chosen_per_token_log_probs = all_per_token_log_probs[chosen_idx]
                    action_probs = action_probs_tensor.cpu().tolist()  # Convert to list for storage
                    generated_message = None  # Discrete actions don't generate message
                
                # Combine all log_probs
                full_generation_log_probs = reasoning_log_probs + prefix_log_probs + chosen_per_token_log_probs
                output_token_count = len(full_generation_log_probs)
                
                results.append((
                    chosen_idx,
                    full_generation_log_probs,
                    reasoning,
                    input_lengths[batch_idx],
                    output_token_count,
                    action_probs,  # Add action distribution for entropy
                    generated_message  # Generated text for SPEAK actions
                ))
            
            timing["action_selection"] = time.time() - t_action_start
            total_time = time.time() - t_batch_start
            
            # Print timing summary with VRAM
            vram_info = ""
            if torch.cuda.is_available():
                vram_gb = torch.cuda.memory_allocated() / 1024**3
                vram_info = f" | vram={vram_gb:.1f}GB"
            
            print(f"⏱️  BATCH (n={batch_size}): "
                  f"tok={timing['tokenization']*1000:.0f}ms "
                  f"prefill={timing['prefill']*1000:.0f}ms "
                  f"reason={timing['reasoning_loop']*1000:.0f}ms/{timing['reasoning_tokens']}tok "
                  f"action={timing['action_selection']*1000:.0f}ms/{timing['action_forward_passes']}fwd "
                  f"total={total_time*1000:.0f}ms{vram_info}", flush=True)
            
            return results
    
    def _build_global_state(self, history_items: List[History], current_turn_index: int) -> str: # MARK: .      GLOBAL
        """
        Build global state from History.to_text() + ONLY LAST reasoning per player.
        
        OPTIMIZATION: Instead of including all reasoning from all turns,
        we only keep the most recent reasoning for each player. This reduces
        token count while preserving critical information.
        """
        global_view = "=== Global Game State ===\n\n"
        
        # Track last reasoning for each player
        last_reasoning_per_player = {}
        
        # First pass: collect game history and track last reasoning per player
        for i in range(min(current_turn_index + 1, len(history_items))):
            hist = history_items[i]
            
            # Track last reasoning for this player
            if hasattr(hist, 'llm_cot') and hist.llm_cot:
                player_name = hist.action_taken.player_name
                last_reasoning_per_player[player_name] = hist.llm_cot[:3500]
            
            # Add game history (WITHOUT reasoning blocks)
            # Use to_text() for clean, ANSI-free representation for value head
            global_view += f"<turn_{i}> {hist.to_text()} </turn_{i}>\n"
        
        # Second pass: append only the LAST reasoning for each player
        if last_reasoning_per_player:
            global_view += "\n=== Player Reasoning (Most Recent) ===\n"
            for player_name, reasoning in last_reasoning_per_player.items():
                global_view += f"<player_{player_name}_reasoning>{reasoning}</player_{player_name}_reasoning>\n"
        
        return global_view
    
    def _determine_winner_from_end_reason(self, end_reason: EndGameReason) -> PlayerRole:
        """Map EndGameReason to winner role"""
        if end_reason in [EndGameReason.NO_ACTIONS_LEFT, 
                          EndGameReason.NO_IMPOSTORS_LEFT, 
                          EndGameReason.ALL_TASKS_DONE]:
            return PlayerRole.CREWMATE
        elif end_reason == EndGameReason.TOO_SMALL_NUMBER_OF_CREWMATES_LEFT:
            return PlayerRole.IMPOSTOR
        else:
            raise ValueError(f"Unknown end reason: {end_reason}")
    
    def _save_trajectory_to_disk(self, engine: GameEngine, turns_data: List, winner_role: PlayerRole, end_reason: EndGameReason):
        """Save completed trajectory to disk for later analysis using GameJSONEncoder format"""
        if not self.config.save_trajectories:
            return
            
        try:
            import os
            from datetime import datetime
            
            # Get iteration from trainer (use +1 to match checkpoint naming convention)
            iteration = getattr(self.trainer, 'iteration', 0) if self.trainer else 0
            iteration_num = iteration + 1  # Match checkpoint folder naming (1-indexed)
            
            # Create iteration-specific trajectories directory
            traj_dir = os.path.join(self.config.checkpoint_dir, f"iteration_{iteration_num}", "trajectories")
            os.makedirs(traj_dir, exist_ok=True)
            
            # Generate filename with timestamp + microseconds to avoid collision
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"traj_{timestamp}_{winner_role.name}_{len(turns_data)}turns.json"
            filepath = os.path.join(traj_dir, filename)
            
            # Save using the same format as GameEngine
            with open(filepath, 'w') as f:
                json_str = json.dumps(
                    (engine.history, engine.players, engine.game_config),
                    indent=2,
                    cls=GameJSONEncoder
                )
                f.write(json_str)
            
            if self.config.debug:
                print(f"💾 Trajectory saved to: {filepath}")
                
        except Exception as e:
            print(f"⚠️  Warning: Failed to save trajectory: {e}")
    
    def collect_trajectory(self) -> Optional[Trajectory]: # MARK: .      COLLECT
        """
        Run one complete game using self-play and collect trajectory.
        Returns None if game fails to complete properly.
        """
        start_time = time.time()
        
        if self.config.debug:
            print(f"\n{'#'*80}")
            print("🎮 STARTING NEW GAME")
            print(f"{'#'*80}\n")
        
        engine = GameEngine(self.game_config)
        turns_data = []
        
        player_roles = {p.name: p.role for p in engine.players}
        
        if self.config.debug:
            print(f"Players: {list(player_roles.keys())}")
            print(f"Roles: {list(player_roles.values())}\n")
        
        max_turns = 200
        turn_count = 0
        
        while turn_count < max_turns:
            turn_context = engine.get_turn_context()
            if turn_context is None or turn_context[0] is None:
                break
            
            turn_history, actions_player_can_take, conversation, _ = turn_context
            
            if not actions_player_can_take:
                raise ValueError("No actions available for player")
            
            current_player_name = turn_history.action_taken.player_name
            current_player_role = player_roles[current_player_name]
            
            if self.config.debug:
                print(f"\n{'─'*80}")
                print(f"Turn {turn_count}: {current_player_name} ({current_player_role.name})")
                print(f"Available actions: {len(actions_player_can_take)}")
                print(f"Conversation length: {len(str(conversation))} chars")
                print(f"{'─'*80}")
            
            # Generate action using policy
            try:
                turn_gen_start = time.time()
                action_idx, log_prob, reasoning, input_tokens, output_tokens, action_probs = self._generate_action_with_policy(
                    conversation,
                    actions_player_can_take
                )
                turn_gen_time = time.time() - turn_gen_start
                chosen_action = actions_player_can_take[action_idx]
                
                if self.config.debug:
                    print(f"✅ Action selected: {chosen_action.command_perspective}")
                    print(f"⏱️  Generation time: {turn_gen_time:.3f}s")
            except Exception as e:
                if self.config.debug:
                    print(f"❌ Error generating action: {e}")
                    import traceback
                    traceback.print_exc()
                return None
            
            # Build global state for critic
            global_state = self._build_global_state(engine.history, len(engine.history))
            
            # Store turn data
            turn_data = TurnData(
                player_name=current_player_name,
                player_role=current_player_role,
                conversation=conversation.copy(),
                actions=actions_player_can_take.copy(),
                chosen_action_idx=action_idx,
                chosen_action=chosen_action,
                reasoning=reasoning,
                generation_log_probs=log_prob,
                action_probs=action_probs,  # Store for entropy computation
                global_state_repr=global_state,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                generation_time=turn_gen_time,
            )
            turns_data.append(turn_data)
            
            # Report progress if callback provided
            if hasattr(self, 'progress_callback') and self.progress_callback:
                self.progress_callback(turn_count + 1, max_turns)
            
            # Execute action
            game_over, end_reason = engine.step(
                turn_history,
                chosen_action,
                llm_response=chosen_action.command_perspective,
                llm_cot=f"<think>{reasoning}</think>",
                token_usage={},
                pre_discussion_votes=None
            )
            
            if game_over and end_reason:
                winner_role = self._determine_winner_from_end_reason(end_reason)
                
                # Save trajectory to disk for later analysis
                self._save_trajectory_to_disk(engine, turns_data, winner_role, end_reason)
                
                # Calculate statistics
                collection_time = time.time() - start_time
                total_input_tokens = sum(turn.input_tokens for turn in turns_data)
                total_output_tokens = sum(turn.output_tokens for turn in turns_data)
                
                if self.config.debug:
                    print(f"\n{'#'*80}")
                    print("🏁 GAME OVER")
                    print(f"{'#'*80}")
                    print(f"Turns: {turn_count + 1}")
                    print(f"End reason: {end_reason.name}")
                    print(f"Winner: {winner_role.name}")
                    print(f"Collection time: {collection_time:.2f}s")
                    print(f"Input tokens: {total_input_tokens}, Output tokens: {total_output_tokens}")
                    print(f"{'#'*80}\n")
                
                trajectory = Trajectory(
                    turns=turns_data,
                    winner_role=winner_role,
                    end_reason=end_reason,
                    collection_time=collection_time,
                    total_input_tokens=total_input_tokens,
                    total_output_tokens=total_output_tokens,
                )
                
                return trajectory
            
            turn_count += 1
        
        if self.config.debug:
            print(f"Game exceeded max turns ({max_turns})")
        return None


# ============================================================================
# MARK: MAPPO TRAINER
# ============================================================================

class MAPPOTrainer:
    """Main trainer implementing MAPPO algorithm"""
    
    def _print_vram_summary(self, context: str):
        """Helper to print a VRAM summary for debugging."""
        if not self.config.debug or not torch.cuda.is_available():
            return
        
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
        
        print(f"\n{'='*20} VRAM @ {context} {'='*20}")
        print(f"  Allocated:        {allocated:5.2f} GB")
        print(f"  Reserved (Cache): {reserved:5.2f} GB")
        print(f"  Max Allocated:    {max_allocated:5.2f} GB")
        print(f"{'='*50}\n")
        
        # Reset max for the next step
        torch.cuda.reset_peak_memory_stats()
    
    def __init__(self, config: MAPPOConfig):
        self.config = config
        self._set_seeds(config.seed)
        
        # Initialize wandb (with resume support if needed)
        if config.resume_from:
            # Resume existing run if resuming from checkpoint
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=config.__dict__,
                resume="allow",  # Resume if run with same name exists
                id=f"{config.wandb_run_name}_{config.seed}"  # Use consistent run ID
            )
        else:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=config.__dict__
            )
        
        # Load model with LoRA (SINGLE INSTANCE)
        print(f"\n{'='*80}")
        print(f"🔧 INITIALIZING MODEL")
        print(f"{'='*80}")
        print(f"Model: {config.model_name}")
        print(f"Max sequence length: {config.max_seq_length}")
        print(f"LoRA rank: {config.lora_rank}")
        print(f"4-bit quantization: {config.load_in_4bit}")
        print(f"{'='*80}\n", flush=True)
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            config.model_name,
            load_in_4bit=config.load_in_4bit,
            max_seq_length=config.max_seq_length,
            attn_implementation="flash_attention_2",
        )
        print("✅ Base model loaded")
        
        # Load checkpoint or apply fresh LoRA adapters
        if config.load_model:
            self._load_model_checkpoint(config.load_model)
        else:
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=config.lora_rank,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0.0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=config.seed,
            )
            print("✅ LoRA adapters applied")
        
        # Compile model for optimized inference
        # print("Compiling model with torch.compile()...")
        # try:
        #     self.model = torch.compile(self.model, mode="reduce-overhead")
        #     print("✅ Model compiled successfully")
        # except Exception as e:
        #     print(f"⚠️  torch.compile() failed (will continue without): {e}")
        
        # Initialize value head in float32 for numerical stability
        # PyTorch will automatically upcast float16 inputs to float32
        hidden_size = self.model.config.hidden_size
        self.value_head = ValueHead(hidden_size).to(device=self.model.device, dtype=torch.float32)
        print(f"✅ Value head initialized (hidden_size={hidden_size}, dtype=torch.float32)")
        
        # Load value head checkpoint if specified
        if config.load_head:
            self._load_value_head_checkpoint(config.load_head)
        
        # Optimizers
        self.actor_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.actor_lr,
            weight_decay=0.01
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.value_head.parameters(),
            lr=config.critic_lr,
            weight_decay=0.01
        )
        print(f"✅ Optimizers initialized (actor_lr={config.actor_lr}, critic_lr={config.critic_lr})")
        
        # Initialize actor (shares same model instance and references trainer for iteration tracking)
        self.actor = MAPPOActor(
            model=self.model,
            trainer=self,  # Pass trainer reference for iteration tracking
            tokenizer=self.tokenizer,
            value_head=self.value_head,
            config=config
        )
        print("✅ MAPPO Actor initialized")
        
        # Create output directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        self.iteration = 0
        self.training_stats = []
        self.start_iteration = 0  # Track where to start training loop
        
        # Handle resume if specified
        if config.resume_from:
            self.start_iteration = self.load_checkpoint_for_resume(config.resume_from)
            self.iteration = self.start_iteration
        else:
            # Pretrain value head if enabled (skip if loading checkpoint or resuming)
            if config.pretrain_value_head and not config.load_head:
                self.pretrain_value_head()
    
    def _set_seeds(self, seed: int):
        """Set random seeds for full reproducibility (critical for scientific papers)"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Deterministic operations (trades speed for reproducibility)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        # Set environment variable for hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
    
    def pretrain_value_head(self): # MARK: .      PRETRAINING
        """Pre-train value head on existing game trajectories in data/ folder"""
        print(f"\n{'='*80}")
        print("PRETRAINING VALUE HEAD ON EXISTING GAMES")
        print(f"{'='*80}")
        
        data_files = list(Path(self.config.data_dir).glob("game_state_*.json"))
        print(f"Found {len(data_files)} game files")
        
        training_data = []
        
        for file_path in data_files:
            try:
                with open(file_path, 'r') as f:
                    history_items: List[History]
                    players: List[Player]
                    history_items, players, _ = json.load(f, object_hook=game_object_hook)
                
                # Determine outcome
                end_reason = get_end_game_reason(history_items, players)
                
                if end_reason is None:
                    raise ValueError(f"Game {file_path} ended without a clear winner")
                
                winner_role = self.actor._determine_winner_from_end_reason(end_reason)
                
                # Create training examples from each turn
                for i, hist in enumerate(history_items):
                    if hist.action_taken.player_name == "System":
                        continue
                    
                    player_name = hist.action_taken.player_name
                    player_role = next((p.role for p in players if p.name == player_name), None)
                    
                    # Build global state
                    global_state = self.actor._build_global_state(history_items, i)
                    
                    # Reward for this player
                    if end_reason == EndGameReason.NO_ACTIONS_LEFT:
                        reward = 0.3 if winner_role == player_role else -0.3
                    else:
                        reward = 1.0 if winner_role == player_role else -1.0
                    
                    training_data.append((global_state, reward))
                
            except Exception as e:
                if self.config.debug:
                    print(f"  Error loading {file_path.name}: {e}")
                continue
        
        print(f"Created {len(training_data)} training examples")
        
        if len(training_data) == 0:
            print("No training data available for value head pretraining")
            return
        
        # Analyze token counts before training
        print("\nAnalyzing token counts...")
        token_counts = []
        for global_state, _ in training_data:
            tokens = self.tokenizer(global_state, return_tensors="pt")
            num_tokens = tokens.input_ids.shape[1]
            token_counts.append(num_tokens)
        
        print(f"Token count statistics:")
        print(f"  Min: {min(token_counts)} tokens")
        print(f"  Max: {max(token_counts)} tokens")
        print(f"  Mean: {sum(token_counts)/len(token_counts):.1f} tokens")
        print(f"  Median: {sorted(token_counts)[len(token_counts)//2]} tokens")
        
        # Show examples that exceed max_seq_length
        max_len = self.config.max_seq_length
        exceeding = [tc for tc in token_counts if tc > max_len]
        if exceeding:
            print(f"\n⚠️  WARNING: {len(exceeding)}/{len(token_counts)} examples exceed max_seq_length={max_len}")
            print(f"  These will be truncated during training")
        print()
        
        # Train value head
        self.value_head.train()
        optimizer = torch.optim.Adam(self.value_head.parameters(), lr=self.config.pretrain_lr)
        
        # Create sorted list by token count for debugging
        training_data_with_tokens = [
            (token_counts[i], training_data[i][0], training_data[i][1]) 
            for i in range(len(training_data))
        ]
        # training_data_with_tokens.sort(key=lambda x: x[0])  # Sort by token count
        
        for epoch in range(self.config.pretrain_epochs):
            total_loss = 0.0
            random.shuffle(training_data_with_tokens)
            training_data_with_tokens = training_data_with_tokens[:self.config.pretrain_examples]
            
            
            for idx, (num_tokens, global_state, target_value) in enumerate(training_data_with_tokens):
                print(f"  Processing example {idx+1}/{len(training_data_with_tokens)}: {num_tokens} tokens")
                
                # Tokenize global state
                inputs = self.tokenizer(
                    global_state,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_seq_length
                ).to(self.model.device)
                
                # Get hidden states from model
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-1]
                
                # Predict value
                predicted_value = self.value_head(hidden_states).squeeze(-1)
                
                # MSE loss
                loss = F.mse_loss(
                    predicted_value, 
                    torch.tensor([target_value], device=self.model.device, dtype=predicted_value.dtype)
                )
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(training_data_with_tokens)
            print(f"  Epoch {epoch+1}/{self.config.pretrain_epochs}: Loss = {avg_loss:.4f}")
            
            # Log to wandb with explicit step to avoid auto-increment confusion
            wandb.log({
                "pretrain/epoch": epoch + 1,
                "pretrain/loss": avg_loss,
            }, step=epoch)
            
            # Save value head after each epoch
            pretrain_checkpoint_dir = Path(self.config.checkpoint_dir) / "pretrain"
            pretrain_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = pretrain_checkpoint_dir / f"value_head_epoch_{epoch+1}.pt"
            torch.save(self.value_head.state_dict(), checkpoint_path)
            print(f"  💾 Saved value head to: {checkpoint_path}")
        
        print("Value head pretraining complete!\n")
    
    def _compute_value(self, global_state_repr: str) -> torch.Tensor: # MARK: .      VALUE
        """Compute value estimate for a global state
        
        Note: This function does NOT change model mode. Caller is responsible for
        setting appropriate mode before calling. Uses torch.no_grad() for model
        forward pass regardless of mode.
        """
        inputs = self.tokenizer(
            global_state_repr,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length
        ).to(self.model.device)
        
        # Model forward without gradients (hidden states don't need gradients)
        # Don't change model mode - let caller control it
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1].detach()
        
        # CRITICAL: Delete outputs and inputs immediately to free VRAM
        del outputs, inputs
        
        # Explicitly enable gradients for the value_head pass
        with torch.enable_grad():
            value = self.value_head(hidden_states).squeeze(-1)
        
        # CRITICAL: Delete hidden_states after value computation
        del hidden_states
        
        return value
    
    def _compute_value_batch(
        self,
        global_state_reprs: List[str],
        batch_size: int = 16
    ) -> List[float]:  # MARK: .      VALUE BATCH
        """
        Batched value computation for efficient GAE.
        
        Args:
            global_state_reprs: List of global state strings
            batch_size: Number of states to process in parallel
            
        Returns:
            List of value estimates (floats)
        """
        device = self.model.device
        all_values = []
        total_batches = (len(global_state_reprs) + batch_size - 1) // batch_size
        start_time = time.time()
        
        for batch_idx, batch_start in enumerate(range(0, len(global_state_reprs), batch_size)):
            batch_states = global_state_reprs[batch_start:batch_start + batch_size]
            
            # Progress print every 5 batches or on last batch
            if batch_idx % 5 == 0 or batch_idx == total_batches - 1:
                elapsed = time.time() - start_time
                if batch_idx > 0:
                    eta = elapsed / (batch_idx + 1) * (total_batches - batch_idx - 1)
                    print(f"    Value batch {batch_idx + 1}/{total_batches} ({len(all_values)}/{len(global_state_reprs)} done) | {elapsed:.1f}s elapsed, ~{eta:.1f}s remaining", end="\r", flush=True)
                else:
                    print(f"    Value batch {batch_idx + 1}/{total_batches} ({len(all_values)}/{len(global_state_reprs)} done)", end="\r", flush=True)
            
            # Tokenize with left-padding for batch processing
            encoded = self.tokenizer(
                batch_states,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length
            ).to(device)
            
            with torch.no_grad():
                outputs = self.model(**encoded, output_hidden_states=True)
                # Get last hidden state for each sequence (last non-padded position)
                hidden_states = outputs.hidden_states[-1]  # [B, seq_len, hidden]
                
                # For left-padded sequences, the last token is always the last position
                last_hidden = hidden_states[:, -1, :]  # [B, hidden]
                
                del outputs, hidden_states
            
            # Compute values (no gradients needed for GAE)
            with torch.no_grad():
                values = self.value_head(last_hidden).squeeze(-1)  # [B]
                
            batch_values = values.cpu().tolist()
            if isinstance(batch_values, float):
                batch_values = [batch_values]
            all_values.extend(batch_values)
            
            del encoded, last_hidden, values
        
        print()  # Newline after progress
        return all_values
    
    def _compute_log_prob(
        self,
        conversation: List[Dict[str, str]],
        actions: List[Action],
        chosen_action_idx: int,
        reasoning: str,
        use_reference_model: bool = False
    ) -> torch.Tensor: # MARK: .      LOGPROB
        """
        GENERATIVE PPO: Re-computes RAW per-token log-probs for the
        ENTIRE generated sequence (reasoning + prefix + action).
        
        Returns:
            Tensor of shape [T] where T = num_generated_tokens
        """
        device = self.model.device
        
        # 1. Build CONTEXT
        input_text = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(device)
        context_ids = inputs.input_ids
        context_len = context_ids.shape[1]

        # 2. Build GENERATED sequence
        reasoning_tokens = self.tokenizer.encode(reasoning, add_special_tokens=False)
        end_think_tokens = self.tokenizer.encode("\n</think>", add_special_tokens=False)
        
        chosen_action = actions[chosen_action_idx]
        is_generative_turn = chosen_action.type == ActionType.SPEAK
        
        if is_generative_turn:
            action_prefix_tokens = self.tokenizer.encode("\n\nTell:", add_special_tokens=False)
        else:
            action_prefix_tokens = self.tokenizer.encode("\n\nAction:", add_special_tokens=False)
        
        action_text = chosen_action.command_perspective
        action_tokens = self.tokenizer.encode(action_text, add_special_tokens=False)

        # This is all the tokens that were "generated"
        generated_ids_list = reasoning_tokens + end_think_tokens + action_prefix_tokens + action_tokens
        generated_ids = torch.tensor([generated_ids_list], device=device)

        # 3. Concatenate
        full_input_ids = torch.cat([context_ids, generated_ids], dim=1)
        
        # ====================================================================
        # Print the sequence length that is about to be processed
        seq_len = full_input_ids.shape[1]
        # if self.config.debug:
        #     print(f"      [DEBUG _compute_log_prob] Processing seq_len: {seq_len}")
        # ====================================================================
        
        # 4. Forward pass
        if use_reference_model:
            with torch.no_grad(), self.model.disable_adapter():
                 outputs = self.model(input_ids=full_input_ids)
        else:
            # Respect the gradient context set by the caller
            # (torch.no_grad() for pre-computation, or model.train() for updates)
            outputs = self.model(input_ids=full_input_ids)
            
        # [1, C+G, V]
        logits = outputs.logits 

        # 5. Get log_probs for the GENERATED tokens
        
        # We need the logits that *predicted* the generated tokens.
        # These start from `context_len - 1` and go to the end.
        # [1, G, V] (G = number of generated tokens)
        generated_logits = logits[:, context_len-1:-1, :]
        
        # CRITICAL: Delete full logits tensor immediately to free VRAM
        # The generated_logits view will hold its own reference
        del logits, outputs
        
        # [1, G, V]
        generated_log_probs = F.log_softmax(generated_logits, dim=-1)
        
        # CRITICAL: Delete generated_logits after computing log_probs
        del generated_logits

        # [1, G, 1]
        generated_ids_expanded = generated_ids.unsqueeze(-1)
        
        # Gather the log_probs for the tokens that were actually chosen
        # [1, G, 1]
        chosen_token_log_probs = torch.gather(
            generated_log_probs, 
            dim=-1, 
            index=generated_ids_expanded
        )
        
        # CRITICAL: Delete generated_log_probs after gathering
        del generated_log_probs, generated_ids_expanded
        
        # Return the TENSOR of all per-token log-probs
        # Squeeze to shape [G]
        result = chosen_token_log_probs.squeeze()
        del chosen_token_log_probs
        
        return result
    
    def _compute_log_prob_batch(
        self,
        turns: List[TurnData],
        use_reference_model: bool = False,
        batch_size: int = 4
    ) -> List[torch.Tensor]:  # MARK: .      LOGPROB BATCH
        """
        Batched version of _compute_log_prob for efficient pre-computation.
        
        Processes multiple turns in parallel using left-padding.
        
        Args:
            turns: List of TurnData objects
            use_reference_model: If True, use base model without LoRA
            batch_size: Number of turns to process in each batch
            
        Returns:
            List of tensors, each of shape [G_i] where G_i = generated tokens for turn i
        """
        device = self.model.device
        results = []
        total_batches = (len(turns) + batch_size - 1) // batch_size
        start_time = time.time()
        
        for batch_idx, batch_start in enumerate(range(0, len(turns), batch_size)):
            batch_turns = turns[batch_start:batch_start + batch_size]
            
            # Progress print every 5 batches or on last batch
            if batch_idx % 5 == 0 or batch_idx == total_batches - 1:
                elapsed = time.time() - start_time
                if batch_idx > 0:
                    eta = elapsed / (batch_idx + 1) * (total_batches - batch_idx - 1)
                    print(f"    LogProb batch {batch_idx + 1}/{total_batches} ({len(results)}/{len(turns)} done) | {elapsed:.1f}s elapsed, ~{eta:.1f}s remaining", end="\r", flush=True)
                else:
                    print(f"    LogProb batch {batch_idx + 1}/{total_batches} ({len(results)}/{len(turns)} done)", end="\r", flush=True)
            
            # Build all sequences for this batch
            all_full_ids = []
            all_context_lens = []
            all_generated_lens = []
            
            for turn in batch_turns:
                # Build context
                input_text = self.tokenizer.apply_chat_template(
                    turn.conversation, tokenize=False, add_generation_prompt=True
                )
                context_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
                context_len = len(context_ids)
                
                # Build generated sequence
                reasoning_tokens = self.tokenizer.encode(turn.reasoning, add_special_tokens=False)
                end_think_tokens = self.tokenizer.encode("\n</think>", add_special_tokens=False)
                
                chosen_action = turn.actions[turn.chosen_action_idx]
                is_generative_turn = chosen_action.type == ActionType.SPEAK
                
                if is_generative_turn:
                    action_prefix_tokens = self.tokenizer.encode("\n\nTell:", add_special_tokens=False)
                else:
                    action_prefix_tokens = self.tokenizer.encode("\n\nAction:", add_special_tokens=False)
                
                action_tokens = self.tokenizer.encode(chosen_action.command_perspective, add_special_tokens=False)
                generated_ids = reasoning_tokens + end_think_tokens + action_prefix_tokens + action_tokens
                
                full_ids = context_ids + generated_ids
                all_full_ids.append(full_ids)
                all_context_lens.append(context_len)
                all_generated_lens.append(len(generated_ids))
            
            # Pad sequences (left-padding for decoder-only models)
            max_len = max(len(ids) for ids in all_full_ids)
            padded_ids = []
            attention_masks = []
            
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            
            for ids in all_full_ids:
                pad_len = max_len - len(ids)
                padded = [pad_token_id] * pad_len + ids
                mask = [0] * pad_len + [1] * len(ids)
                padded_ids.append(padded)
                attention_masks.append(mask)
            
            input_ids = torch.tensor(padded_ids, device=device)
            attention_mask = torch.tensor(attention_masks, device=device)
            
            # Forward pass
            if use_reference_model:
                with torch.no_grad(), self.model.disable_adapter():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            logits = outputs.logits  # [B, max_len, V]
            del outputs
            
            # Extract log probs for each turn in the batch
            for i, turn in enumerate(batch_turns):
                pad_len = max_len - len(all_full_ids[i])
                context_len = all_context_lens[i]
                generated_len = all_generated_lens[i]
                
                # Adjust indices for padding
                # The context starts at pad_len, generated tokens start at pad_len + context_len
                # Logits that predict generated tokens are at positions [pad_len + context_len - 1 : pad_len + context_len + generated_len - 1]
                start_idx = pad_len + context_len - 1
                end_idx = pad_len + context_len + generated_len - 1
                
                generated_logits = logits[i, start_idx:end_idx, :]  # [G, V]
                generated_log_probs = F.log_softmax(generated_logits, dim=-1)  # [G, V]
                
                # Get the actual generated token ids
                generated_ids = all_full_ids[i][context_len:]
                generated_ids_tensor = torch.tensor(generated_ids, device=device)
                
                # Gather log probs for chosen tokens
                chosen_log_probs = generated_log_probs[
                    torch.arange(generated_len, device=device),
                    generated_ids_tensor
                ]  # [G]
                
                results.append(chosen_log_probs)
            
            del logits, input_ids, attention_mask
        
        print()  # Newline after progress
        return results
    
    def _load_recent_trajectories_from_disk(self, num_trajectories: int, iteration: int = None) -> tuple[List[Trajectory], dict]: # MARK: .      LOAD RECENT
        """Load the most recent N trajectories from disk and reconstruct TurnData for PPO update.
        
        This is a DEBUG feature to resume from saved trajectories without re-collection.
        If iteration is specified, loads from that iteration's folder, otherwise from global trajectories.
        """
        print(f"\n{'='*80}")
        print(f"🔄 LOADING {num_trajectories} TRAJECTORIES FROM DISK")
        print(f"{'='*80}\n")
        
        # Use iteration-specific folder if provided, otherwise use global
        if iteration is not None:
            traj_dir = Path(self.config.checkpoint_dir) / f"iteration_{iteration}" / "trajectories"
        else:
            traj_dir = Path(self.config.output_dir).parent / "trajectories"
        
        if not traj_dir.exists():
            raise FileNotFoundError(f"Trajectory directory not found: {traj_dir}")
        
        # Find all trajectory files sorted by modification time (most recent first)
        traj_files = sorted(traj_dir.glob("traj_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        
        if len(traj_files) < num_trajectories:
            raise ValueError(f"Only found {len(traj_files)} trajectories, need {num_trajectories}")
        
        # Load the most recent N trajectories
        recent_files = traj_files[:num_trajectories]
        print(f"Found {len(traj_files)} total trajectories, loading {num_trajectories} most recent:\n")
        for i, f in enumerate(recent_files, 1):
            print(f"  {i}. {f.name}")
        print()
        
        trajectories = []
        total_input_tokens = 0
        total_output_tokens = 0
        
        for traj_file in recent_files:
            print(f"Loading {traj_file.name}...")
            
            try:
                # Load saved game using GameJSONEncoder format
                with open(traj_file, 'r') as f:
                    history_items: List[History]
                    players: List[Player]
                    game_config: GameConfig
                    history_items, players, game_config = json.load(f, object_hook=game_object_hook)
                
                # Determine outcome
                end_reason = get_end_game_reason(history_items, players)
                if end_reason is None:
                    raise ValueError(f"Game {traj_file.name} ended without a clear winner")
                
                winner_role = self.actor._determine_winner_from_end_reason(end_reason)
                player_roles = {p.name: p.role for p in players}
                
                # Reconstruct TurnData for each turn by replaying the game
                # Use GameEngine to get proper conversation context at each turn
                turns_data = []
                
                print(f"  Reconstructing {len(history_items)} turns...")
                
                # Create a fresh engine to replay the game
                replay_engine = GameEngine(game_config)
                replay_engine.players = players
                replay_engine.history = []
                
                for turn_idx, hist in enumerate(history_items):
                    if hist.action_taken.player_name == "System":
                        # Add system action to replay engine and continue
                        replay_engine.history.append(hist)
                        continue
                    
                    player_name = hist.action_taken.player_name
                    player_role = player_roles[player_name]
                    
                    # Get proper conversation context from GameEngine
                    turn_context = replay_engine.get_turn_context(player_name)
                    if turn_context is None or turn_context[0] is None:
                        print(f"    ⚠️ Warning: Could not get turn context for turn {turn_idx}, skipping")
                        replay_engine.history.append(hist)
                        continue
                    
                    _, actions_available, conversation, _ = turn_context
                    
                    # Extract reasoning from llm_cot if available
                    reasoning = hist.llm_cot if hasattr(hist, 'llm_cot') and hist.llm_cot else ""
                    
                    # Build global state
                    global_state = self.actor._build_global_state(replay_engine.history, len(replay_engine.history))
                    
                    # Find which action index was chosen (match by command_perspective)
                    chosen_action_idx = 0
                    for i, action in enumerate(actions_available):
                        if action.command_perspective == hist.action_taken.set_stories().command_perspective:
                            chosen_action_idx = i
                            break
                    
                    # Create TurnData with proper conversation context
                    turn_data = TurnData(
                        player_name=player_name,
                        player_role=player_role,
                        conversation=conversation,  # Proper conversation from GameEngine
                        actions=actions_available,  # All available actions at this turn
                        chosen_action_idx=chosen_action_idx,
                        chosen_action=hist.action_taken,
                        reasoning=reasoning,
                        generation_log_probs=[],  # Will be computed during PPO update
                        global_state_repr=global_state,
                        input_tokens=0,  # Unknown from saved data
                        output_tokens=len(reasoning.split()),  # Rough estimate
                        generation_time=0.0
                    )
                    
                    turns_data.append(turn_data)
                    total_output_tokens += turn_data.output_tokens
                    
                    # Apply the action to replay engine for next turn
                    replay_engine.history.append(hist)
                
                # Create Trajectory object
                trajectory = Trajectory(
                    turns=turns_data,
                    winner_role=winner_role,
                    end_reason=end_reason,
                    collection_time=0.0,  # Not applicable for loaded trajectories
                    total_input_tokens=0,
                    total_output_tokens=total_output_tokens
                )
                
                trajectories.append(trajectory)
                print(f"  ✅ Loaded: {len(turns_data)} turns, winner: {winner_role.name}, end_reason: {end_reason.name}\n")
                
            except Exception as e:
                print(f"  ❌ Error loading {traj_file.name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if len(trajectories) < num_trajectories:
            raise ValueError(f"Only successfully loaded {len(trajectories)}/{num_trajectories} trajectories")
        
        # Create collection stats
        collection_stats = {
            "collection/total_time": 0.0,  # Not applicable
            "collection/avg_time_per_trajectory": 0.0,
            "collection/total_input_tokens": total_input_tokens,
            "collection/total_output_tokens": total_output_tokens,
            "collection/avg_input_tokens_per_trajectory": total_input_tokens / max(len(trajectories), 1),
            "collection/avg_output_tokens_per_trajectory": total_output_tokens / max(len(trajectories), 1),
            "collection/loaded_from_disk": True,
        }
        
        print(f"{'='*80}")
        print(f"📊 LOADED {len(trajectories)} TRAJECTORIES FROM DISK")
        print(f"{'='*80}")
        print(f"Total turns: {sum(len(t.turns) for t in trajectories)}")
        print(f"Total output tokens: {total_output_tokens}")
        print(f"⚠️  WARNING: Log probs will be recomputed during PPO update")
        print(f"{'='*80}\n")
        
        return trajectories, collection_stats
    
    def _compute_collection_stats(self, trajectories: List[Trajectory], collection_time: float) -> dict:
        """Compute collection statistics from trajectories"""
        total_input_tokens = sum(t.total_input_tokens for t in trajectories)
        total_output_tokens = sum(t.total_output_tokens for t in trajectories)
        avg_collection_time = sum(t.collection_time for t in trajectories) / max(len(trajectories), 1)
        
        return {
            "collection/total_time": collection_time,
            "collection/avg_time_per_trajectory": avg_collection_time,
            "collection/total_input_tokens": total_input_tokens,
            "collection/total_output_tokens": total_output_tokens,
            "collection/avg_input_tokens_per_trajectory": total_input_tokens / max(len(trajectories), 1),
            "collection/avg_output_tokens_per_trajectory": total_output_tokens / max(len(trajectories), 1),
        }
    
    def collect_trajectories(self, num_trajectories: int) -> tuple[List[Trajectory], dict]: # MARK: .      COLLECT BATCH
        """Collect multiple trajectories using vectorized batch inference.
        
        Runs N GameEngine instances simultaneously and batches inference across active environments.
        
        Returns:
            trajectories: List of collected trajectories
            collection_stats: Dictionary of collection statistics
        """
        collection_start_time = time.time()
        
        # Check if we have saved trajectories from a previous run of this iteration
        iteration_num = self.iteration + 1
        iteration_dir = Path(self.config.checkpoint_dir) / f"iteration_{iteration_num}"
        traj_dir = iteration_dir / "trajectories"
        
        if traj_dir.exists():
            existing_files = sorted(traj_dir.glob("traj_*.json"), key=lambda p: p.stat().st_mtime)
            if existing_files:
                print(f"\n📂 Found {len(existing_files)} existing trajectories for iteration {iteration_num}")
                # Allow tolerance for missing trajectories (up to 20% missing)
                min_acceptable = int(num_trajectories * 0.8)
                if len(existing_files) >= num_trajectories:
                    print(f"✅ Loading all {num_trajectories} trajectories from disk (no collection needed)")
                    return self._load_recent_trajectories_from_disk(num_trajectories, iteration=iteration_num)
                elif len(existing_files) >= min_acceptable:
                    print(f"⚠️  Found {len(existing_files)}/{num_trajectories} trajectories (>= {min_acceptable} min acceptable)")
                    print(f"✅ Continuing with {len(existing_files)} existing trajectories")
                    return self._load_recent_trajectories_from_disk(len(existing_files), iteration=iteration_num)
        
        print(f"\n{'='*80}")
        print(f"📊 VECTORIZED BATCH COLLECTION: {num_trajectories} TRAJECTORIES")
        print(f"   Inference batch size: {self.config.inference_batch_size}")
        print(f"{'='*80}\n")
        
        # Data structures for vectorized collection
        @dataclass
        class EnvState:
            """State for a single environment"""
            env_id: int
            engine: GameEngine
            turns_data: List[TurnData]
            player_roles: Dict[str, PlayerRole]
            turn_count: int
            start_time: float
            total_input_tokens: int = 0
            total_output_tokens: int = 0
        
        # Initialize all environments
        active_envs: List[EnvState] = []
        for i in range(num_trajectories):
            engine = GameEngine(self.actor.game_config)
            player_roles = {p.name: p.role for p in engine.players}
            
            env_state = EnvState(
                env_id=i,
                engine=engine,
                turns_data=[],
                player_roles=player_roles,
                turn_count=0,
                start_time=time.time()
            )
            active_envs.append(env_state)
        
        if self.config.debug:
            print(f"🎮 Initialized {len(active_envs)} game environments\n")
        
        completed_trajectories: List[Trajectory] = []
        max_turns_per_game = 200
        
        # Main vectorized loop
        step_count = 0
        while active_envs:
            step_count += 1
            avg_turns = sum(e.turn_count for e in active_envs) / len(active_envs)
            print(f"\n📍 Step {step_count}: {len(active_envs)} envs active, {len(completed_trajectories)} done, avg turn={avg_turns:.1f}", flush=True)
            
            # Step 1: Gather observations from active environments
            batch_conversations = []
            batch_actions_list = []
            batch_env_indices = []  # Track which env each batch item belongs to
            batch_player_names = []  # Track which player is acting in each env
            
            for idx, env_state in enumerate(active_envs):
                turn_context = env_state.engine.get_turn_context()
                
                if turn_context is None or turn_context[0] is None:
                    # Game ended, mark for removal
                    continue
                
                turn_history, actions_player_can_take, conversation, _ = turn_context
                
                if not actions_player_can_take:
                    # Skip if no actions available
                    continue
                
                # Store the player name to ensure determinism when applying action later
                current_player_name = turn_history.action_taken.player_name
                
                # Add to batch
                batch_conversations.append(conversation)
                batch_actions_list.append(actions_player_can_take)
                batch_env_indices.append(idx)
                batch_player_names.append(current_player_name)
            
            if not batch_conversations:
                # All environments finished or stalled
                break
            
            # Step 2: Batch inference across all active environments
            # Process in chunks of inference_batch_size to avoid OOM
            all_batch_results = []
            
            for chunk_start in range(0, len(batch_conversations), self.config.inference_batch_size):
                chunk_end = min(chunk_start + self.config.inference_batch_size, len(batch_conversations))
                
                chunk_conversations = batch_conversations[chunk_start:chunk_end]
                chunk_actions = batch_actions_list[chunk_start:chunk_end]
                
                # Call batch generation
                chunk_results = self.actor.generate_action_batch(
                    chunk_conversations,
                    chunk_actions
                )
                
                all_batch_results.extend(chunk_results)
            
            # Step 3: Apply actions to their respective environments
            envs_to_remove = []
            
            for batch_idx, env_list_idx in enumerate(batch_env_indices):
                env_state = active_envs[env_list_idx]
                
                # Get the result for this environment
                action_idx, log_probs, reasoning, input_tokens, output_tokens, action_probs, generated_message = all_batch_results[batch_idx]
                
                # Use stored player name to ensure we get the same player as when we gathered observations
                current_player_name = batch_player_names[batch_idx]
                
                # Get turn context with explicit player name for determinism
                turn_context = env_state.engine.get_turn_context(player_name=current_player_name)
                if turn_context is None or turn_context[0] is None:
                    envs_to_remove.append(env_list_idx)
                    continue
                
                turn_history, actions_player_can_take, conversation, _ = turn_context
                chosen_action = actions_player_can_take[action_idx]
                current_player_role = env_state.player_roles[current_player_name]
                
                # For SPEAK actions, apply the generated message to the new action object
                if generated_message is not None:
                    chosen_action.target_message = generated_message
                    chosen_action.command_perspective = generated_message
                
                # Build global state for critic
                global_state = self.actor._build_global_state(
                    env_state.engine.history,
                    len(env_state.engine.history)
                )
                
                # Store turn data
                turn_data = TurnData(
                    player_name=current_player_name,
                    player_role=current_player_role,
                    conversation=conversation.copy(),
                    actions=actions_player_can_take.copy(),
                    chosen_action_idx=action_idx,
                    chosen_action=chosen_action,
                    reasoning=reasoning,
                    generation_log_probs=log_probs,
                    action_probs=action_probs,  # Store for entropy computation
                    global_state_repr=global_state,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    generation_time=0.0,  # Not tracked in batch mode
                )
                env_state.turns_data.append(turn_data)
                env_state.total_input_tokens += input_tokens
                env_state.total_output_tokens += output_tokens
                
                # Execute action
                game_over, end_reason = env_state.engine.step(
                    turn_history,
                    chosen_action,
                    llm_response=chosen_action.command_perspective,
                    llm_cot=f"<think>{reasoning}</think>",
                    token_usage={},
                    pre_discussion_votes=None
                )
                
                env_state.turn_count += 1
                
                # Check if game finished
                if game_over and end_reason:
                    winner_role = self.actor._determine_winner_from_end_reason(end_reason)
                    collection_time = time.time() - env_state.start_time
                    
                    # Save trajectory to disk
                    self.actor._save_trajectory_to_disk(
                        env_state.engine,
                        env_state.turns_data,
                        winner_role,
                        end_reason
                    )
                    
                    # Create trajectory
                    trajectory = Trajectory(
                        turns=env_state.turns_data,
                        winner_role=winner_role,
                        end_reason=end_reason,
                        collection_time=collection_time,
                        total_input_tokens=env_state.total_input_tokens,
                        total_output_tokens=env_state.total_output_tokens,
                    )
                    completed_trajectories.append(trajectory)
                    envs_to_remove.append(env_list_idx)
                    
                    print(f"\n✅ Env {env_state.env_id} completed: "
                          f"{env_state.turn_count} turns, winner: {winner_role.name}, "
                          f"end_reason: {end_reason.name}, time: {collection_time:.2f}s")
                    print(f"   Progress: {len(completed_trajectories)}/{num_trajectories}")
                
                elif env_state.turn_count >= max_turns_per_game:
                    # Game exceeded max turns
                    if self.config.debug:
                        print(f"⚠️  Env {env_state.env_id} exceeded max turns ({max_turns_per_game})")
                    envs_to_remove.append(env_list_idx)
            
            # Step 4: Remove completed/failed environments
            for env_idx in sorted(envs_to_remove, reverse=True):
                active_envs.pop(env_idx)
        
        collection_total_time = time.time() - collection_start_time
        
        # Compute statistics
        total_input_tokens = sum(t.total_input_tokens for t in completed_trajectories)
        total_output_tokens = sum(t.total_output_tokens for t in completed_trajectories)
        avg_collection_time = sum(t.collection_time for t in completed_trajectories) / max(len(completed_trajectories), 1)
        
        collection_stats = {
            "collection/total_time": collection_total_time,
            "collection/avg_time_per_trajectory": avg_collection_time,
            "collection/total_input_tokens": total_input_tokens,
            "collection/total_output_tokens": total_output_tokens,
            "collection/avg_input_tokens_per_trajectory": total_input_tokens / max(len(completed_trajectories), 1),
            "collection/avg_output_tokens_per_trajectory": total_output_tokens / max(len(completed_trajectories), 1),
        }
        
        print(f"\n{'='*80}")
        print(f"📊 VECTORIZED BATCH COLLECTION SUMMARY")
        print(f"{'='*80}")
        print(f"Collected: {len(completed_trajectories)}/{num_trajectories} trajectories")
        print(f"Total wall-clock time: {collection_total_time:.2f}s")
        print(f"Avg time per trajectory: {avg_collection_time:.2f}s")
        print(f"Total input tokens: {total_input_tokens}")
        print(f"Total output tokens: {total_output_tokens}")
        print(f"{'='*80}\n")
        
        if len(completed_trajectories) < num_trajectories:
            print(f"⚠️  Warning: Only collected {len(completed_trajectories)}/{num_trajectories} trajectories")
        
        return completed_trajectories, collection_stats
    
    def update_policy_mappo(self, trajectories: List[Trajectory]) -> Dict: # MARK: .      UPDATE POLICY
        """
        MAPPO update using PPO clipped surrogate objective with GAE.
        """
        update_start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"🎓 PPO UPDATE - TRAINING POLICY")
        print(f"{'='*80}")
        print(f"Trajectories: {len(trajectories)}")
        self._print_vram_summary("Start of PPO update")
        print(f"Computing advantages with GAE...")
        
        # Compute returns and advantages for all turns
        all_turns = []
        role_wins = defaultdict(int)
        
        # Track tokens processed during training
        total_training_tokens = 0
        
        # GAE timing
        gae_start_time = time.time()
        total_gae_turns = 0
        total_value_time = 0.0
        
        # ============================================================
        # BATCHED VALUE COMPUTATION: Compute all values at once
        # ============================================================
        print(f"  Computing values for all turns in batches...")
        value_batch_size = self.config.value_batch_size
        
        # Collect all global states and map back to turns
        all_global_states = []
        turn_to_traj_map = []  # (traj_idx, turn_idx_in_traj)
        
        for traj_idx, traj in enumerate(trajectories):
            for turn_idx, turn in enumerate(traj.turns):
                all_global_states.append(turn.global_state_repr)
                turn_to_traj_map.append((traj_idx, turn_idx))
        
        value_start = time.time()
        all_values = self._compute_value_batch(all_global_states, batch_size=value_batch_size)
        total_value_time = time.time() - value_start
        print(f"    ✅ Values computed: {total_value_time:.2f}s ({total_value_time/len(all_global_states)*1000:.1f}ms/turn)")
        
        # Assign values back to turns
        for i, (traj_idx, turn_idx) in enumerate(turn_to_traj_map):
            trajectories[traj_idx].turns[turn_idx].value_estimate = all_values[i]
        
        # ============================================================
        # VECTORIZED GAE: Compute advantages per trajectory using torch
        # ============================================================
        print(f"  Computing GAE advantages (vectorized)...")
        gae_compute_start = time.time()
        
        for traj_idx, traj in enumerate(trajectories):
            # Track wins
            for turn in traj.turns:
                if traj.get_reward_for_role(turn.player_role) > 0:
                    role_wins[turn.player_role] += 1
                    break
            
            num_turns = len(traj.turns)
            total_gae_turns += num_turns
            
            # Get final reward (sparse - only at end)
            final_reward = traj.get_reward_for_role(traj.turns[-1].player_role)
            
            # Build tensors for vectorized GAE
            values = torch.tensor([t.value_estimate for t in traj.turns], dtype=torch.float32)
            rewards = torch.zeros(num_turns, dtype=torch.float32)
            rewards[-1] = final_reward
            
            # Vectorized GAE computation (reverse cumulative sum)
            # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            # For last step: delta_T = r_T + gamma * 0 - V(s_T) = r_T - V(s_T)
            next_values = torch.cat([values[1:], torch.zeros(1)])
            deltas = rewards + self.config.gamma * next_values - values
            
            # GAE: A_t = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
            # Computed backwards: A_t = delta_t + gamma * lambda * A_{t+1}
            advantages = torch.zeros(num_turns, dtype=torch.float32)
            gae = 0.0
            for t in reversed(range(num_turns)):
                gae = deltas[t] + self.config.gamma * self.config.gae_lambda * gae
                advantages[t] = gae
            
            returns = advantages + values
            
            # Store in turns
            for turn, adv, ret in zip(traj.turns, advantages.tolist(), returns.tolist()):
                turn.advantage = adv
                turn.returns = ret
                all_turns.append(turn)
        
        gae_compute_time = time.time() - gae_compute_start
        print(f"    ✅ GAE computed: {gae_compute_time:.2f}s")
        
        # Normalize advantages and pre-convert to tensors for PPO loop
        advantages_tensor = torch.tensor([t.advantage for t in all_turns], device=self.model.device)
        returns_tensor = torch.tensor([t.returns for t in all_turns], device=self.model.device)
        adv_mean = advantages_tensor.mean().item()
        adv_std = advantages_tensor.std().item()
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Store pre-converted tensors on each turn to avoid repeated tensor creation in PPO loop
        for i, turn in enumerate(all_turns):
            turn.advantage = advantages_tensor[i].item()
            turn.advantage_tensor = advantages_tensor[i:i+1]  # Keep as 1-element tensor
            turn.returns_tensor = returns_tensor[i:i+1]  # Keep as 1-element tensor
        
        gae_total_time = time.time() - gae_start_time
        avg_gae_per_turn = gae_total_time / max(total_gae_turns, 1)
        avg_gae_per_traj = gae_total_time / max(len(trajectories), 1)
        avg_value_time = total_value_time / max(total_gae_turns, 1)
        
        print(f"Processing {len(all_turns)} turns across {len(trajectories)} trajectories")
        if self.config.debug:
            print(f"Advantage stats - Mean: {adv_mean:.4f}, Std: {adv_std:.4f}")
            print(f"Normalized advantage - Min: {advantages_tensor.min().item():.4f}, Max: {advantages_tensor.max().item():.4f}")
        print(f"\nGAE Timing:")
        print(f"  Total time: {gae_total_time:.2f}s")
        print(f"  Avg per trajectory: {avg_gae_per_traj:.2f}s")
        print(f"  Avg per turn: {avg_gae_per_turn*1000:.1f}ms")
        print(f"  Avg value computation: {avg_value_time*1000:.1f}ms/turn")
        self._print_vram_summary("After GAE/Value computation")
        
        # ====================================================================
        # START STABILITY FIX: Pre-compute old_log_probs for consistency
        # Using BATCHED computation for efficiency
        # ====================================================================
        logprob_batch_size = self.config.logprob_batch_size
        print(f"\nPre-computing log_probs for {len(all_turns)} turns (batch_size={logprob_batch_size})...")
        print("Emptying CUDA cache before pre-computation...")
        torch.cuda.empty_cache()
        self._print_vram_summary("Before LogProb Pre-computation")
        
        # Use .eval() to match the policy state during rollout
        self.model.eval()
        
        precomp_start_time = time.time()
        successful_turns = 0
        
        with torch.no_grad():
            # Compute old_log_probs (current policy) in batches
            old_start = time.time()
            print(f"  Computing old_log_probs (current policy)...")
            try:
                old_log_probs_list = self._compute_log_prob_batch(
                    all_turns,
                    use_reference_model=False,
                    batch_size=logprob_batch_size
                )
                for i, (turn, log_probs) in enumerate(zip(all_turns, old_log_probs_list)):
                    turn.precomputed_old_log_probs = log_probs.detach().cpu()
                old_time = time.time() - old_start
                print(f"    ✅ old_log_probs: {old_time:.2f}s ({old_time/len(all_turns)*1000:.1f}ms/turn)")
            except Exception as e:
                print(f"    ❌ Batched old_log_prob failed: {e}")
                print(f"    Falling back to sequential computation...")
                old_time = 0.0
                for i, turn in enumerate(all_turns):
                    try:
                        old_log_prob = self._compute_log_prob(
                            turn.conversation, turn.actions,
                            turn.chosen_action_idx, turn.reasoning,
                            use_reference_model=False
                        )
                        turn.precomputed_old_log_probs = old_log_prob.detach().cpu()
                    except Exception as e2:
                        turn.precomputed_old_log_probs = None
                        if self.config.debug:
                            print(f"      Error on turn {i}: {e2}")
                old_time = time.time() - old_start
            
            # Compute ref_log_probs (base model without LoRA) in batches
            ref_start = time.time()
            print(f"  Computing ref_log_probs (reference model)...")
            try:
                ref_log_probs_list = self._compute_log_prob_batch(
                    all_turns,
                    use_reference_model=True,
                    batch_size=logprob_batch_size
                )
                for i, (turn, log_probs) in enumerate(zip(all_turns, ref_log_probs_list)):
                    turn.precomputed_ref_log_probs = log_probs.detach().cpu()
                    if turn.precomputed_old_log_probs is not None:
                        successful_turns += 1
                ref_time = time.time() - ref_start
                print(f"    ✅ ref_log_probs: {ref_time:.2f}s ({ref_time/len(all_turns)*1000:.1f}ms/turn)")
            except Exception as e:
                print(f"    ❌ Batched ref_log_prob failed: {e}")
                print(f"    Falling back to sequential computation...")
                ref_time = 0.0
                for i, turn in enumerate(all_turns):
                    try:
                        ref_log_prob = self._compute_log_prob(
                            turn.conversation, turn.actions,
                            turn.chosen_action_idx, turn.reasoning,
                            use_reference_model=True
                        )
                        turn.precomputed_ref_log_probs = ref_log_prob.detach().cpu()
                        if turn.precomputed_old_log_probs is not None:
                            successful_turns += 1
                    except Exception as e2:
                        turn.precomputed_ref_log_probs = None
                        if self.config.debug:
                            print(f"      Error on turn {i}: {e2}")
                ref_time = time.time() - ref_start
        
        precomp_total_time = time.time() - precomp_start_time
        
        print(f"\nPre-computation complete: {successful_turns}/{len(all_turns)} successful")
        print(f"Precompute Timing:")
        print(f"  Total time: {precomp_total_time:.2f}s")
        print(f"  Avg per turn: {precomp_total_time/len(all_turns)*1000:.1f}ms")
        print(f"  old_log_prob total: {old_time:.2f}s")
        print(f"  ref_log_prob total: {ref_time:.2f}s")
        
        # Final cache clear after precomputation
        print("\n🧹 Final GPU cache clear after precomputation...")
        torch.cuda.empty_cache()
        
        self._print_vram_summary("After LogProb Pre-computation + Final Clear")
        # ====================================================================
        # END STABILITY FIX
        # ====================================================================
        
        # PPO epochs
        stats = {
            "num_trajectories": len(trajectories),
            "num_turns": len(all_turns),
            "avg_trajectory_length": len(all_turns) / max(len(trajectories), 1),
            "advantage_mean": adv_mean,
            "advantage_std": adv_std,
        }
        
        # Minibatch size for gradient accumulation (configurable)
        minibatch_size = self.config.ppo_minibatch_size
        
        for epoch in range(self.config.ppo_epochs):
            epoch_start_time = time.time()
            random.shuffle(all_turns)
            self._print_vram_summary(f"Start of Epoch {epoch + 1}")
            
            if self.config.debug:
                print(f"\n{'─'*80}")
                print(f"📈 PPO Epoch {epoch + 1}/{self.config.ppo_epochs}")
                print(f"   Minibatch size: {minibatch_size}, Total turns: {len(all_turns)}")
                print(f"{'─'*80}")
            
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_kl_loss = 0.0
            total_entropy = 0.0
            num_updates = 0
            
            # Track PPO-specific metrics
            total_ratio = 0.0
            total_clipped = 0.0
            total_actor_grad_norm = 0.0
            total_critic_grad_norm = 0.0
            
            # Track timing for this epoch
            epoch_logprob_time = 0.0
            epoch_value_time = 0.0
            epoch_backward_time = 0.0
            epoch_optimizer_time = 0.0
            epoch_turns_processed = 0
            
            # Process turns in minibatches
            for minibatch_idx in range(0, len(all_turns), minibatch_size):
                minibatch_start_time = time.time()
                minibatch = all_turns[minibatch_idx : minibatch_idx + minibatch_size]
                
                # Zero gradients for this minibatch
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                
                # Set model to train mode ONCE per minibatch (not per turn)
                # This is required for gradient checkpointing to work during backward()
                self.model.train()
                self.value_head.train()
                
                minibatch_num = minibatch_idx // minibatch_size + 1
                total_minibatches = math.ceil(len(all_turns) / minibatch_size)
                
                # Accumulate gradients within the minibatch
                minibatch_logprob_time = 0.0
                minibatch_value_time = 0.0
                minibatch_backward_time = 0.0
                
                if self.config.debug:
                    print(f"\n  🔄 Minibatch {minibatch_num}/{total_minibatches}")
                    self._print_vram_summary(f"Minibatch {minibatch_num} - START")
                
                for i, turn in enumerate(minibatch):
                    # Explicitly control model state for each computation step
                    # to ensure gradient checkpointing works correctly
                    device = self.model.device
                    
                    # Create turn identifier for debugging
                    turn_idx = minibatch_idx + i
                    turn_identifier = f"Turn {turn_idx}/{len(all_turns)}: {turn.player_name} ({turn.player_role.name})"
                    
                    # Print progress every 10 turns
                    if (turn_idx + 1) % 10 == 0:
                        print(f"  📊 Progress: {turn_idx + 1}/{len(all_turns)} turns processed", flush=True)
                    
                    try:
                        # =======================================================
                        # Step 1: Get current_log_prob (ACTOR)
                        # =======================================================
                        # Model already in .train() mode (set once per minibatch)
                        logprob_start = time.time()
                        # if self.config.debug:
                        #     self._print_vram_summary(f"{turn_identifier} - Before LogProb")
                        current_log_prob = self._compute_log_prob(
                            turn.conversation,
                            turn.actions,
                            turn.chosen_action_idx,
                            turn.reasoning,
                            use_reference_model=False
                        )
                        logprob_time = time.time() - logprob_start
                        minibatch_logprob_time += logprob_time
                        
                        # if self.config.debug:
                        #     self._print_vram_summary(f"{turn_identifier} - After LogProb")
                        
                        # Step 2: Use pre-computed ref_log_prob and old_log_prob
                        # Move from CPU back to GPU for computation
                        ref_log_prob = turn.precomputed_ref_log_probs.to(device)
                        old_log_prob_tensor = turn.precomputed_old_log_probs.to(device)
                        
                        if old_log_prob_tensor is None or ref_log_prob is None:
                            if self.config.debug:
                                print(f"      ⚠️ Skipping turn {i} (failed pre-computation)")
                            continue
                        
                        # Ensure lengths match (in case of rare truncation)
                        T = min(old_log_prob_tensor.shape[0], current_log_prob.shape[0])
                        old_log_prob_tensor = old_log_prob_tensor[:T]
                        current_log_prob = current_log_prob[:T]
                        ref_log_prob = ref_log_prob[:T]
                        
                        # Track tokens (only 1 forward pass per turn now: current)
                        total_training_tokens += turn.input_tokens + turn.output_tokens
                        
                        epoch_turns_processed += 1
                        
                        # =======================================================
                        # Step 3: Get value_pred (CRITIC)
                        # =======================================================
                        # Model and value_head already in .train() mode (set once per minibatch)
                        value_start = time.time()
                        value_pred = self._compute_value(turn.global_state_repr)
                        value_time = time.time() - value_start
                        minibatch_value_time += value_time
                        
                        # if self.config.debug:
                        #     self._print_vram_summary(f"{turn_identifier} - After Value")
                        
                        # =======================================================
                        # Step 4: Compute Losses
                        # =======================================================
                        
                        # PER-TOKEN Importance sampling ratio
                        # Clamp log-ratio BEFORE exp to prevent numerical explosion
                        # Max ratio of ~7.4 (exp(2)) is reasonable for stability
                        log_ratio = current_log_prob - old_log_prob_tensor
                        log_ratio = torch.clamp(log_ratio, min=-2.0, max=2.0)
                        ratio = torch.exp(log_ratio)
                        
                        # Track ratio and clipping (use mean for tracking)
                        total_ratio += ratio.mean().item()
                        ratio_clipped_mask = (ratio < (1 - self.config.clip_epsilon)) | (ratio > (1 + self.config.clip_epsilon))
                        total_clipped += ratio_clipped_mask.float().mean().item()
                        
                        # PER-TOKEN Clipped surrogate objective
                        # Use pre-converted tensor if available, otherwise create
                        adv = turn.advantage_tensor if turn.advantage_tensor is not None else torch.tensor([turn.advantage], device=self.model.device)
                        surr1 = ratio * adv
                        surr2 = torch.clamp(
                            ratio,
                            1 - self.config.clip_epsilon,
                            1 + self.config.clip_epsilon
                        ) * adv
                        # Average per-token losses
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        # PER-TOKEN KL divergence penalty (also clamp to prevent explosion)
                        kl_div_per_token = torch.clamp(current_log_prob - ref_log_prob, min=-10.0, max=10.0)
                        kl_loss = self.config.kl_penalty_coef * kl_div_per_token.mean()
                        
                        # Value loss
                        value_target = turn.returns_tensor if turn.returns_tensor is not None else torch.tensor([turn.returns], device=self.model.device, dtype=value_pred.dtype)
                        value_loss = F.mse_loss(value_pred, value_target)
                        
                        # Entropy bonus: compute from action distribution (discrete actions)
                        # H(π) = -Σ p(a) * log(p(a))
                        if turn.action_probs is not None and len(turn.action_probs) > 1:
                            # Proper entropy from discrete action distribution
                            action_probs_tensor = torch.tensor(turn.action_probs, device=device)
                            # Avoid log(0) by clamping
                            action_probs_tensor = torch.clamp(action_probs_tensor, min=1e-10)
                            entropy = -(action_probs_tensor * torch.log(action_probs_tensor)).sum().item()
                        else:
                            # Generative turn (single action) or missing probs: use token-level proxy
                            entropy = -current_log_prob.mean().item()
                        
                        # =======================================================
                        # Step 5: Backward Passes
                        # =======================================================
                        
                        # 1. Actor Loss (backprops only to model/LoRA)
                        actor_loss = (policy_loss + kl_loss - self.config.entropy_coef * entropy)
                        actor_loss = actor_loss / len(minibatch)  # Average over minibatch
                        
                        # 2. Critic Loss (backprops only to value_head)
                        critic_loss = self.config.value_loss_coef * value_loss
                        critic_loss = critic_loss / len(minibatch)  # Average over minibatch
                        
                        # NaN/Inf check - skip this turn if loss is invalid
                        if torch.isnan(actor_loss) or torch.isnan(critic_loss) or torch.isinf(actor_loss) or torch.isinf(critic_loss):
                            if self.config.debug:
                                print(f"      ⚠️ Skipping update for turn {i} due to NaN/Inf loss.")
                                print(f"         Policy Loss: {policy_loss.item()}, KL Loss: {kl_loss.item()}, Value Loss: {value_loss.item()}")
                            continue  # Skip this turn, do not backpropagate
                        
                        # Explicitly delete intermediate tensors to free VRAM *before* backward()
                        try:
                            del current_log_prob, value_pred, ratio, log_ratio
                            del surr1, surr2, kl_div_per_token, adv, value_target
                        except NameError:
                            pass  # In case a tensor wasn't created
                        
                        # Model already in .train() mode (set once per minibatch)
                        backward_start = time.time()
                        actor_loss.backward()
                        # if self.config.debug:
                        #     self._print_vram_summary(f"{turn_identifier} - After Actor Backward")
                        critic_loss.backward()
                        # if self.config.debug:
                        #     self._print_vram_summary(f"{turn_identifier} - After Critic Backward")
                        backward_time = time.time() - backward_start
                        minibatch_backward_time += backward_time
                        
                        
                        total_policy_loss += policy_loss.item()
                        total_value_loss += value_loss.item()
                        total_kl_loss += kl_loss.item()
                        total_entropy += entropy
                        num_updates += 1
                    
                    except torch.cuda.OutOfMemoryError as oom_error:
                        print(f"\n{'='*80}")
                        print(f"❌❌ CAUGHT OOM on: {turn_identifier} ❌❌")
                        print(f"Error: {oom_error}")
                        
                        # CRITICAL: Delete any tensors that were created before the OOM
                        # These are the massive tensors holding VRAM hostage
                        try:
                            del current_log_prob
                        except NameError:
                            pass
                        try:
                            del ref_log_prob, old_log_prob_tensor
                        except NameError:
                            pass
                        try:
                            del value_pred
                        except NameError:
                            pass
                        try:
                            del ratio, surr1, surr2, kl_div_per_token, adv, value_target
                        except NameError:
                            pass
                        try:
                            del actor_loss, critic_loss, policy_loss, value_loss, kl_loss
                        except NameError:
                            pass
                        
                        # Tokenize and get exact lengths
                        actor_seq_len = "N/A"
                        try:
                            # Re-run _compute_log_prob logic to get length
                            input_text = self.tokenizer.apply_chat_template(turn.conversation, tokenize=False, add_generation_prompt=True)
                            context_ids = self.tokenizer(input_text, return_tensors="pt").to(device).input_ids
                            reasoning_tokens = self.tokenizer.encode(turn.reasoning, add_special_tokens=False)
                            end_think_tokens = self.tokenizer.encode("\n</think>", add_special_tokens=False)
                            chosen_action = turn.actions[turn.chosen_action_idx]
                            action_prefix_tokens = self.tokenizer.encode("\n\nTell:" if chosen_action.type == ActionType.SPEAK else "\n\nAction:", add_special_tokens=False)
                            action_tokens = self.tokenizer.encode(chosen_action.command_perspective, add_special_tokens=False)
                            generated_ids_list = reasoning_tokens + end_think_tokens + action_prefix_tokens + action_tokens
                            generated_ids = torch.tensor([generated_ids_list], device=device)
                            full_input_ids = torch.cat([context_ids, generated_ids], dim=1)
                            actor_seq_len = full_input_ids.shape[1]
                            # Clean up these debug tensors too
                            del context_ids, generated_ids, full_input_ids
                        except Exception as e_len:
                            actor_seq_len = f"Error getting length: {e_len}"

                        critic_seq_len = "N/A"
                        try:
                            critic_inputs = self.tokenizer(turn.global_state_repr, return_tensors="pt")
                            critic_seq_len = critic_inputs.input_ids.shape[1]
                            del critic_inputs
                        except Exception as e_len:
                            critic_seq_len = f"Error getting length: {e_len}"

                        print(f"\n  Problematic Turn Details:")
                        print(f"  Actor Seq Len (LogProb):  {actor_seq_len}")
                        print(f"  Critic Seq Len (Value): {critic_seq_len}")
                        print(f"  (Config max_seq_length is {self.config.max_seq_length})")
                        print(f"{'='*80}\n")
                        
                        # CRITICAL: Empty cache to recover from OOM and continue loop
                        torch.cuda.empty_cache()
                        
                        if self.config.debug:
                            self._print_vram_summary(f"After OOM cleanup")
                        raise oom_error
                
                # End of minibatch: clip gradients and step optimizers
                optimizer_start = time.time()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.value_head.parameters(),
                    self.config.max_grad_norm
                )
                
                total_actor_grad_norm += actor_grad_norm.item()
                total_critic_grad_norm += critic_grad_norm.item()
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                optimizer_time = time.time() - optimizer_start
                
                # Track epoch-level timing
                epoch_logprob_time += minibatch_logprob_time
                epoch_value_time += minibatch_value_time
                epoch_backward_time += minibatch_backward_time
                epoch_optimizer_time += optimizer_time
                
                minibatch_total_time = time.time() - minibatch_start_time
                avg_turn_time = minibatch_total_time / max(len(minibatch), 1)
                
                # Print timing summary for this minibatch
                print(f"\n  ✅ Minibatch {minibatch_num}/{total_minibatches} Complete: "
                      f"{minibatch_total_time:.2f}s total, {avg_turn_time*1000:.1f}ms/turn")
                print(f"    Time breakdown: logprob={minibatch_logprob_time*1000:.0f}ms, "
                      f"value={minibatch_value_time*1000:.0f}ms, "
                      f"backward={minibatch_backward_time*1000:.0f}ms, "
                      f"opt={optimizer_time*1000:.0f}ms")
                
                # Print VRAM at end of minibatch
                self._print_vram_summary(f"Minibatch {minibatch_num} - END")
            
            epoch_total_time = time.time() - epoch_start_time
            avg_epoch_turn_time = epoch_total_time / max(epoch_turns_processed, 1)
            
            print(f"\n  PPO Epoch {epoch+1}/{self.config.ppo_epochs} Summary:")
            print(f"    Losses: Policy={total_policy_loss/max(num_updates,1):.4f}, "
                  f"Value={total_value_loss/max(num_updates,1):.4f}, "
                  f"KL={total_kl_loss/max(num_updates,1):.4f}")
            print(f"    Timing: {epoch_total_time:.2f}s total, {avg_epoch_turn_time*1000:.1f}ms/turn")
            print(f"      Breakdown: logprob={epoch_logprob_time:.2f}s, "
                  f"value={epoch_value_time:.2f}s, "
                  f"backward={epoch_backward_time:.2f}s, "
                  f"optimizer={epoch_optimizer_time:.2f}s")
            
            # Clear cache between epochs to fight fragmentation
            print(f"  Epoch {epoch+1} complete. Emptying CUDA cache...")
            torch.cuda.empty_cache()
        
        # Calculate update time
        update_time = time.time() - update_start_time
        
        # Compile stats
        for role in [PlayerRole.CREWMATE, PlayerRole.IMPOSTOR]:
            wins = role_wins.get(role, 0)
            stats[f"{role.name}_wins"] = wins
            stats[f"{role.name}_win_rate"] = wins / max(len(trajectories), 1)
        
        # Add averaged losses to stats for wandb logging
        avg_policy_loss = total_policy_loss / max(num_updates, 1)
        avg_value_loss = total_value_loss / max(num_updates, 1)
        avg_kl_loss = total_kl_loss / max(num_updates, 1)
        num_grad_steps = max(math.ceil(len(all_turns) / minibatch_size) * self.config.ppo_epochs, 1)
        
        stats["policy_loss"] = avg_policy_loss
        stats["value_loss"] = avg_value_loss
        stats["kl_loss"] = avg_kl_loss
        stats["avg_entropy"] = total_entropy / max(num_updates, 1)
        stats["avg_ratio"] = total_ratio / max(num_updates, 1)
        stats["clip_fraction"] = total_clipped / max(num_updates, 1)
        stats["actor_grad_norm"] = total_actor_grad_norm / num_grad_steps
        stats["critic_grad_norm"] = total_critic_grad_norm / num_grad_steps
        
        # Explained variance: how well value function predicts returns (critical for papers)
        # EV = 1 - Var(returns - values) / Var(returns)
        returns_arr = np.array([t.returns for t in all_turns])
        values_arr = np.array([t.value_estimate for t in all_turns])
        var_returns = np.var(returns_arr)
        explained_var = 1 - np.var(returns_arr - values_arr) / (var_returns + 1e-8) if var_returns > 0 else 0
        stats["explained_variance"] = float(np.clip(explained_var, -1, 1))
        
        # Add timing and token statistics
        stats["training/update_time"] = update_time
        stats["training/total_tokens_processed"] = total_training_tokens
        stats["training/tokens_per_second"] = total_training_tokens / update_time if update_time > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"⏱️  UPDATE TIME: {update_time:.2f}s")
        print(f"🔢 TOKENS PROCESSED: {total_training_tokens:,} ({total_training_tokens/update_time:.0f} tokens/sec)")
        print(f"{'='*80}\n")
        
        return stats
    
    def train(self): # MARK: .      TRAIN
        """Main training loop"""
        print("=" * 80)
        print("MAPPO TRAINING FOR AMONG THEM")
        print("=" * 80)
        print(f"Model: {self.config.model_name}")
        print(f"Players: {self.config.num_players}")
        print(f"Policy iterations: {self.config.num_policy_iterations}")
        print(f"Trajectories per iteration: {self.config.trajectories_per_iteration}")
        if self.start_iteration > 0:
            print(f"▶️  Resuming from iteration: {self.start_iteration + 1}")
        print("=" * 80)
        
        # Calculate step offset for wandb logging
        # Pretraining uses steps 0 to (pretrain_epochs - 1)
        # Training iterations start after pretraining
        wandb_step_offset = self.config.pretrain_epochs if self.config.pretrain_value_head else 0
        
        for iteration in range(self.start_iteration, self.config.num_policy_iterations):
            self.iteration = iteration
            iteration_start_time = time.time()
            
            print(f"\n{'='*80}")
            print(f"ITERATION {iteration + 1}/{self.config.num_policy_iterations}")
            print(f"{'='*80}")
            self._print_vram_summary("Start of iteration")
            
            # Collect trajectories via self-play (auto-loads existing if available)
            trajectories, collection_stats = self.collect_trajectories(self.config.trajectories_per_iteration)

            self._print_vram_summary("After trajectory collection")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._print_vram_summary("After collection cache empty")
            
            if not trajectories:
                print("Warning: No trajectories collected, skipping iteration")
                continue
            
            # Update policy with MAPPO
            stats = self.update_policy_mappo(trajectories)
            self._print_vram_summary("After PPO update")
            
            # Merge collection stats with training stats
            stats.update(collection_stats)
            
            # Log statistics
            iteration_time = time.time() - iteration_start_time
            stats["iteration"] = iteration + 1
            stats["iteration_time"] = iteration_time
            
            self.training_stats.append(stats)
            
            # Print summary
            print(f"\n{'='*80}")
            print(f"📊 ITERATION {iteration + 1} SUMMARY")
            print(f"{'='*80}")
            print(f"Time: {iteration_time:.2f}s")
            print(f"Trajectories: {stats['num_trajectories']}")
            print(f"Total turns: {stats['num_turns']}")
            print(f"Avg trajectory length: {stats['avg_trajectory_length']:.1f}")
            print(f"CREWMATE wins: {stats.get('CREWMATE_wins', 0)} ({stats.get('CREWMATE_win_rate', 0)*100:.1f}%)")
            print(f"IMPOSTOR wins: {stats.get('IMPOSTOR_wins', 0)} ({stats.get('IMPOSTOR_win_rate', 0)*100:.1f}%)")
            print(f"{'='*80}\n")
            
            # Log to wandb with step offset to avoid conflicts with pretraining steps
            if iteration % self.config.log_every_n_iterations == 0:
                # Compute turn-level statistics across all trajectories
                all_turn_input_tokens = [turn.input_tokens for traj in trajectories for turn in traj.turns]
                all_turn_output_tokens = [turn.output_tokens for traj in trajectories for turn in traj.turns]
                all_turn_total_tokens = [turn.input_tokens + turn.output_tokens for traj in trajectories for turn in traj.turns]
                all_turn_gen_times = [turn.generation_time for traj in trajectories for turn in traj.turns]
                all_turn_tok_per_sec = [
                    (turn.input_tokens + turn.output_tokens) / turn.generation_time 
                    if turn.generation_time > 0 else 0 
                    for traj in trajectories for turn in traj.turns
                ]
                
                # Compute trajectory-level statistics
                all_traj_collection_times = [traj.collection_time for traj in trajectories]
                all_traj_input_tokens = [traj.total_input_tokens for traj in trajectories]
                all_traj_output_tokens = [traj.total_output_tokens for traj in trajectories]
                all_traj_total_tokens = [traj.total_input_tokens + traj.total_output_tokens for traj in trajectories]
                all_traj_tok_per_sec = [
                    (traj.total_input_tokens + traj.total_output_tokens) / traj.collection_time
                    if traj.collection_time > 0 else 0
                    for traj in trajectories
                ]
                
                # Add granular statistics to stats dict
                stats.update({
                    # Turn-level stats
                    "turn/input_tokens_mean": np.mean(all_turn_input_tokens),
                    "turn/input_tokens_std": np.std(all_turn_input_tokens),
                    "turn/input_tokens_min": np.min(all_turn_input_tokens),
                    "turn/input_tokens_max": np.max(all_turn_input_tokens),
                    "turn/output_tokens_mean": np.mean(all_turn_output_tokens),
                    "turn/output_tokens_std": np.std(all_turn_output_tokens),
                    "turn/output_tokens_min": np.min(all_turn_output_tokens),
                    "turn/output_tokens_max": np.max(all_turn_output_tokens),
                    "turn/total_tokens_mean": np.mean(all_turn_total_tokens),
                    "turn/total_tokens_std": np.std(all_turn_total_tokens),
                    "turn/generation_time_mean": np.mean(all_turn_gen_times),
                    "turn/generation_time_std": np.std(all_turn_gen_times),
                    "turn/generation_time_min": np.min(all_turn_gen_times),
                    "turn/generation_time_max": np.max(all_turn_gen_times),
                    "turn/tokens_per_second_mean": np.mean(all_turn_tok_per_sec),
                    "turn/tokens_per_second_std": np.std(all_turn_tok_per_sec),
                    
                    # Trajectory-level stats
                    "trajectory/collection_time_mean": np.mean(all_traj_collection_times),
                    "trajectory/collection_time_std": np.std(all_traj_collection_times),
                    "trajectory/collection_time_min": np.min(all_traj_collection_times),
                    "trajectory/collection_time_max": np.max(all_traj_collection_times),
                    "trajectory/input_tokens_mean": np.mean(all_traj_input_tokens),
                    "trajectory/input_tokens_std": np.std(all_traj_input_tokens),
                    "trajectory/output_tokens_mean": np.mean(all_traj_output_tokens),
                    "trajectory/output_tokens_std": np.std(all_traj_output_tokens),
                    "trajectory/total_tokens_mean": np.mean(all_traj_total_tokens),
                    "trajectory/total_tokens_std": np.std(all_traj_total_tokens),
                    "trajectory/tokens_per_second_mean": np.mean(all_traj_tok_per_sec),
                    "trajectory/tokens_per_second_std": np.std(all_traj_tok_per_sec),
                })
                
                # Log all stats with explicit step
                wandb_step = iteration + wandb_step_offset
                if self.config.debug:
                    print(f"\n📊 WandB Logging:")
                    print(f"  iteration={iteration}, wandb_step_offset={wandb_step_offset}, final_step={wandb_step}")
                    print(f"  Logging {len(stats)} metrics to step {wandb_step}")
                
                wandb.log(stats, step=wandb_step)
            
            # Save checkpoint
            if (iteration + 1) % self.config.save_every_n_iterations == 0:
                self.save_checkpoint(iteration + 1)
        
        # Save final model
        self.save_final_model()
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        
        wandb.finish()
    
    def save_checkpoint(self, iteration: int): # MARK: .      LOAD SAVE
        """Save model checkpoint with full training state for resume"""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"iteration_{iteration}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving checkpoint to {checkpoint_path}...")
        self.model.save_pretrained(str(checkpoint_path))
        self.tokenizer.save_pretrained(str(checkpoint_path))
        
        # Save value head
        torch.save(self.value_head.state_dict(), checkpoint_path / "value_head.pt")
        
        # Save optimizer states and iteration number for resume
        # 'iteration' stores the checkpoint folder number (1-indexed)
        # When resuming, we'll use this to determine the next loop iteration
        training_state = {
            'iteration': iteration,  # Checkpoint folder number (e.g., 3 for iteration_3/)
            'optimizer_actor_state': self.actor_optimizer.state_dict(),
            'optimizer_critic_state': self.critic_optimizer.state_dict(),
            'training_stats': self.training_stats,
        }
        torch.save(training_state, checkpoint_path / "training_state.pt")
        
        # Save training stats as JSON for readability
        with open(checkpoint_path / "training_stats.json", 'w') as f:
            json.dump(self.training_stats, f, indent=2, default=str)
        
        print(f"✅ Checkpoint saved (iteration {iteration})")
    
    def load_checkpoint_for_resume(self, checkpoint_id: str) -> int:
        """Load checkpoint and restore full training state for resume
        
        Args:
            checkpoint_id: Iteration number (e.g., "50") or "latest"
            
        Returns:
            Starting iteration number (checkpoint_iteration + 1)
        """
        checkpoint_dir = Path(self.config.checkpoint_dir)
        
        # Find the checkpoint path
        if checkpoint_id == "latest":
            # Find the latest VALID checkpoint (one with actual model weights)
            # Iteration folders may exist from trajectory saving without checkpoint being saved
            checkpoints = sorted(checkpoint_dir.glob("iteration_*"))
            valid_checkpoints = []
            for cp in checkpoints:
                # Check if this directory has actual checkpoint files
                has_adapter = (cp / "adapter_model.safetensors").exists() or (cp / "adapter_model.bin").exists()
                has_value_head = (cp / "value_head.pt").exists()
                if has_adapter and has_value_head:
                    valid_checkpoints.append(cp)
            
            if not valid_checkpoints:
                raise FileNotFoundError(f"No valid checkpoints found in {checkpoint_dir}. "
                                       f"Found {len(checkpoints)} iteration folders but none contain saved weights.")
            
            checkpoint_path = valid_checkpoints[-1]
            print(f"📥 Found latest valid checkpoint: {checkpoint_path.name} (out of {len(checkpoints)} iteration folders)")
        else:
            checkpoint_path = checkpoint_dir / f"iteration_{checkpoint_id}"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"\n{'='*80}")
        print(f"📥 RESUMING TRAINING FROM CHECKPOINT")
        print(f"{'='*80}")
        print(f"Checkpoint: {checkpoint_path}")
        
        # Load model (LoRA adapters)
        # NOTE: self.model already has LoRA applied via get_peft_model() in __init__
        # We need to load the saved adapter weights, not apply PEFT again
        print("Loading model LoRA adapter weights...")
        from peft import set_peft_model_state_dict
        import safetensors
        
        # Load the adapter weights from the checkpoint
        adapter_path = checkpoint_path / "adapter_model.safetensors"
        if adapter_path.exists():
            # Load safetensors file
            from safetensors.torch import load_file
            adapter_weights = load_file(str(adapter_path))
            # Set the weights directly on the existing PEFT model
            set_peft_model_state_dict(self.model, adapter_weights)
            del adapter_weights  # Free memory
            print("✅ Model adapter weights loaded")
        else:
            # Fallback: try loading from adapter_model.bin
            adapter_bin_path = checkpoint_path / "adapter_model.bin"
            if adapter_bin_path.exists():
                adapter_weights = torch.load(adapter_bin_path, map_location=self.model.device, weights_only=True)
                set_peft_model_state_dict(self.model, adapter_weights)
                del adapter_weights  # Free memory
                print("✅ Model adapter weights loaded (from .bin)")
            else:
                raise FileNotFoundError(f"No adapter weights found in {checkpoint_path}")
        
        # Load value head
        print("Loading value head...")
        value_head_path = checkpoint_path / "value_head.pt"
        state_dict = torch.load(value_head_path, map_location=self.model.device, weights_only=True)
        self.value_head.load_state_dict(state_dict)
        del state_dict  # Free memory
        print("✅ Value head loaded")
        
        # Load training state (optimizers, iteration, stats)
        print("Loading training state...")
        training_state_path = checkpoint_path / "training_state.pt"
        if training_state_path.exists():
            training_state = torch.load(training_state_path, map_location=self.model.device, weights_only=False)
            
            # Restore optimizer states
            self.actor_optimizer.load_state_dict(training_state['optimizer_actor_state'])
            self.critic_optimizer.load_state_dict(training_state['optimizer_critic_state'])
            
            # Restore training stats
            self.training_stats = training_state['training_stats']
            
            # Get saved iteration (this is the checkpoint folder number, 1-indexed)
            # Example: checkpoint iteration_3/ has saved_iteration=3
            # This means: completed loop iterations 0,1,2 (displayed as ITERATION 1,2,3)
            # Next loop iteration should be: 3 (displayed as ITERATION 4)
            # Folder for next iteration: iteration_4/
            saved_iteration = training_state['iteration']
            
            del training_state  # Free memory
            
            # The checkpoint folder number equals the next loop iteration (0-indexed)
            next_loop_iteration = saved_iteration
            
            print(f"✅ Training state loaded from checkpoint: iteration_{saved_iteration}/")
            print(f"   Completed: loop iterations 0-{saved_iteration-1} (displayed as ITERATION 1-{saved_iteration})")
            print(f"   Will resume: loop iteration {next_loop_iteration} (displayed as ITERATION {next_loop_iteration + 1})")
            print(f"   Folder: iteration_{next_loop_iteration + 1}/")
            print(f"{'='*80}\n")
            
            return next_loop_iteration
        else:
            print("⚠️  Warning: training_state.pt not found, only model weights loaded")
            print(f"{'='*80}\n")
            return 0
    
    def save_final_model(self):
        """Save final trained model"""
        final_path = Path(self.config.output_dir) / "final_model"
        final_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving final model to {final_path}...")
        self.model.save_pretrained(str(final_path))
        self.tokenizer.save_pretrained(str(final_path))
        
        # Save value head
        torch.save(self.value_head.state_dict(), final_path / "value_head.pt")
        
        # Save final stats
        with open(final_path / "training_stats.json", 'w') as f:
            json.dump(self.training_stats, f, indent=2, default=str)
    
    def _load_model_checkpoint(self, checkpoint_id: str):
        """Load model from checkpoint
        
        Args:
            checkpoint_id: Either iteration number (e.g., "50"), "final", or direct file path
        """
        # Check if it's a direct path
        if "/" in checkpoint_id or "\\" in checkpoint_id:
            checkpoint_path = Path(checkpoint_id)
        elif checkpoint_id == "final":
            checkpoint_path = Path(self.config.output_dir) / "final_model"
        else:
            checkpoint_path = Path(self.config.checkpoint_dir) / f"iteration_{checkpoint_id}"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"\n{'='*60}")
        print(f"📥 Loading model from: {checkpoint_path}")
        print(f"{'='*60}")
        
        # Load LoRA adapters
        # NOTE: This is called INSTEAD of get_peft_model() when loading from checkpoint
        print("Loading LoRA adapters from checkpoint...")
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(
            self.model,
            str(checkpoint_path),
            is_trainable=True
        )
        print("✅ Model LoRA adapters loaded")
    
    def _load_value_head_checkpoint(self, checkpoint_id: str):
        """Load value head from checkpoint
        
        Args:
            checkpoint_id: Either iteration number (e.g., "50"), "final", or direct file path to .pt file
        """
        # Check if it's a direct path to a .pt file
        if checkpoint_id.endswith(".pt"):
            value_head_path = Path(checkpoint_id)
        elif "/" in checkpoint_id or "\\" in checkpoint_id:
            # It's a directory path
            checkpoint_path = Path(checkpoint_id)
            value_head_path = checkpoint_path / "value_head.pt"
        elif checkpoint_id == "final":
            checkpoint_path = Path(self.config.output_dir) / "final_model"
            value_head_path = checkpoint_path / "value_head.pt"
        else:
            checkpoint_path = Path(self.config.checkpoint_dir) / f"iteration_{checkpoint_id}"
            value_head_path = checkpoint_path / "value_head.pt"
        
        if not value_head_path.exists():
            raise FileNotFoundError(f"Value head checkpoint not found: {value_head_path}")
        
        print(f"\n{'='*60}")
        print(f"📥 Loading value head from: {value_head_path}")
        print(f"{'='*60}")
        
        # Load value head state dict
        state_dict = torch.load(value_head_path, map_location=self.model.device, weights_only=True)
        self.value_head.load_state_dict(state_dict)
        del state_dict  # Free memory
        print("✅ Value head loaded")


# ============================================================================
# MARK: MAIN
# ============================================================================

def main():
    """Main entry point"""
    config = MAPPOConfig(
        # Testing config
        # Model configuration
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        max_seq_length=60000, # 3k reasoning per turn with 11 turns avg per player. 20 turns = 60k
        max_reasoning_tokens=700, 
        load_in_4bit=True,
        lora_rank=16,
        
        # Game configuration
        num_players=5,
        num_impostors=1,
        num_tasks=2,
        map_size=0,
        num_task_phase_actions_per_player=8,
        num_discuss_phase_actions_per_player=2,
        impostor_cooldown=1,
        
        # Training configuration
        num_policy_iterations=200,
        trajectories_per_iteration=32,
        inference_batch_size=32,  # Number of environments to batch during inference
        actor_lr=1e-5,
        critic_lr=1e-4,
        gradient_accumulation_steps=1,
        
        # PPO-specific
        ppo_epochs=4,
        ppo_minibatch_size=256,
        logprob_batch_size=8,  # Small batch - full sequences are very long
        value_batch_size=8,
        clip_epsilon=0.2,
        gamma=1.0,
        gae_lambda=0.95,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=1.0,
        kl_penalty_coef=0.1,  # KL divergence penalty
        
        # Value head pretraining
        pretrain_value_head=True,
        pretrain_epochs=0,
        pretrain_examples=100,
        pretrain_lr=1e-4,
        data_dir=os.environ.get("MAPPO_DATA_DIR", "/content/among_them/data"), # TODO change them
        
        # Output paths
        output_dir=os.environ.get("MAPPO_OUTPUT_DIR", "/content/drive/MyDrive/among_them/outputs/mappo_training"),
        checkpoint_dir=os.environ.get("MAPPO_CHECKPOINT_DIR", "/content/drive/MyDrive/among_them/outputs/mappo_checkpoints"),
        
        # Checkpoint loading (for transfer learning - starts training from iteration 0)
        # load_model="50",                                                                     # Load model from iteration 50
        # load_model="final",                                                                   # Load final saved model 
        load_model=os.environ.get("MAPPO_LOAD_MODEL", os.path.expanduser("/content/drive/MyDrive/among_them/sft_unsampled")),  # Direct path to checkpoint dir
        # load_head="50",                                                                      # Load value head from iteration 50
        # load_head="final",                                                                   # Load final saved value head
        load_head=os.environ.get("MAPPO_LOAD_HEAD", os.path.expanduser("/content/drive/MyDrive/among_them/outputs/mappo_checkpoints/pretrain/value_head_epoch_10.pt")),  # Direct path to .pt file
        
        # Resume training (restores full state: weights + optimizers + iteration)
        # resume_from="50",                                                                    # Resume from iteration 50
        # resume_from="latest",                                                                # Resume from latest checkpoint
        resume_from=None,                                                                      # Resume from latest checkpoint
        
        # Logging
        wandb_project="among-them-mappo",
        wandb_run_name="mappo-5players-batch-without-think",
        log_every_n_iterations=1,
        save_every_n_iterations=1,
        save_trajectories=True,
        
        # Debug
        debug=True,
        seed=42,
        load_trajectories_from_disk=None,  # Load last 8 trajectories from disk and skip collection (debug only)
    )
    
    if os.environ.get("WANDB_API_KEY"):
        wandb.login()
    else:
        raise ValueError("WANDB_API_KEY environment variable not set")
        
    print("WANDB login successful")
    trainer = MAPPOTrainer(config)
    
    # Catch all errors and save stacktrace to file
    try:
        trainer.train()
    except Exception as e:
        error_file = "/home/ai155842/among_them/outputs/error_log.txt"
        with open(error_file, "w") as f:
            f.write(f"{'='*80}\n")
            f.write(f"TRAINING ERROR - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Error: {str(e)}\n\n")
            f.write("Full Traceback:\n")
            import traceback
            f.write(traceback.format_exc())
        
        print(f"\n{'='*80}")
        print(f"❌ TRAINING CRASHED - Stacktrace saved to {error_file}")
        print(f"{'='*80}")
        print(traceback.format_exc())
        raise  # Re-raise to ensure Colab shows the error


if __name__ == "__main__":
    main()
