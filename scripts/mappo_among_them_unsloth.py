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

# Prevent tokenizer deadlocks in multiprocessing
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from pathlib import Path

# Try to use loky for better Jupyter/Colab compatibility
# Falls back to standard concurrent.futures if loky not available
try:
    from loky import ProcessPoolExecutor, as_completed
    LOKY_AVAILABLE = True
except ImportError:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    LOKY_AVAILABLE = False

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
    max_parallel_workers: int = 1  # Parallel doesn't work on Colab (complex class serialization fails)
    actor_lr: float = 1e-6
    critic_lr: float = 3e-6
    gradient_accumulation_steps: int = 4
    
    # PPO-specific
    ppo_epochs: int = 4
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
    
    # Token statistics
    input_tokens: int = 0
    output_tokens: int = 0
    generation_time: float = 0.0  # Time to generate this turn in seconds
    
    # Value estimates (filled during training)
    value_estimate: Optional[float] = None
    advantage: Optional[float] = None
    returns: Optional[float] = None


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
        
        self.game_config = GameConfig(
            num_tasks=config.num_tasks,
            num_players=config.num_players,
            num_impostors=config.num_impostors,
            map_size=config.map_size,
            num_task_phase_actions_per_player=config.num_task_phase_actions_per_player,
            num_discuss_phase_actions_per_player=config.num_discuss_phase_actions_per_player,
            impostor_cooldown=config.impostor_cooldown
        )
    
    def _generate_action_with_policy(
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
                token_id = next_token.item()
                
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
                    token_id = next_token.item()
                    
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
                
                # Update the action object
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
                        probs = torch.softmax(current_logits_branch, dim=-1)
                        token_prob = probs[token_id].item()
                        log_prob = math.log(max(token_prob, 1e-10))
                        
                        # Stream action token if debug mode
                        # if self.config.debug:
                        #     token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                        #     print(f"{token_text}(p={token_prob:.3f}) ", end="", flush=True)
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
                action_probs = torch.softmax(torch.tensor(selection_scores, device=device), dim=-1)
                chosen_idx = torch.multinomial(action_probs, num_samples=1).item()
                chosen_per_token_log_probs = all_per_token_log_probs[chosen_idx]
            
            # Combine all log_probs for PPO update
            full_generation_log_probs = reasoning_log_probs + prefix_log_probs + chosen_per_token_log_probs
            output_token_count = len(full_generation_log_probs)
            
            # Extract reasoning text
            reasoning = self.tokenizer.decode(generated_reason_ids)
            
            if self.config.debug:
                print(f"\n{'='*60}")
                print(f"🎯 ACTION GENERATION")
                print(f"{'='*60}")
                print(f"Generated {len(generated_reason_ids)} reasoning tokens")
                print(f"Reasoning: {reasoning[:200]}..." if len(reasoning) > 200 else f"Reasoning: {reasoning}")
                if not is_generative_turn:
                    print(f"\nAction probabilities:")
                    for idx, (action, prob) in enumerate(zip(actions, action_probs)):
                        marker = "👉" if idx == chosen_idx else "  "
                        print(f"{marker} [{idx}] {action.command_perspective[:60]}: {prob.item():.4f}")
                print(f"\nTotal generation tokens: {len(full_generation_log_probs)}")
                print(f"Input tokens: {input_token_count}, Output tokens: {output_token_count}")
                
                # TIMING: Print detailed timing breakdown
                total_time = timing["prefill"] + timing["reasoning_loop"] + timing["prefix_append"] + timing["kv_clone_total"] + timing["action_generation"]
                print(f"\n{'─'*60}")
                print(f"⏱️  TIMING BREAKDOWN")
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
            
            return chosen_idx, full_generation_log_probs, reasoning, input_token_count, output_token_count
    
    def _build_global_state(self, history_items: List[History], current_turn_index: int) -> str:
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
            
            # Create trajectories directory if it doesn't exist
            traj_dir = os.path.join(os.path.dirname(self.config.output_dir), "trajectories")
            os.makedirs(traj_dir, exist_ok=True)
            
            # Generate filename with timestamp and metadata
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            iteration = getattr(self.trainer, 'iteration', 0) if self.trainer else 0
            filename = f"traj_{timestamp}_iter{iteration:03d}_{winner_role.name}_{len(turns_data)}turns.json"
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
    
    def collect_trajectory(self) -> Optional[Trajectory]:
        """
        Run one complete game using self-play and collect trajectory.
        Returns None if game fails to complete properly.
        """
        start_time = time.time()
        
        if self.config.debug:
            print(f"\n{'#'*80}")
            print(f"🎮 STARTING NEW GAME")
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
                action_idx, log_prob, reasoning, input_tokens, output_tokens = self._generate_action_with_policy(
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
                generation_log_probs=log_prob,  # Now a List[float]
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
                    print(f"🏁 GAME OVER")
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
# MARK: PARALLEL WORKER
# ============================================================================

def run_game_worker(
    config_dict: dict,
    model_name: str,
    lora_adapter_path: str,
    value_head_path: str,
    iteration: int,
    worker_id: int,
    progress_file: str
) -> Optional[Trajectory]:
    """
    Worker function that runs ONE game in a separate process.
    Writes progress to progress_file for real-time monitoring.
    """
    import sys
    import traceback as tb
    from pathlib import Path
    
    def log_progress(msg: str):
        """Write progress to file for parent to monitor"""
        with open(progress_file, 'a') as f:
            timestamp = time.strftime("%H:%M:%S")
            f.write(f"[{timestamp}] {msg}\n")
            f.flush()
    
    try:
        log_progress(f"Worker {worker_id} started (PID: {os.getpid()})")
        print(f"[Worker {worker_id}] Process started (PID: {os.getpid()})")
        sys.stdout.flush()
        
        # 1. Reconstruct config from dict
        print(f"[Worker {worker_id}] Reconstructing config...")
        sys.stdout.flush()
        config = MAPPOConfig(**config_dict)

        # Stagger worker startup to avoid simultaneous GPU memory allocation
        print(f"[Worker {worker_id}] Staggering startup by {(worker_id % config.max_parallel_workers) * 3}s...")
        sys.stdout.flush()
        time.sleep((worker_id % config.max_parallel_workers) * 3)
        
        print(f"[Worker {worker_id}] Loading model from {model_name}...")
        sys.stdout.flush()
        
        # Disable debug in workers to avoid log clutter
        config.debug = False
        
        # 2. Load base model and tokenizer
        log_progress("Loading model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name,
            load_in_4bit=config.load_in_4bit,
            max_seq_length=config.max_seq_length,
            attn_implementation="flash_attention_2",
        )
        log_progress("Model loaded")
        
        # 3. Load LoRA adapters (policy)
        log_progress("Loading LoRA adapters...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model,
            str(lora_adapter_path),
            is_trainable=False  # Not training in workers, just collecting
        )
        log_progress("LoRA adapters loaded")
        
        # 4. Compile model for optimized inference
        # log_progress("Compiling model...")
        # try:
        #     model = torch.compile(model, mode="reduce-overhead")
        #     log_progress("Model compiled")
        # except Exception as e:
        #     log_progress(f"torch.compile failed: {e}")
        
        # 5. Load value head (critic)
        log_progress("Loading value head...")
        hidden_size = model.config.hidden_size
        value_head = ValueHead(hidden_size).to(device=model.device, dtype=torch.float32)
        state_dict = torch.load(value_head_path, map_location=model.device, weights_only=True)
        value_head.load_state_dict(state_dict)
        log_progress("Value head loaded")
        
        # 6. Create mock trainer for iteration tracking (needed for trajectory saving)
        print(f"[Worker {worker_id}] Creating actor...")
        sys.stdout.flush()
        class MockTrainer:
            pass
        mock_trainer = MockTrainer()
        mock_trainer.iteration = iteration
        
        # 7. Create actor
        actor = MAPPOActor(
            model=model,
            tokenizer=tokenizer,
            value_head=value_head,
            config=config,
            trainer=mock_trainer
        )
        print(f"[Worker {worker_id}] Actor created")
        sys.stdout.flush()
        
        # 8. Collect ONE trajectory with turn-by-turn progress
        log_progress("Starting game...")
        
        # Pass progress callback to actor for turn-by-turn updates
        actor.progress_callback = lambda turn, total: log_progress(f"Turn {turn}/{total} complete")
        
        trajectory = actor.collect_trajectory()
        log_progress(f"Game complete: {len(trajectory.turns) if trajectory else 0} turns")
        
        # 9. Clean up GPU memory before exit
        log_progress("Cleaning up...")
        del model, tokenizer, value_head, actor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_progress("Complete!")
        
        return trajectory
        
    except Exception as e:
        print(f"\n{'='*80}", file=sys.stderr)
        print(f"FATAL ERROR in worker {worker_id} (PID: {os.getpid()})", file=sys.stderr)
        print(f"{'='*80}", file=sys.stderr)
        print(f"Exception type: {type(e).__name__}", file=sys.stderr)
        print(f"Exception message: {e}", file=sys.stderr)
        print(f"\nFull traceback:", file=sys.stderr)
        tb.print_exc(file=sys.stderr)
        print(f"{'='*80}\n", file=sys.stderr)
        sys.stderr.flush()
        return None


# ============================================================================
# MARK: MAPPO TRAINER
# ============================================================================

class MAPPOTrainer:
    """Main trainer implementing MAPPO algorithm"""
    
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
        print(f"{'='*80}\n")
        
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
        """Set random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def pretrain_value_head(self):
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
            
            # Log to wandb
            wandb.log({
                "pretrain/epoch": epoch + 1,
                "pretrain/loss": avg_loss,
            })
            
            # Save value head after each epoch
            pretrain_checkpoint_dir = Path(self.config.checkpoint_dir) / "pretrain"
            pretrain_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = pretrain_checkpoint_dir / f"value_head_epoch_{epoch+1}.pt"
            torch.save(self.value_head.state_dict(), checkpoint_path)
            print(f"  💾 Saved value head to: {checkpoint_path}")
        
        print("Value head pretraining complete!\n")
    
    def _compute_value(self, global_state_repr: str) -> torch.Tensor:
        """Compute value estimate for a global state"""
        # Note: Caller controls value_head.train() vs .eval() mode
        
        inputs = self.tokenizer(
            global_state_repr,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length
        ).to(self.model.device)
        
        # Base model always in eval for value computation (no gradients needed)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
        
        # Detach hidden_states from the model's computation graph
        hidden_states = hidden_states.detach()
        
        # Explicitly enable gradients for the value_head pass
        with torch.enable_grad():
            value = self.value_head(hidden_states).squeeze(-1)
        
        return value
    
    def _compute_log_prob(
        self,
        conversation: List[Dict[str, str]],
        actions: List[Action],
        chosen_action_idx: int,
        reasoning: str,
        use_reference_model: bool = False
    ) -> torch.Tensor:
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
        
        # 4. Forward pass
        if use_reference_model:
            with torch.no_grad(), self.model.disable_adapter():
                 outputs = self.model(input_ids=full_input_ids)
        else:
            with torch.enable_grad():
                 outputs = self.model(input_ids=full_input_ids)
            
        # [1, C+G, V]
        logits = outputs.logits 

        # 5. Get log_probs for the GENERATED tokens
        
        # We need the logits that *predicted* the generated tokens.
        # These start from `context_len - 1` and go to the end.
        # [1, G, V] (G = number of generated tokens)
        generated_logits = logits[:, context_len-1:-1, :]
        
        # [1, G, V]
        generated_log_probs = F.log_softmax(generated_logits, dim=-1)

        # [1, G, 1]
        generated_ids_expanded = generated_ids.unsqueeze(-1)
        
        # Gather the log_probs for the tokens that were actually chosen
        # [1, G, 1]
        chosen_token_log_probs = torch.gather(
            generated_log_probs, 
            dim=-1, 
            index=generated_ids_expanded
        )
        
        # Return the TENSOR of all per-token log-probs
        # Squeeze to shape [G]
        return chosen_token_log_probs.squeeze()
    
    def _collect_trajectories_sequential(self, num_trajectories: int) -> tuple[List[Trajectory], dict]:
        """Sequential trajectory collection (fallback when parallel doesn't work)"""
        collection_start_time = time.time()
        trajectories = []
        attempts = 0
        max_attempts = num_trajectories * 3
        
        print(f"\n{'='*80}")
        print(f"📊 SEQUENTIAL COLLECTION: {num_trajectories} TRAJECTORIES")
        print(f"{'='*80}\n")
        
        while len(trajectories) < num_trajectories and attempts < max_attempts:
            if self.config.debug:
                print(f"\n🔄 Attempt {attempts + 1}/{max_attempts}")
            
            trajectory = self.actor.collect_trajectory()
            
            if trajectory is not None:
                trajectories.append(trajectory)
                print(f"\n✅ Collected {len(trajectories)}/{num_trajectories}: "
                      f"{len(trajectory.turns)} turns, winner: {trajectory.winner_role.name}, "
                      f"end_reason: {trajectory.end_reason.name}, "
                      f"time: {trajectory.collection_time:.2f}s, "
                      f"tokens: {trajectory.total_input_tokens}in/{trajectory.total_output_tokens}out")
            else:
                print(f"  ❌ Failed trajectory (attempt {attempts + 1})")
            
            attempts += 1
        
        collection_total_time = time.time() - collection_start_time
        
        if len(trajectories) < num_trajectories:
            print(f"\n⚠️  Warning: Only collected {len(trajectories)}/{num_trajectories} trajectories")
        
        # Aggregate statistics
        total_input_tokens = sum(t.total_input_tokens for t in trajectories)
        total_output_tokens = sum(t.total_output_tokens for t in trajectories)
        avg_collection_time = sum(t.collection_time for t in trajectories) / max(len(trajectories), 1)
        
        collection_stats = {
            "collection/total_time": collection_total_time,
            "collection/avg_time_per_trajectory": avg_collection_time,
            "collection/total_input_tokens": total_input_tokens,
            "collection/total_output_tokens": total_output_tokens,
            "collection/avg_input_tokens_per_trajectory": total_input_tokens / max(len(trajectories), 1),
            "collection/avg_output_tokens_per_trajectory": total_output_tokens / max(len(trajectories), 1),
        }
        
        print(f"\n{'='*80}")
        print(f"📊 SEQUENTIAL COLLECTION SUMMARY")
        print(f"{'='*80}")
        print(f"Total time: {collection_total_time:.2f}s")
        print(f"Avg time per trajectory: {avg_collection_time:.2f}s")
        print(f"Total input tokens: {total_input_tokens}")
        print(f"Total output tokens: {total_output_tokens}")
        print(f"{'='*80}\n")
        
        return trajectories, collection_stats
    
    def collect_trajectories(self, num_trajectories: int) -> tuple[List[Trajectory], dict]:
        """Collect multiple trajectories (parallel if max_parallel_workers > 1, else sequential)
        
        Returns:
            trajectories: List of collected trajectories
            collection_stats: Dictionary of collection statistics
        """
        collection_start_time = time.time()
        trajectories = []
        
        # Fall back to sequential collection if parallel disabled or loky not available
        if self.config.max_parallel_workers <= 1:
            return self._collect_trajectories_sequential(num_trajectories)
        
        if not LOKY_AVAILABLE:
            print(f"\n⚠️  WARNING: loky not installed. Install with: pip install loky")
            print(f"   Falling back to sequential collection for Colab compatibility.")
            return self._collect_trajectories_sequential(num_trajectories)
        
        print(f"\n{'='*80}")
        print(f"📊 PARALLEL COLLECTION: {num_trajectories} TRAJECTORIES")
        print(f"   Using loky ProcessPoolExecutor (Jupyter/Colab compatible)")
        print(f"   Max parallel workers: {self.config.max_parallel_workers}")
        print(f"{'='*80}\n")
        
        # 1. Save current policy and value head for workers to load
        temp_policy_dir = Path(self.config.checkpoint_dir) / "temp_policy_for_workers"
        temp_policy_dir.mkdir(parents=True, exist_ok=True)
        
        # Create progress directory for real-time monitoring
        progress_dir = Path(self.config.checkpoint_dir) / "worker_progress"
        progress_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving current policy to {temp_policy_dir}...")
        self.model.save_pretrained(str(temp_policy_dir))
        temp_value_head_path = temp_policy_dir / "value_head.pt"
        torch.save(self.value_head.state_dict(), temp_value_head_path)
        print("Policy saved.\n")
        
        # 2. Convert config to dict for pickling
        config_dict = asdict(self.config)
        
        # 3. loky handles spawn context automatically with cloudpickle
        # No need to explicitly set mp_context with loky
        
        # 4. Collect trajectories in batches to avoid OOM
        # Process in batches of max_parallel_workers
        num_workers = min(self.config.max_parallel_workers, num_trajectories)
        total_collected = 0
        worker_id_counter = 0
        
        while total_collected < num_trajectories:
            batch_size = min(num_workers, num_trajectories - total_collected)
            print(f"\n🔄 Launching batch of {batch_size} workers (collected {total_collected}/{num_trajectories})...")
            
            # 5. Launch worker pool for this batch (loky auto-handles spawn context)
            with ProcessPoolExecutor(max_workers=batch_size) as executor:
                # Submit jobs for this batch with progress files
                futures = {}
                for i in range(batch_size):
                    wid = worker_id_counter + i
                    progress_file = str(progress_dir / f"worker_{wid}.log")
                    # Clear previous progress file
                    Path(progress_file).write_text(f"=== Worker {wid} Log ===\n")
                    
                    future = executor.submit(
                        run_game_worker,
                        config_dict,
                        self.config.model_name,
                        str(temp_policy_dir),
                        str(temp_value_head_path),
                        self.iteration,
                        wid,
                        progress_file
                    )
                    futures[future] = wid
                
                # Collect results as they complete
                for future in as_completed(futures):
                    worker_id = futures[future]
                    try:
                        trajectory = future.result()
                        if trajectory is not None:
                            trajectories.append(trajectory)
                            print(f"\n✅ Worker {worker_id} completed: "
                                  f"{len(trajectory.turns)} turns, winner: {trajectory.winner_role.name}, "
                                  f"end_reason: {trajectory.end_reason.name}, "
                                  f"time: {trajectory.collection_time:.2f}s, "
                                  f"tokens: {trajectory.total_input_tokens}in/{trajectory.total_output_tokens}out")
                            print(f"   Progress: {len(trajectories)}/{num_trajectories}")
                        else:
                            print(f"❌ Worker {worker_id} failed to collect trajectory")
                    except Exception as e:
                        print(f"❌ Worker {worker_id} raised exception: {e}")
            
            # Update counters for next batch
            total_collected = len(trajectories)
            worker_id_counter += batch_size
            
            # Add delay between batches to ensure cleanup
            if total_collected < num_trajectories:
                print(f"\n⏳ Waiting 10s before next batch...")
                time.sleep(10)
        
        collection_total_time = time.time() - collection_start_time
        
        if len(trajectories) < num_trajectories:
            print(f"\n⚠️  Warning: Only collected {len(trajectories)}/{num_trajectories} trajectories")
        
        # Aggregate statistics
        total_input_tokens = sum(t.total_input_tokens for t in trajectories)
        total_output_tokens = sum(t.total_output_tokens for t in trajectories)
        avg_collection_time = sum(t.collection_time for t in trajectories) / max(len(trajectories), 1)
        
        collection_stats = {
            "collection/total_time": collection_total_time,
            "collection/avg_time_per_trajectory": avg_collection_time,
            "collection/total_input_tokens": total_input_tokens,
            "collection/total_output_tokens": total_output_tokens,
            "collection/avg_input_tokens_per_trajectory": total_input_tokens / max(len(trajectories), 1),
            "collection/avg_output_tokens_per_trajectory": total_output_tokens / max(len(trajectories), 1),
        }
        
        print(f"\n{'='*80}")
        print(f"📊 PARALLEL COLLECTION SUMMARY")
        print(f"{'='*80}")
        print(f"Total wall-clock time: {collection_total_time:.2f}s")
        print(f"Avg time per trajectory: {avg_collection_time:.2f}s")
        print(f"Speedup: {avg_collection_time * num_trajectories / collection_total_time:.1f}x")
        print(f"Total input tokens: {total_input_tokens}")
        print(f"Total output tokens: {total_output_tokens}")
        print(f"{'='*80}\n")
        
        return trajectories, collection_stats
    
    def update_policy_mappo(self, trajectories: List[Trajectory]) -> Dict:
        """
        MAPPO update using PPO clipped surrogate objective with GAE.
        """
        update_start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"🎓 PPO UPDATE - TRAINING POLICY")
        print(f"{'='*80}")
        print(f"Trajectories: {len(trajectories)}")
        print(f"Computing advantages with GAE...")
        
        # Compute returns and advantages for all turns
        all_turns = []
        role_wins = defaultdict(int)
        
        # Track tokens processed during training
        total_training_tokens = 0
        
        for traj in trajectories:
            # Track wins
            for turn in traj.turns:
                if traj.get_reward_for_role(turn.player_role) > 0:
                    role_wins[turn.player_role] += 1
                    break
            
            # Compute GAE advantages
            returns = []
            advantages = []
            
            # Get rewards - sparse reward only at end
            final_reward = traj.get_reward_for_role(traj.turns[-1].player_role)
            
            # Bootstrap from final state
            next_value = 0.0
            gae = 0.0
            
            # Backward pass
            for turn in reversed(traj.turns):
                # Compute value estimate
                turn.value_estimate = self._compute_value(turn.global_state_repr).item()
                
                # TD error
                reward = final_reward if turn == traj.turns[-1] else 0.0
                delta = reward + self.config.gamma * next_value - turn.value_estimate
                
                # GAE
                gae = delta + self.config.gamma * self.config.gae_lambda * gae
                advantages.insert(0, gae)
                returns.insert(0, gae + turn.value_estimate)
                
                next_value = turn.value_estimate
            
            # Store in turns
            for turn, adv, ret in zip(traj.turns, advantages, returns):
                turn.advantage = adv
                turn.returns = ret
                all_turns.append(turn)
        
        # Normalize advantages
        advantages_tensor = torch.tensor([t.advantage for t in all_turns], device=self.model.device)
        adv_mean = advantages_tensor.mean().item()
        adv_std = advantages_tensor.std().item()
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        for i, turn in enumerate(all_turns):
            turn.advantage = advantages_tensor[i].item()
        
        print(f"Processing {len(all_turns)} turns across {len(trajectories)} trajectories")
        if self.config.debug:
            print(f"Advantage stats - Mean: {adv_mean:.4f}, Std: {adv_std:.4f}")
            print(f"Normalized advantage - Min: {advantages_tensor.min().item():.4f}, Max: {advantages_tensor.max().item():.4f}")
        
        # PPO epochs
        stats = {
            "num_trajectories": len(trajectories),
            "num_turns": len(all_turns),
            "avg_trajectory_length": len(all_turns) / max(len(trajectories), 1),
            "advantage_mean": adv_mean,
            "advantage_std": adv_std,
        }
        
        for epoch in range(self.config.ppo_epochs):
            random.shuffle(all_turns)
            
            if self.config.debug:
                print(f"\n{'─'*80}")
                print(f"📈 PPO Epoch {epoch + 1}/{self.config.ppo_epochs}")
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
            
            # Zero gradients at start of epoch
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            
            for i, turn in enumerate(all_turns):
                # Explicitly control model state for each computation step
                # to ensure gradient checkpointing works correctly
                device = self.model.device
                
                try:
                    if self.config.debug and i % 10 == 0:
                        print(f"  Turn {i}/{len(all_turns)}: {turn.player_name} ({turn.player_role.name})")
                    
                    # Step 1: Get current_log_prob
                    # Model MUST be in .train() mode for checkpointing to work with .backward()
                    self.model.train()
                    current_log_prob = self._compute_log_prob(
                        turn.conversation,
                        turn.actions,
                        turn.chosen_action_idx,
                        turn.reasoning,
                        use_reference_model=False
                    )
                    
                    # Step 2: Get ref_log_prob
                    # Model MUST be in .eval() mode for .disable_adapter() to work
                    self.model.eval()
                    ref_log_prob = self._compute_log_prob(
                        turn.conversation,
                        turn.actions,
                        turn.chosen_action_idx,
                        turn.reasoning,
                        use_reference_model=True  # Disables LoRA adapters
                    )
                    
                    # Convert old log probs from list to tensor
                    old_log_prob_tensor = torch.tensor(
                        turn.generation_log_probs, device=device, dtype=current_log_prob.dtype
                    )
                    
                    # Ensure lengths match (in case of rare truncation)
                    T = min(old_log_prob_tensor.shape[0], current_log_prob.shape[0])
                    old_log_prob_tensor = old_log_prob_tensor[:T]
                    current_log_prob = current_log_prob[:T]
                    ref_log_prob = ref_log_prob[:T]
                    
                    # Track tokens (2 forward passes per turn: current + reference)
                    total_training_tokens += 2 * (turn.input_tokens + turn.output_tokens)
                    
                    if self.config.debug and i % 10 == 0:
                        print(f"    Current log prob: {current_log_prob.sum().item():.4f}, "
                              f"Ref log prob: {ref_log_prob.sum().item():.4f}, "
                              f"Old log prob: {old_log_prob_tensor.sum().item():.4f}, "
                              f"Tokens: {T}")
                
                except Exception as e:
                    if self.config.debug:
                        print(f"❌ Error computing log prob for turn {i}: {e}")
                        import traceback
                        traceback.print_exc()
                    continue
                
                # Step 3: Get value_pred
                # Base model set to .eval() inside _compute_value
                # Value_head must be in .train() mode to learn
                self.value_head.train()
                value_pred = self._compute_value(turn.global_state_repr)
                
                # Now compute losses
                
                # PER-TOKEN Importance sampling ratio
                # ratio is shape [T]
                ratio = torch.exp(current_log_prob - old_log_prob_tensor)
                
                # Track ratio and clipping (use mean for tracking)
                total_ratio += ratio.mean().item()
                ratio_clipped_mask = (ratio < (1 - self.config.clip_epsilon)) | (ratio > (1 + self.config.clip_epsilon))
                total_clipped += ratio_clipped_mask.float().mean().item()
                
                # PER-TOKEN Clipped surrogate objective
                # adv is a scalar, but broadcast to shape [T]
                adv = torch.tensor([turn.advantage], device=self.model.device)
                surr1 = ratio * adv
                surr2 = torch.clamp(
                    ratio,
                    1 - self.config.clip_epsilon,
                    1 + self.config.clip_epsilon
                ) * adv
                # Average per-token losses
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # PER-TOKEN KL divergence penalty
                kl_div_per_token = (current_log_prob - ref_log_prob)
                kl_loss = self.config.kl_penalty_coef * kl_div_per_token.mean()
                
                # Value loss
                value_target = torch.tensor([turn.returns], device=self.model.device, dtype=value_pred.dtype)
                value_loss = F.mse_loss(value_pred, value_target)
                
                # Entropy (approximate - would need full action distribution)
                entropy = 0.01
                
                # SEPARATE BACKWARD PASSES to avoid gradient checkpointing conflict
                
                # 1. Actor Loss (backprops only to model/LoRA)
                actor_loss = (policy_loss + kl_loss - self.config.entropy_coef * entropy)
                actor_loss = actor_loss / self.config.gradient_accumulation_steps
                
                # CRITICAL: Set model to train mode before backward() to ensure
                # gradient checkpoint re-runs in train mode (not eval mode from _compute_value)
                self.model.train()
                actor_loss.backward()
                
                # 2. Critic Loss (backprops only to value_head)
                critic_loss = self.config.value_loss_coef * value_loss
                critic_loss = critic_loss / self.config.gradient_accumulation_steps
                critic_loss.backward()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_kl_loss += kl_loss.item()
                total_entropy += entropy
                num_updates += 1
                
                # Optimizer step with gradient accumulation
                if (i + 1) % self.config.gradient_accumulation_steps == 0 or (i + 1) == len(all_turns):
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
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
            
            print(f"  PPO Epoch {epoch+1}/{self.config.ppo_epochs}: "
                  f"Policy Loss={total_policy_loss/max(num_updates,1):.4f}, "
                  f"Value Loss={total_value_loss/max(num_updates,1):.4f}, "
                  f"KL Loss={total_kl_loss/max(num_updates,1):.4f}")
        
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
        num_grad_steps = max(num_updates // self.config.gradient_accumulation_steps, 1)
        
        stats["policy_loss"] = avg_policy_loss
        stats["value_loss"] = avg_value_loss
        stats["kl_loss"] = avg_kl_loss
        stats["avg_entropy"] = total_entropy / max(num_updates, 1)
        stats["avg_ratio"] = total_ratio / max(num_updates, 1)
        stats["clip_fraction"] = total_clipped / max(num_updates, 1)
        stats["actor_grad_norm"] = total_actor_grad_norm / num_grad_steps
        stats["critic_grad_norm"] = total_critic_grad_norm / num_grad_steps
        
        # Add timing and token statistics
        stats["training/update_time"] = update_time
        stats["training/total_tokens_processed"] = total_training_tokens
        stats["training/tokens_per_second"] = total_training_tokens / update_time if update_time > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"⏱️  UPDATE TIME: {update_time:.2f}s")
        print(f"🔢 TOKENS PROCESSED: {total_training_tokens:,} ({total_training_tokens/update_time:.0f} tokens/sec)")
        print(f"{'='*80}\n")
        
        return stats
    
    def train(self):
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
            
            # Collect trajectories via self-play
            trajectories, collection_stats = self.collect_trajectories(self.config.trajectories_per_iteration)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if not trajectories:
                print("Warning: No trajectories collected, skipping iteration")
                continue
            
            # Update policy with MAPPO
            stats = self.update_policy_mappo(trajectories)
            
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
                
                # Log all stats
                wandb.log(stats, step=iteration + wandb_step_offset)
            
            # Save checkpoint
            if (iteration + 1) % self.config.save_every_n_iterations == 0:
                self.save_checkpoint(iteration + 1)
        
        # Save final model
        self.save_final_model()
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        
        wandb.finish()
    
    def save_checkpoint(self, iteration: int):
        """Save model checkpoint with full training state for resume"""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"iteration_{iteration}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving checkpoint to {checkpoint_path}...")
        self.model.save_pretrained(str(checkpoint_path))
        self.tokenizer.save_pretrained(str(checkpoint_path))
        
        # Save value head
        torch.save(self.value_head.state_dict(), checkpoint_path / "value_head.pt")
        
        # Save optimizer states and iteration number for resume
        training_state = {
            'iteration': iteration,
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
            # Find the latest checkpoint
            checkpoints = sorted(checkpoint_dir.glob("iteration_*"))
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
            checkpoint_path = checkpoints[-1]
            print(f"📥 Found latest checkpoint: {checkpoint_path.name}")
        else:
            checkpoint_path = checkpoint_dir / f"iteration_{checkpoint_id}"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"\n{'='*80}")
        print(f"📥 RESUMING TRAINING FROM CHECKPOINT")
        print(f"{'='*80}")
        print(f"Checkpoint: {checkpoint_path}")
        
        # Load model (LoRA adapters)
        print("Loading model LoRA adapters...")
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(
            self.model,
            str(checkpoint_path),
            is_trainable=True
        )
        print("✅ Model loaded")
        
        # Load value head
        print("Loading value head...")
        value_head_path = checkpoint_path / "value_head.pt"
        state_dict = torch.load(value_head_path, map_location=self.model.device, weights_only=True)
        self.value_head.load_state_dict(state_dict)
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
            
            # Get saved iteration
            saved_iteration = training_state['iteration']
            
            print(f"✅ Training state loaded (was at iteration {saved_iteration})")
            print(f"{'='*80}\n")
            
            # Return next iteration to start from
            return saved_iteration + 1
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
        
        # Load LoRA adapters onto base model
        # This should be called BEFORE get_peft_model() is applied
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
        state_dict = torch.load(value_head_path, map_location=self.model.device)
        self.value_head.load_state_dict(state_dict)
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
        trajectories_per_iteration=4,
        actor_lr=1e-6,
        critic_lr=3e-6,
        gradient_accumulation_steps=2,
        max_parallel_workers=1,  # Requires loky: pip install loky
        
        # PPO-specific
        ppo_epochs=2,
        clip_epsilon=0.2,
        gamma=1.0,
        gae_lambda=0.95,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=1.0,
        kl_penalty_coef=0.1,  # KL divergence penalty
        
        # Value head pretraining
        pretrain_value_head=True,
        pretrain_epochs=10,
        pretrain_examples=100,
        pretrain_lr=1e-4,
        data_dir="/content/among_them/data", # TODO change them
        
        # Output paths
        output_dir="/content/drive/MyDrive/among_them/outputs/mappo_training",
        checkpoint_dir="/content/drive/MyDrive/among_them/outputs/mappo_checkpoints",
        
        # Checkpoint loading (for transfer learning - starts training from iteration 0)
        # load_model="50",                                                                     # Load model from iteration 50
        # load_model="final",                                                                   # Load final saved model 
        load_model="/content/drive/MyDrive/among_them/sft_sampled",  # Direct path to checkpoint dir
        # load_head="50",                                                                      # Load value head from iteration 50
        # load_head="final",                                                                   # Load final saved value head
        load_head="/content/drive/MyDrive/among_them/outputs/mappo_checkpoints/pretrain/value_head_epoch_10.pt",  # Direct path to .pt file
        
        # Resume training (restores full state: weights + optimizers + iteration)
        # resume_from="50",                                                                    # Resume from iteration 50
        # resume_from="latest",                                                                # Resume from latest checkpoint
        resume_from=None,                                                                      # No resume (fresh training)
        
        # Logging
        wandb_project="among-them-mappo",
        wandb_run_name="mappo-5players",
        log_every_n_iterations=1,
        save_every_n_iterations=1,
        save_trajectories=True,
        
        # Debug
        debug=True,
        seed=42
    )
    
    trainer = MAPPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
