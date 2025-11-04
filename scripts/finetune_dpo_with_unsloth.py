#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DPO (Direct Preference Optimization) training script for Among Them game AI
Based on Unsloth framework for efficient training

DPO trains the model to prefer "chosen" responses over "rejected" ones,
which is ideal for teaching the model better reasoning patterns.
"""

import os
import pathlib
import shutil
import torch
import numpy as np
import random
import wandb
from datasets import load_dataset, Features, Value
from transformers import EarlyStoppingCallback
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from trl import DPOTrainer, DPOConfig

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model configuration
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# Alternative: "unsloth/Meta-Llama-3.1-8B-Instruct"

# Dataset paths (adjust for your environment)
drive_dataset_dir = "/content/drive/MyDrive/among_them/dpo/among_them_dpo_dataset"
local_dataset_dir = "/content/dpo/among_them_dpo_dataset"
lora_weights_output_dir = "/content/drive/MyDrive/among_them/dpo/among_them_dpo_lora_checkpoints"
merged_model_dir = "/content/drive/MyDrive/among_them/dpo/deepseek_r1_dpo_merged"

# Model hyperparameters
max_seq_length = 12500
lora_rank = 16
dtype = None  # None for auto-detection (fp16/bf16)
load_in_4bit = True

# DPO-specific hyperparameters
dpo_beta = 0.1  # Controls how much to penalize rejected responses (typical: 0.1-0.5)
# Lower beta = more conservative, higher beta = stronger preference learning

# Training hyperparameters
per_device_train_batch_size = 1
per_device_eval_batch_size = 1
gradient_accumulation_steps = 8
max_steps = 1000
learning_rate = 5e-7  # DPO typically uses lower LR than SFT
warmup_steps = 50

# Reference model option
# If None, uses the same model as the training model (implicit reference)
# For better stability, you can load a separate reference model
use_separate_reference_model = False

# Wandb configuration
wandb_project = "among-them-dpo"
wandb_run_name = f"dpo-beta{dpo_beta}-lr{learning_rate}"

# ============================================================================
# SETUP
# ============================================================================

print("=" * 80)
print("DPO Training Script for Among Them AI")
print("=" * 80)

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Initialize wandb
wandb.init(
    project=wandb_project,
    name=wandb_run_name,
    config={
        "model_name": model_name,
        "max_seq_length": max_seq_length,
        "lora_rank": lora_rank,
        "load_in_4bit": load_in_4bit,
        "dpo_beta": dpo_beta,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": per_device_train_batch_size * gradient_accumulation_steps,
        "max_steps": max_steps,
        "warmup_steps": warmup_steps,
    },
    tags=["dpo", "unsloth", "among-them"],
)

# ============================================================================
# LOAD MODEL AND TOKENIZER
# ============================================================================

print(f"\nLoading model: {model_name}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    load_in_4bit=load_in_4bit,
    dtype=dtype,
    max_seq_length=max_seq_length,
    fast_inference=False,  # Training mode
    gpu_memory_utilization=0.6
)

# Prevent tokenizer from adding duplicate BOS tokens
tokenizer.add_bos_token = False
print(f"Set tokenizer.add_bos_token to: {tokenizer.add_bos_token}")

# ============================================================================
# ATTACH LORA ADAPTERS
# ============================================================================

print("\nAttaching LoRA adapters (QLoRA)...")
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
    use_rslora=False,
    loftq_config=None,
)

# ============================================================================
# SETUP CHAT TEMPLATE
# ============================================================================

# DeepSeek R1 custom template with <think> support
DEEPSEEK_WITH_THINK = r"""
{{ bos_token }}
{% for m in messages %}
    {% set last = loop.index == messages|length %}
    {% if m.role == "user" -%}
        <｜User｜>{{ m.content }}
    {% elif m.role == "assistant" -%}
        <｜Assistant｜>{{ m.content }}{% if not last %}<｜end▁of▁sentence｜>{% endif %}
    {% endif %}
    {% if last and m.role != "assistant" -%}<｜Assistant｜>{% endif %}
{% endfor %}
{% if add_generation_prompt -%}<think>{% endif %}
"""

print("\nSetting up chat template...")
# For DeepSeek models
if "deepseek" in model_name.lower():
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=(DEEPSEEK_WITH_THINK, tokenizer.eos_token),
        map_eos_token=False,
    )
# For Llama models
elif "llama" in model_name.lower():
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )
else:
    print("Warning: Unknown model type, using default chat template")

tokenizer.truncation_side = "left"
tokenizer.model_max_length = max_seq_length

# ============================================================================
# LOAD REFERENCE MODEL (OPTIONAL)
# ============================================================================

ref_model = None
if use_separate_reference_model:
    print("\nLoading separate reference model...")
    ref_model, _ = FastLanguageModel.from_pretrained(
        model_name,
        load_in_4bit=load_in_4bit,
        dtype=dtype,
        max_seq_length=max_seq_length,
        fast_inference=False,
    )
    print("Reference model loaded successfully")
else:
    print("\nUsing implicit reference model (same as training model)")

# ============================================================================
# LOAD DATASET
# ============================================================================

print(f"\nLoading DPO dataset...")

# Copy dataset from drive to local if needed (for Colab)
if not pathlib.Path(local_dataset_dir).exists():
    if pathlib.Path(drive_dataset_dir).exists():
        print(f"Copying dataset from {drive_dataset_dir} → {local_dataset_dir}")
        shutil.copytree(drive_dataset_dir, local_dataset_dir)
    else:
        print(f"Warning: Dataset not found at {drive_dataset_dir}")
        print("Using local dataset directory")
        os.makedirs(local_dataset_dir, exist_ok=True)

dataset_dir = local_dataset_dir

# Custom dataset loading to handle mixed string/list formats
def load_mixed_format_dpo_data(file_path: str, split_name: str):
    """
    Load DPO dataset that may contain both string and list formats.
    Normalizes all data to list format for consistent processing.
    """
    print(f"Loading {split_name} dataset from {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")
    
    if not isinstance(data, list):
        raise ValueError(f"Dataset must be a JSON array, got {type(data)}")
    
    print(f"  Raw data contains {len(data)} examples")
    
    # Normalize all to list format (messages)
    normalized = []
    format_counts = {"string_prompt": 0, "list_prompt": 0, "string_chosen": 0, "list_chosen": 0, "string_rejected": 0, "list_rejected": 0}
    
    for i, item in enumerate(data):
        try:
            # Validate required fields
            if not all(key in item for key in ["prompt", "chosen", "rejected"]):
                raise ValueError(f"Example {i} missing required fields. Found: {list(item.keys())}")
            
            # Normalize prompt
            if isinstance(item["prompt"], str):
                prompt = [{"role": "user", "content": item["prompt"].rstrip()}]
                format_counts["string_prompt"] += 1
            elif isinstance(item["prompt"], list):
                # Validate list format
                if not all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in item["prompt"]):
                    raise ValueError(f"Example {i} prompt list has invalid message format")
                prompt = item["prompt"]
                format_counts["list_prompt"] += 1
            else:
                raise ValueError(f"Example {i} prompt must be string or list, got {type(item['prompt'])}")
            
            # Normalize chosen
            if isinstance(item["chosen"], str):
                chosen = [{"role": "assistant", "content": item["chosen"].rstrip()}]
                format_counts["string_chosen"] += 1
            elif isinstance(item["chosen"], list):
                if not all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in item["chosen"]):
                    raise ValueError(f"Example {i} chosen list has invalid message format")
                chosen = item["chosen"]
                format_counts["list_chosen"] += 1
            else:
                raise ValueError(f"Example {i} chosen must be string or list, got {type(item['chosen'])}")
                
            # Normalize rejected
            if isinstance(item["rejected"], str):
                rejected = [{"role": "assistant", "content": item["rejected"].rstrip()}]
                format_counts["string_rejected"] += 1
            elif isinstance(item["rejected"], list):
                if not all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in item["rejected"]):
                    raise ValueError(f"Example {i} rejected list has invalid message format")
                rejected = item["rejected"]
                format_counts["list_rejected"] += 1
            else:
                raise ValueError(f"Example {i} rejected must be string or list, got {type(item['rejected'])}")
            
            normalized.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            })
            
        except Exception as e:
            print(f"  Warning: Skipping example {i} due to error: {e}")
            continue
    
    print(f"  Normalized to {len(normalized)} valid examples")
    print(f"  Format breakdown: {format_counts}")
    
    return Dataset.from_list(normalized)

# Import json for custom loading
import json
from datasets import Dataset

# Load datasets with mixed format handling
try:
    train_dataset = load_mixed_format_dpo_data(
        os.path.join(dataset_dir, "among_them_dpo_train.json"),
        "train"
    )
    eval_dataset = load_mixed_format_dpo_data(
        os.path.join(dataset_dir, "among_them_dpo_eval.json"),
        "validation"
    )
    print(f"\n✅ Successfully loaded datasets:")
    print(f"   Training: {len(train_dataset)} examples")
    print(f"   Validation: {len(eval_dataset)} examples")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    print("Please ensure your dataset files exist and are in the correct format")
    print("Expected format: JSON array with 'prompt', 'chosen', 'rejected' fields")
    print("Each field can be either a string or a list of message dicts")
    raise

# ============================================================================
# FORMAT DATASET FOR DPO
# ============================================================================

def format_dpo_example(example):
    """
    Convert normalized DPO dataset to the format expected by DPOTrainer.
    
    Since data is already normalized to list format by load_mixed_format_dpo_data(),
    this function simply returns the data as-is.
    
    DPOTrainer expects:
    - prompt: list of message dicts (conversation history)
    - chosen: list of message dicts (preferred completion)
    - rejected: list of message dicts (rejected completion)
    """
    # Data is already in the correct format from our custom loader
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }

print("\nPreparing dataset for DPO training...")
print("Dataset format (already normalized):")
print(f"  Example prompt type: {type(train_dataset[0]['prompt'])}")
print(f"  Example chosen type: {type(train_dataset[0]['chosen'])}")
print(f"  Example rejected type: {type(train_dataset[0]['rejected'])}")
print(f"  Sample prompt: {train_dataset[0]['prompt'][:1]}...")  # Show first message only

# Apply final formatting (mostly identity function now)
train_dataset = train_dataset.map(
    format_dpo_example,
    remove_columns=train_dataset.column_names
)
eval_dataset = eval_dataset.map(
    format_dpo_example,
    remove_columns=eval_dataset.column_names
)

print("\n✅ Dataset ready for DPO training")
print(f"  Training examples: {len(train_dataset)}")
print(f"  Validation examples: {len(eval_dataset)}")

# ============================================================================
# SETUP DPO TRAINER
# ============================================================================

print("\nSetting up DPO trainer...")

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,  # None means use implicit reference
    args=DPOConfig(
        output_dir=lora_weights_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        max_steps=max_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        
        # DPO-specific parameters
        beta=dpo_beta,  # The temperature parameter for DPO loss
        loss_type="sigmoid",  # Options: "sigmoid", "hinge", "ipo", "kto_pair"
        
        # Evaluation and checkpointing
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        
        # Optimizer
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=SEED,
        report_to="wandb",
        
        # Generation config for evaluation
        max_length=max_seq_length,
        max_prompt_length=max_seq_length // 2,
    ),
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=10,
            early_stopping_threshold=0.001
        )
    ],
)

print("\nTrainer configuration:")
print(f"  - Beta (DPO temperature): {dpo_beta}")
print(f"  - Learning rate: {learning_rate}")
print(f"  - Batch size: {per_device_train_batch_size}")
print(f"  - Gradient accumulation: {gradient_accumulation_steps}")
print(f"  - Effective batch size: {per_device_train_batch_size * gradient_accumulation_steps}")
print(f"  - Max steps: {max_steps}")

# ============================================================================
# TRAIN
# ============================================================================

print("\n" + "=" * 80)
print("STARTING DPO TRAINING")
print("=" * 80)

trainer_stats = trainer.train()

print("\n" + "=" * 80)
print("TRAINING COMPLETED")
print("=" * 80)
print(f"Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds")
print(f"Training time: {trainer_stats.metrics['train_runtime']/60:.2f} minutes")

# Log final metrics to wandb
wandb.log({
    "final_train_loss": trainer_stats.metrics.get("train_loss", 0),
    "final_eval_loss": trainer_stats.metrics.get("eval_loss", 0),
    "training_time_seconds": trainer_stats.metrics["train_runtime"],
    "training_time_minutes": trainer_stats.metrics["train_runtime"] / 60,
})

# ============================================================================
# SAVE MODEL
# ============================================================================

print(f"\nSaving LoRA weights to {lora_weights_output_dir}...")
trainer.save_model(lora_weights_output_dir)
tokenizer.save_pretrained(lora_weights_output_dir)

# ============================================================================
# MERGE AND SAVE FULL MODEL
# ============================================================================

print("\nMerging LoRA adapters with base model...")
from peft import PeftModel

base_model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    load_in_4bit=False,
    dtype=dtype,
    max_seq_length=max_seq_length
)

peft_model = PeftModel.from_pretrained(base_model, lora_weights_output_dir)
merged_model = peft_model.merge_and_unload()

print(f"Saving merged model to {merged_model_dir}...")
merged_model.save_pretrained(merged_model_dir, max_shard_size="2GB")
base_tokenizer.save_pretrained(merged_model_dir)

print("\n" + "=" * 80)
print("✅ DPO TRAINING COMPLETE!")
print("=" * 80)
print(f"LoRA weights saved to: {lora_weights_output_dir}")
print(f"Merged model saved to: {merged_model_dir}")

# ============================================================================
# EVALUATION EXAMPLE
# ============================================================================

print("\n" + "=" * 80)
print("QUICK EVALUATION")
print("=" * 80)

FastLanguageModel.for_inference(merged_model)

# Test prompt
test_prompt = """You are playing a social deduction game. Analyze this situation:

A player was found dead in electrical. Three players were in cafeteria together during the kill. 
One player was alone in medbay. Who is most suspicious?"""

messages = [{"role": "user", "content": test_prompt}]

# Test with thinking enabled (for DeepSeek R1)
if "deepseek" in model_name.lower():
    print("\n" + "=" * 80)
    print("TESTING WITH REASONING MODE (enable_thinking=True)")
    print("=" * 80)
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,  # Enable <think> tags
        return_tensors="pt",
    ).to("cuda")
    
    print("\nGenerating response with reasoning...")
    with torch.no_grad():
        outputs = merged_model.generate(
            inputs,
            max_new_tokens=1024,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
        )
    
    response_thinking = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=False)
    print("\nModel response (with thinking):")
    print(response_thinking)
    
    # Log to wandb
    wandb.log({
        "example_with_thinking": wandb.Html(f"<pre>{response_thinking}</pre>")
    })

# Test without thinking (standard chat mode)
print("\n" + "=" * 80)
print("TESTING WITHOUT REASONING MODE (enable_thinking=False)")
print("=" * 80)

if "deepseek" in model_name.lower():
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,  # Disable <think> tags
        return_tensors="pt",
    ).to("cuda")
else:
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

print("\nGenerating response...")
with torch.no_grad():
    outputs = merged_model.generate(
        inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
    )

response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=False)
print("\nModel response (without thinking):")
print(response)

# Log to wandb
wandb.log({
    "example_without_thinking": wandb.Html(f"<pre>{response}</pre>")
})

print("\n" + "=" * 80)
print("SCRIPT COMPLETE")
print("=" * 80)

# Finish wandb run
wandb.finish()
print("\nWandB run finished. View results at:", wandb.run.url if wandb.run else "N/A")
