#!/usr/bin/env python3
"""
SFT training script for SLURM clusters.

Trains on multi-turn conversations with gradients only on the last assistant response.
Uses environment variables for paths to work seamlessly on clusters.
"""

import json
import os
from datetime import datetime
from pathlib import Path

from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from peft import PeftModel
import wandb

def _require_env(name: str) -> str:
    """Get required environment variable or fail fast."""
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value


# Configuration - all values must be set explicitly via environment
DEBUG = _require_env("SFT_DEBUG").lower() == "true"
BASE_MODEL_NAME = _require_env("SFT_BASE_MODEL")
MAX_SEQ_LENGTH = int(_require_env("SFT_MAX_SEQ_LENGTH"))
LORA_RANK = int(_require_env("SFT_LORA_RANK"))
DTYPE = None
LOAD_IN_4BIT = _require_env("SFT_LOAD_IN_4BIT").lower() == "true"
NUM_EPOCHS = int(_require_env("SFT_NUM_EPOCHS"))
BATCH_SIZE = int(_require_env("SFT_BATCH_SIZE"))
GRAD_ACCUM_STEPS = int(_require_env("SFT_GRAD_ACCUM_STEPS"))
LEARNING_RATE = float(_require_env("SFT_LEARNING_RATE"))
WANDB_PROJECT = _require_env("SFT_WANDB_PROJECT")
WANDB_RUN_NAME = _require_env("SFT_WANDB_RUN_NAME")

# Paths from environment
DATA_DIR = Path(_require_env("SFT_DATA_DIR"))
OUTPUT_DIR = Path(_require_env("SFT_OUTPUT_DIR"))
MERGED_MODEL_DIR = Path(_require_env("SFT_MERGED_MODEL_DIR"))


def load_datasets(data_dir: Path):
    """Load train and eval datasets."""
    print(f"Loading datasets from {data_dir}...")
    
    train_file = data_dir / "among_them_train.json"
    eval_file = data_dir / "among_them_eval.json"
    
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not eval_file.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_file}")
    
    raw_datasets = load_dataset(
        "json",
        data_files={
            "train": str(train_file),
            "validation": str(eval_file),
        },
        keep_in_memory=False,
    )
    
    print(f"  Train: {len(raw_datasets['train'])} examples")
    print(f"  Eval: {len(raw_datasets['validation'])} examples")
    
    return raw_datasets["train"], raw_datasets["validation"]


def validate_dataset(dataset, name: str):
    """Validate that last assistant message has <think> block, others don't."""

    print(f"\nValidating {name}...")
    errors = []
    
    for idx, example in enumerate(dataset):
        conversations = example["conversations"]
        assistant_msgs = [
            (i, msg) for i, msg in enumerate(conversations)
            if msg.get("role") == "assistant"
        ]
        
        if not assistant_msgs:
            continue
        
        for msg_idx, (_, msg) in enumerate(assistant_msgs):
            content = msg.get("content", "") or ""
            is_last = msg_idx == len(assistant_msgs) - 1
            has_think = "<think>" in content and "</think>" in content
            
            if is_last and not has_think:
                errors.append(f"Example {idx}: Last assistant missing <think>")
            elif not is_last and has_think:
                errors.append(f"Example {idx}: Non-last assistant has <think>")
    
    if errors:
        for e in errors[:5]:
            print(f"  ERROR: {e}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")
        raise ValueError(f"Validation failed with {len(errors)} errors")
    
    print(f"  ✓ Validated {len(dataset)} examples")


def prepare_model_and_tokenizer():
    """Load model and tokenizer with LoRA."""
    print(f"\nLoading model: {BASE_MODEL_NAME}")
    print(f"  MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}")
    print(f"  LORA_RANK: {LORA_RANK}")
    print(f"  LOAD_IN_4BIT: {LOAD_IN_4BIT}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        BASE_MODEL_NAME,
        load_in_4bit=LOAD_IN_4BIT,
        dtype=DTYPE,
        max_seq_length=MAX_SEQ_LENGTH,
        gpu_memory_utilization=0.6,
    )
    
    # LoRA alpha = 2*r is recommended for better learning dynamics
    # Lower dropout (0.05) for small models per SOTA guidelines
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
    tokenizer.model_max_length = MAX_SEQ_LENGTH
    
    # Chat template that preserves <think> blocks
    tokenizer.chat_template = TRAINING_CHAT_TEMPLATE
    
    return model, tokenizer


def format_example(example, tokenizer):
    """Format conversation using chat template."""
    formatted_text = tokenizer.apply_chat_template(
        example["conversations"],
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False,
    )
    return {"text": formatted_text}


def mask_all_but_last_response(trainer, assistant_token="<｜Assistant｜>"):
    """Mask all tokens except the last assistant response for training."""
    tokenizer = trainer.tokenizer
    
    assistant_ids = tokenizer.encode(assistant_token, add_special_tokens=False)
    if len(assistant_ids) != 1:
        raise ValueError(f"Expected single token for '{assistant_token}', got {len(assistant_ids)}")
    assistant_token_id = assistant_ids[0]
    
    def mask_example(example):
        input_ids = example["input_ids"]
        assistant_positions = [i for i, tid in enumerate(input_ids) if tid == assistant_token_id]
        
        if not assistant_positions:
            return {"labels": [-100] * len(input_ids)}
        
        last_pos = assistant_positions[-1]
        labels = [-100 if i <= last_pos else tid for i, tid in enumerate(input_ids)]
        return {"labels": labels}
    
    trainer.train_dataset = trainer.train_dataset.map(
        mask_example, batched=False, desc="Masking train dataset"
    )
    
    if trainer.eval_dataset is not None:
        trainer.eval_dataset = trainer.eval_dataset.map(
            mask_example, batched=False, desc="Masking eval dataset"
        )
    
    trainer.data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )
    
    return trainer


def train(model, tokenizer, train_dataset, eval_dataset):
    """Run SFT training."""
    start_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"Training started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # Format datasets
    train_dataset = train_dataset.map(
        lambda x: format_example(x, tokenizer), batched=False
    )
    eval_dataset = eval_dataset.map(
        lambda x: format_example(x, tokenizer), batched=False
    )
    
    # Filter by length
    def filter_by_length(example):
        tokens = tokenizer.encode(example["text"], add_special_tokens=False)
        return len(tokens) <= MAX_SEQ_LENGTH
    
    orig_train = len(train_dataset)
    orig_eval = len(eval_dataset)
    train_dataset = train_dataset.filter(filter_by_length)
    eval_dataset = eval_dataset.filter(filter_by_length)
    
    print(f"Filtered by length (max={MAX_SEQ_LENGTH}):")
    print(f"  Train: {orig_train} → {len(train_dataset)}")
    print(f"  Eval: {orig_eval} → {len(eval_dataset)}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=20,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        seed=42,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        dataset_num_proc=2,
        packing=False,  # Required for custom label masking
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )
    
    # Apply last-response-only masking
    print("\nApplying last-response-only masking...")
    trainer = mask_all_but_last_response(trainer)
    
    # Verify masking
    labels = trainer.train_dataset[0]["labels"]
    total = len(labels)
    masked = sum(1 for x in labels if x == -100)
    print(f"  Total tokens: {total}")
    print(f"  Masked: {masked} ({masked/total*100:.1f}%)")
    print(f"  Trainable: {total - masked} ({(total-masked)/total*100:.1f}%)")
    
    print("\nStarting training...")
    trainer.train()
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Save training stats to disk
    stats_file = OUTPUT_DIR / "training_stats.json"
    training_stats = {
        "config": {
            "base_model": BASE_MODEL_NAME,
            "max_seq_length": MAX_SEQ_LENGTH,
            "lora_rank": LORA_RANK,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "learning_rate": LEARNING_RATE,
            "load_in_4bit": LOAD_IN_4BIT,
        },
        "dataset": {
            "train_examples": len(train_dataset),
            "eval_examples": len(eval_dataset),
            "original_train": orig_train,
            "original_eval": orig_eval,
        },
        "timing": {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "duration_seconds": duration.total_seconds(),
        },
        "log_history": trainer.state.log_history,
    }
    
    with open(stats_file, "w") as f:
        json.dump(training_stats, f, indent=2)
    print(f"\nSaved training stats to {stats_file}")
    
    print(f"\n{'='*60}")
    print(f"Training completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print(f"{'='*60}\n")
    
    return trainer


def merge_and_save(lora_dir: Path, output_dir: Path):
    """Merge LoRA weights with base model."""
    print(f"\nMerging LoRA weights from {lora_dir}...")
    
    base_model, base_tokenizer = FastLanguageModel.from_pretrained(
        BASE_MODEL_NAME,
        load_in_4bit=False,
        dtype=DTYPE,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    
    peft_model = PeftModel.from_pretrained(base_model, str(lora_dir))
    merged_model = peft_model.merge_and_unload()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {output_dir}...")
    merged_model.save_pretrained(str(output_dir), max_shard_size="2GB")
    base_tokenizer.save_pretrained(str(output_dir))
    
    print("✓ Model merged and saved")


def main():
    print("="*60)
    print("SFT Training (SLURM)")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  BASE_MODEL: {BASE_MODEL_NAME}")
    print(f"  MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}")
    print(f"  LORA_RANK: {LORA_RANK}")
    print(f"  NUM_EPOCHS: {NUM_EPOCHS}")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print(f"  GRAD_ACCUM_STEPS: {GRAD_ACCUM_STEPS}")
    print(f"  LEARNING_RATE: {LEARNING_RATE}")
    print(f"\nPaths:")
    print(f"  DATA_DIR: {DATA_DIR}")
    print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"  MERGED_MODEL_DIR: {MERGED_MODEL_DIR}")
    
    # Initialize wandb
    if os.environ.get("WANDB_API_KEY"):
        wandb.login()
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            config={
                "base_model": BASE_MODEL_NAME,
                "max_seq_length": MAX_SEQ_LENGTH,
                "lora_rank": LORA_RANK,
                "num_epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "grad_accum_steps": GRAD_ACCUM_STEPS,
                "learning_rate": LEARNING_RATE,
                "load_in_4bit": LOAD_IN_4BIT,
            }
        )
        print(f"\nWandB: {WANDB_PROJECT}/{WANDB_RUN_NAME}")
    else:
        print("\nWandB: disabled (no WANDB_API_KEY)")
    
    # Load and validate data
    train_dataset, eval_dataset = load_datasets(DATA_DIR)
    validate_dataset(train_dataset, "train")
    validate_dataset(eval_dataset, "eval")
    
    # Prepare model
    model, tokenizer = prepare_model_and_tokenizer()
    
    # Train
    trainer = train(model, tokenizer, train_dataset, eval_dataset)
    
    # Save LoRA weights
    print(f"\nSaving LoRA weights to {OUTPUT_DIR}...")
    trainer.save_model(str(OUTPUT_DIR))
    
    # Merge and save full model
    merge_and_save(OUTPUT_DIR, MERGED_MODEL_DIR)
    
    # Finish wandb
    if os.environ.get("WANDB_API_KEY"):
        wandb.finish()
    
    print("\n" + "="*60)
    print("SFT Training Complete!")
    print("="*60)


# Chat template that preserves <think> blocks during training
TRAINING_CHAT_TEMPLATE = """
{%- if not add_generation_prompt is defined -%}
	{%- set add_generation_prompt = false -%}
{%- endif -%}
{%- set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt="") -%}
{%- for message in messages -%}
	{%- if message["role"] == "system" -%}
		{%- set ns.system_prompt = message["content"] -%}
	{%- endif -%}
{%- endfor -%}
{{- bos_token -}}
{{- ns.system_prompt -}}
{%- for message in messages -%}
	{%- if message["role"] == "user" -%}
		{%- set ns.is_tool = false -%}
		{{- "<｜User｜>" + message["content"] -}}
	{%- endif -%}
	{%- if message["role"] == "assistant" and message["content"] is none -%}
		{%- set ns.is_tool = false -%}
		{%- for tool in message["tool_calls"] -%}
			{%- if not ns.is_first -%}
				{{- "<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>" + tool["type"] + "<｜tool▁sep｜>" + tool["function"]["name"] + "\\n" + "```json" + "\\n" + tool["function"]["arguments"] + "\\n" + "```" + "<｜tool▁call▁end｜>" -}}
				{%- set ns.is_first = true -%}
			{%- else -%}
				{{- "\\n" + "<｜tool▁call▁begin｜>" + tool["type"] + "<｜tool▁sep｜>" + tool["function"]["name"] + "\\n" + "```json" + "\\n" + tool["function"]["arguments"] + "\\n" + "```" + "<｜tool▁call▁end｜>" -}}
				{{- "<｜tool▁calls▁end｜><｜end▁of▁sentence｜>" -}}
			{%- endif -%}
		{%- endfor -%}
	{%- endif -%}
	{%- if message["role"] == "assistant" and message["content"] is not none -%}
		{%- if ns.is_tool -%}
			{{- "<｜tool▁outputs▁end｜>" + message["content"] + "<｜end▁of▁sentence｜>" -}}
			{%- set ns.is_tool = false -%}
		{%- else -%}
			{{- "<｜Assistant｜>" + message["content"] + "<｜end▁of▁sentence｜>" -}}
		{%- endif -%}
	{%- endif -%}
	{%- if message["role"] == "tool" -%}
		{%- set ns.is_tool = true -%}
		{%- if ns.is_output_first -%}
			{{- "<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>" + message["content"] + "<｜tool▁output▁end｜>" -}}
			{%- set ns.is_output_first = false -%}
		{%- else -%}
			{{- "\\n<｜tool▁output▁begin｜>" + message["content"] + "<｜tool▁output▁end｜>" -}}
		{%- endif -%}
	{%- endif -%}
{%- endfor -%}
{%- if ns.is_tool -%}
	{{- "<｜tool▁outputs▁end｜>" -}}
{%- endif -%}
{%- if add_generation_prompt and not ns.is_tool -%}
	{{- "<｜Assistant｜><think>\\n" -}}
{%- endif -%}
"""


if __name__ == "__main__":
    main()
