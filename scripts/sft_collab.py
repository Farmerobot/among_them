# Global Configuration
DEBUG = False  # Set to True to print detailed formatting and masking examples
BASE_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MAX_SEQ_LENGTH = 7000
LORA_RANK = 8  # Reduced from 16 to save memory (4x fewer parameters)
DTYPE = None
LOAD_IN_4BIT = True

# Dataset and output paths for both configurations
CONFIGS = {
    "sampled": {
        "drive_dataset_dir": "/content/drive/MyDrive/among_them/sampled/among_them_dataset",
        "local_dataset_dir": "/content/sampled/among_them_dataset",
        "lora_weights_dir": "/content/drive/MyDrive/among_them/sampled/among_them_lora_checkpoints",
        "merged_model_dir": "/content/drive/MyDrive/among_them/sampled/deepseek_r1_merged",
        "hf_gguf_repo_id": "Farmerobot/deepseek-r1-among-them-gguf",
        "cache_dir": "/content/sampled/datasets_cache"
    },
    "unsampled": {
        "drive_dataset_dir": "/content/drive/MyDrive/among_them/unsampled/among_them_dataset",
        "local_dataset_dir": "/content/unsampled/among_them_dataset",
        "lora_weights_dir": "/content/drive/MyDrive/among_them/unsampled/among_them_lora_checkpoints",
        "merged_model_dir": "/content/drive/MyDrive/among_them/unsampled/deepseek_r1_merged",
        "hf_gguf_repo_id": "Farmerobot/deepseek-r1-among-them-unsampled-gguf",
        "cache_dir": "/content/unsampled/datasets_cache"
    }
}

def copy_dataset_to_local(drive_dir, local_dir):
    """Copy dataset from Google Drive to local Colab VM for faster access."""
    if os.path.exists(local_dir):
        print(f"Removing existing local copy at {local_dir}...")
        shutil.rmtree(local_dir)
    
    print(f"Copying dataset from {drive_dir} to {local_dir}...")
    shutil.copytree(drive_dir, local_dir)
    print("Copy complete.")
    return local_dir

def load_and_prepare_dataset(dataset_dir, cache_dir):
    """Load and prepare the dataset for training."""
    print(f"Loading datasets from {dataset_dir}...")

    # Load datasets with conversations format (no feature enforcement)
    raw_datasets = load_dataset(
        "json",
        data_files={
            "train": os.path.join(dataset_dir, "among_them_train.json"),
            "validation": os.path.join(dataset_dir, "among_them_eval.json")
        },
        keep_in_memory=False,  # Let datasets library manage caching
        cache_dir=cache_dir,
    )

    return raw_datasets["train"], raw_datasets["validation"]

def validate_assistant_messages(dataset, dataset_name="dataset"):
    """Validate assistant message structure in the dataset.
    
    Checks:
    1. Last assistant message always has a <think> block
    2. Non-last assistant messages never have <think> blocks
    3. All assistant messages have content outside <think> blocks
    """
    import re
    
    print(f"\nValidating assistant messages in {dataset_name}...")
    
    errors = []
    warnings = []
    stats = {
        "total_examples": 0,
        "total_assistant_messages": 0,
        "last_with_think": 0,
        "non_last_without_think": 0,
    }
    
    for idx, example in enumerate(dataset):
        stats["total_examples"] += 1
        conversations = example["conversations"]
        
        # Get all assistant messages with their indices
        assistant_msgs = [
            (i, msg) for i, msg in enumerate(conversations) 
            if msg.get("role") == "assistant"
        ]
        
        if not assistant_msgs:
            warnings.append(f"Example {idx}: No assistant messages found")
            continue
        
        stats["total_assistant_messages"] += len(assistant_msgs)
        
        for msg_idx, (conv_idx, msg) in enumerate(assistant_msgs):
            content = msg.get("content", "") or ""
            is_last = msg_idx == len(assistant_msgs) - 1
            
            has_think_open = "<think>" in content
            has_think_close = "</think>" in content
            has_think_block = has_think_open and has_think_close
            
            # Extract content outside think block
            content_outside_think = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            
            if is_last:
                # Last assistant message must have think block
                if not has_think_block:
                    errors.append(
                        f"Example {idx}, assistant msg {msg_idx} (last): "
                        f"Missing <think> block"
                    )
                else:
                    stats["last_with_think"] += 1
                
                # Check for mismatched tags
                if has_think_open != has_think_close:
                    errors.append(
                        f"Example {idx}, assistant msg {msg_idx} (last): "
                        f"Mismatched think tags (open={has_think_open}, close={has_think_close})"
                    )
            else:
                # Non-last assistant messages must NOT have think block
                if has_think_open or has_think_close:
                    errors.append(
                        f"Example {idx}, assistant msg {msg_idx}: "
                        f"Unexpected <think> block in non-last assistant message"
                    )
                else:
                    stats["non_last_without_think"] += 1
            
            # All assistant messages must have content outside think block
            if not content_outside_think:
                errors.append(
                    f"Example {idx}, assistant msg {msg_idx}: "
                    f"No content outside <think> block"
                )
    
    # Print summary
    print(f"  Total examples: {stats['total_examples']}")
    print(f"  Total assistant messages: {stats['total_assistant_messages']}")
    print(f"  Last messages with think block: {stats['last_with_think']}")
    print(f"  Non-last messages without think: {stats['non_last_without_think']}")
    
    if warnings:
        print(f"\n  Warnings ({len(warnings)}):")
        for w in warnings[:5]:
            print(f"    - {w}")
        if len(warnings) > 5:
            print(f"    ... and {len(warnings) - 5} more")
    
    if errors:
        print(f"\n  ❌ Errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"    - {e}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more")
        raise ValueError(f"Dataset validation failed with {len(errors)} errors. Fix the dataset before training.")
    
    print(f"  ✓ All assistant messages validated successfully!")
    return True

def prepare_model_and_tokenizer(model_name, max_seq_length, load_in_4bit, dtype):
    """Load model and tokenizer with LoRA configuration."""
    print(f"Loading model {model_name}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        load_in_4bit=load_in_4bit,
        dtype=dtype,
        max_seq_length=max_seq_length,
        gpu_memory_utilization=0.6,  # Reserve 40% for training overhead
    )

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_RANK,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # DeepSeek R1 uses its own chat template - don't override it
    # The model's tokenizer already has the correct template from HuggingFace
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Disable automatic BOS token addition since the chat template already includes it
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
    
    # CRITICAL: Set model_max_length to prevent truncation during train_on_responses_only
    # The tokenizer defaults to 1024, but our sequences are longer
    tokenizer.model_max_length = max_seq_length
    
    # Override chat template to PRESERVE <think> blocks during training
    # The default template strips thinking via: content.split("</think>")[-1]
    # We remove those lines to keep the full reasoning process
    training_chat_template = """
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
			{#- MODIFIED: Keep full content including <think> blocks for training -#}
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
    
    tokenizer.chat_template = training_chat_template
    
    if DEBUG:
        print("✓ Modified chat template to preserve <think> blocks during training")

    return model, tokenizer

def format_dataset(example, tokenizer):
    """Format examples using chat template matching in-game inference.
    
    Applies the DeepSeek R1 chat template with special tokens like
    <｜User｜>, <｜Assistant｜>, <｜end▁of▁sentence｜> etc.
    This matches what the model sees during gameplay via tokenizer.apply_chat_template()
    """
    conversations = example["conversations"]
    
    # Apply chat template without special tokens (BOS/EOS)
    # SFTTrainer will add BOS/EOS during tokenization to avoid duplicates
    # add_generation_prompt=False because we have the full conversation including assistant responses
    formatted_text = tokenizer.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False
    )
    
    # DEBUG: Verify <think> blocks are preserved (only check once)
    if DEBUG and not hasattr(format_dataset, "_verified"):
        has_think_raw = any("<think>" in msg.get("content", "") for msg in conversations if msg.get("role") == "assistant")
        has_think_formatted = "<think>" in formatted_text
        
        if has_think_raw:
            if has_think_formatted:
                think_idx = formatted_text.find("<think>")
                snippet = formatted_text[think_idx:think_idx+100]
                print(f"\n✓ <think> blocks PRESERVED in formatted output!")
                print(f"  Sample snippet: {snippet}...")
            else:
                print(f"\n❌ WARNING: <think> blocks were REMOVED!")
                print(f"  This means the chat template modification didn't work.")
        format_dataset._verified = True
    
    return {"text": formatted_text}

def mask_all_but_last_response(trainer, assistant_token="<｜Assistant｜>"):
    """Mask all tokens except the last assistant response.
    
    In multi-turn conversations where past assistant messages had their think blocks
    removed, we only want to train on the final response (which has the think block).
    All prior messages (user AND past assistant) are kept for context but masked.
    """
    tokenizer = trainer.tokenizer
    
    # Get the token ID for the assistant marker
    assistant_ids = tokenizer.encode(assistant_token, add_special_tokens=False)
    if len(assistant_ids) != 1:
        raise ValueError(f"Expected single token for '{assistant_token}', got {len(assistant_ids)} tokens: {assistant_ids}")
    assistant_token_id = assistant_ids[0]
    
    def mask_example(example):
        input_ids = example["input_ids"]
        
        # Find all positions of the assistant token
        assistant_positions = [i for i, token_id in enumerate(input_ids) if token_id == assistant_token_id]
        
        if not assistant_positions:
            # No assistant responses found - mask everything
            return {"labels": [-100] * len(input_ids)}
        
        # Get the position of the LAST assistant token
        last_assistant_pos = assistant_positions[-1]
        
        # Create labels: mask everything before last assistant, train on the rest
        # We also mask the assistant token itself (train starts after it)
        labels = []
        for i, token_id in enumerate(input_ids):
            if i <= last_assistant_pos:
                labels.append(-100)  # Masked - not trained on
            else:
                labels.append(token_id)  # Train on this token
        
        return {"labels": labels}
    
    # Apply masking to train dataset
    trainer.train_dataset = trainer.train_dataset.map(
        mask_example,
        batched=False,
        desc="Masking all but last response"
    )
    
    # Apply masking to eval dataset if present
    if trainer.eval_dataset is not None:
        trainer.eval_dataset = trainer.eval_dataset.map(
            mask_example,
            batched=False,
            desc="Masking eval dataset"
        )
    
    # Set data collator that properly pads labels with -100
    trainer.data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )
    
    return trainer

def train_model(model, tokenizer, train_dataset, eval_dataset, output_dir, max_seq_length):
    """Train the model using SFTTrainer with last-response-only training.
    
    Only trains on the last assistant response while keeping all prior messages
    (both user and past assistant) as masked context.
    """
    training_start_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"Training started at: {training_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    print("Setting up trainer...")

    # Format datasets (converts conversations to text)
    train_dataset = train_dataset.map(
        lambda x: format_dataset(x, tokenizer),
        batched=False
    )
    eval_dataset = eval_dataset.map(
        lambda x: format_dataset(x, tokenizer),
        batched=False
    )
    
    # Filter out sequences that are too long to prevent OOM
    def filter_by_length(example):
        tokens = tokenizer.encode(example["text"], add_special_tokens=False)
        return len(tokens) <= max_seq_length
    
    original_train_size = len(train_dataset)
    original_eval_size = len(eval_dataset)
    
    train_dataset = train_dataset.filter(filter_by_length)
    eval_dataset = eval_dataset.filter(filter_by_length)
    
    print(f"\nFiltered dataset by length (max_seq_length={max_seq_length}):")
    print(f"  Train: {original_train_size} → {len(train_dataset)} examples ({original_train_size - len(train_dataset)} removed)")
    print(f"  Eval: {original_eval_size} → {len(eval_dataset)} examples ({original_eval_size - len(eval_dataset)} removed)")
    
    # Debug: Print first formatted example to verify format
    if DEBUG:
        print("\n" + "="*60)
        print("FORMATTED TEXT SAMPLE:")
        print("="*60)
        print(train_dataset[0]["text"])
        print("="*60 + "\n")
    
    # Debug tokenization of special tokens and patterns
    print("DEBUG: Tokenizer special tokens:")
    print(f"  User token alone: {tokenizer.encode('<｜User｜>', add_special_tokens=False)}")
    print(f"  User token + newline: {tokenizer.encode('<｜User｜>\n', add_special_tokens=False)}")
    print(f"  Assistant token alone: {tokenizer.encode('<｜Assistant｜>', add_special_tokens=False)}")
    print(f"  Assistant token + newline: {tokenizer.encode('<｜Assistant｜>\n', add_special_tokens=False)}")
    print(f"  EOS token: {tokenizer.encode('<｜end▁of▁sentence｜>', add_special_tokens=False)}")
    
    # Check sequence length
    full_text = train_dataset[0]["text"]
    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
    print(f"\nSequence length check:")
    print(f"  Full text length: {len(full_text)} chars")
    print(f"  Full token count: {len(full_tokens)} tokens")
    print(f"  Max seq length setting: {max_seq_length} tokens")
    print(f"  Tokenizer model_max_length: {tokenizer.model_max_length} tokens")
    if len(full_tokens) > max_seq_length:
        print(f"  ⚠️ WARNING: Sequence will be TRUNCATED by {len(full_tokens) - max_seq_length} tokens")
    print("")

    # CRITICAL: Unsloth's SFTTrainer tokenization uses dataset_num_proc which defaults to high values
    # This can cause issues. We need to ensure it uses max_seq_length correctly.
    # Use SFTConfig instead of TrainingArguments for better compatibility
    
    training_args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        max_seq_length=max_seq_length,  # CRITICAL: Must be in SFTConfig for Unsloth
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=5,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=10,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        seed=42,
        report_to="none",
        dataset_num_proc=2,
        packing=False,  # Must be False for custom label masking to work
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Setup trainer (following official Unsloth pattern)
    # Note: Don't use custom data_collator - SFTTrainer's default handles label padding correctly
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10, early_stopping_threshold=0)],
    )

    # Apply custom masking - only train on the LAST assistant response
    # Past assistant messages (without think blocks) are masked like user messages
    print("\nApplying last-response-only masking...")
    trainer = mask_all_but_last_response(
        trainer,
        assistant_token="<｜Assistant｜>",
    )
    
    # Verify masking worked
    print("\nVerifying masking (first example):")
    labels = trainer.train_dataset[0]["labels"]
    total_tokens = len(labels)
    masked_tokens = sum(1 for x in labels if x == -100)
    trainable_tokens = total_tokens - masked_tokens
    
    print(f"  Total tokens: {total_tokens}")
    print(f"  Masked tokens (-100): {masked_tokens} ({masked_tokens/total_tokens*100:.1f}%)")
    print(f"  Trainable tokens: {trainable_tokens} ({trainable_tokens/total_tokens*100:.1f}%)")
    
    if trainable_tokens == 0:
        print("\n  ❌ ERROR: All tokens masked!")
    elif masked_tokens == 0:
        print("\n  ⚠️ WARNING: No tokens masked (training on full conversation)")
    else:
        print("\n  ✓ Masking successful - training on last assistant response only")
    
    # Debug: Print full masked example with visible masking
    if DEBUG:
        print("\nMASKED EXAMPLE:")
        print("="*60)
        input_ids = trainer.train_dataset[0]["input_ids"]
        labels = trainer.train_dataset[0]["labels"]
        
        # Build output showing masked regions
        output_parts = []
        i = 0
        while i < len(labels):
            if labels[i] == -100:
                # Count consecutive masked tokens
                masked_count = 0
                while i < len(labels) and labels[i] == -100:
                    masked_count += 1
                    i += 1
                output_parts.append(f"<{masked_count} masked tokens>")
            else:
                # Decode visible token
                output_parts.append(tokenizer.decode([input_ids[i]], skip_special_tokens=False))
                i += 1
        
        print("".join(output_parts))
        print("="*60)
    
    print("="*60 + "\n")

    print("Starting training...")
    trainer.train()
    
    training_end_time = datetime.now()
    training_duration = training_end_time - training_start_time
    
    print(f"\n{'='*60}")
    print(f"Training completed at: {training_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {training_duration}")
    print(f"{'='*60}\n")

    return trainer

def merge_and_save_model(model_name, lora_weights_dir, output_dir, max_seq_length, dtype):
    """Merge LoRA weights with base model and save."""
    print("Merging LoRA adapters...")

    base_model, base_tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        load_in_4bit=False,
        dtype=dtype,
        max_seq_length=max_seq_length
    )

    peft_model = PeftModel.from_pretrained(base_model, lora_weights_dir)
    merged_model = peft_model.merge_and_unload()

    print(f"Saving merged model to {output_dir}...")
    merged_model.save_pretrained(output_dir, max_shard_size="2GB")
    base_tokenizer.save_pretrained(output_dir)

    print("Optimizing model for inference...")
    FastLanguageModel.for_inference(merged_model)

    print("Model merged and saved successfully!")
    return merged_model, base_tokenizer

def convert_local_to_gguf_and_upload(local_model_dir, dst_repo, outtype="f16"):
    """Convert local HF model to GGUF and upload to HuggingFace."""
    print(f"🔄 Converting {local_model_dir} to GGUF...")

    workdir = Path.cwd()
    llama_cpp = workdir / "llama.cpp"

    # Clone llama.cpp if needed
    if not llama_cpp.exists():
        print("Cloning llama.cpp...")
        !git clone https://github.com/ggml-org/llama.cpp

    gguf_file = workdir / f"deepseek-r1-{outtype}.gguf"

    # Convert to GGUF directly from local directory
    print(f"Converting to GGUF ({outtype})...")
    subprocess.run([
        sys.executable,
        str(llama_cpp / "convert_hf_to_gguf.py"),
        str(local_model_dir),
        "--outfile", str(gguf_file),
        "--outtype", outtype
    ], check=True)

    # Create repository if it doesn't exist
    api = HfApi()
    try:
        print(f"Checking if repository {dst_repo} exists...")
        api.repo_info(repo_id=dst_repo, repo_type="model")
        print(f"Repository {dst_repo} already exists.")
    except Exception:
        print(f"Repository {dst_repo} not found. Creating it...")
        api.create_repo(
            repo_id=dst_repo,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        print(f"✅ Created repository {dst_repo}")

    # Upload to HF
    print(f"🚀 Uploading to {dst_repo}...")
    api.upload_file(
        path_or_fileobj=str(gguf_file),
        path_in_repo=gguf_file.name,
        repo_id=dst_repo,
        repo_type="model",
    )

    print(f"✅ Done! File is live at https://huggingface.co/{dst_repo}")

    # Cleanup
    if gguf_file.exists():
        gguf_file.unlink()

def train_and_save_pipeline(config_name):
    """Complete training pipeline for a given configuration."""
    print(f"\n{'='*60}")
    print(f"Starting training for: {config_name}")
    print(f"{'='*60}\n")

    config = CONFIGS[config_name]

    # Copy dataset to local
    dataset_dir = copy_dataset_to_local(
        config["drive_dataset_dir"],
        config["local_dataset_dir"]
    )

    # Load datasets
    train_dataset, eval_dataset = load_and_prepare_dataset(
        dataset_dir,
        config["cache_dir"]
    )

    # Validate assistant message structure
    validate_assistant_messages(train_dataset, "train")
    validate_assistant_messages(eval_dataset, "validation")

    # Prepare model
    model, tokenizer = prepare_model_and_tokenizer(
        BASE_MODEL_NAME,
        MAX_SEQ_LENGTH,
        LOAD_IN_4BIT,
        DTYPE
    )

    # Train
    trainer = train_model(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        config["lora_weights_dir"],
        MAX_SEQ_LENGTH
    )

    # Save LoRA weights
    print(f"Saving LoRA weights to {config['lora_weights_dir']}...")
    trainer.save_model(config["lora_weights_dir"])

    # Merge and save full model
    merged_model, merged_tokenizer = merge_and_save_model(
        BASE_MODEL_NAME,
        config["lora_weights_dir"],
        config["merged_model_dir"],
        MAX_SEQ_LENGTH,
        DTYPE
    )

    # Convert to GGUF and upload (skip HF upload of non-GGUF format)
    convert_local_to_gguf_and_upload(
        config["merged_model_dir"],
        config["hf_gguf_repo_id"]
    )

    print(f"\n{'='*60}")
    print(f"Completed training for: {config_name}")
    print(f"{'='*60}\n")

train_and_save_pipeline("unsampled")
# train_and_save_pipeline("sampled")