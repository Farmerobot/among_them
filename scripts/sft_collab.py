# Prepare dataset (Alpaca format: 'instruction' and 'output' fields)
print(f"Loading datasets from {dataset_dir}...")
features = Features({
    "instruction": Value("string"),
    "output": Value("string")
})
raw_datasets = load_dataset(
    "json",
    data_files={
        "train": os.path.join(dataset_dir, "among_them_train.json"),
        "validation": os.path.join(dataset_dir, "among_them_eval.json")
    },
    features=features
)
train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["validation"]

# Define formatting function for Alpaca
def formatting_prompts_func(example):
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return example

# Prevent the tokenizer from adding a *second* BOS token automatically
# The chat template already includes it.
tokenizer.add_bos_token = False
print(f"Set tokenizer.add_bos_token to: {tokenizer.add_bos_token}") # Verify change

columns_to_remove = ["instruction", "output"]

train_dataset = train_dataset.map(formatting_prompts_func, remove_columns=columns_to_remove)
eval_dataset = eval_dataset.map(formatting_prompts_func, remove_columns=columns_to_remove)

# Setup SFT trainer (only train on the assistant's outputs)
print("Setting up trainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        output_dir=lora_weights_output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=1000,
        learning_rate=2e-4,
        warmup_steps=100,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        save_steps=200,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
    )
)

# Mask loss to only train on assistant responses
trainer = train_on_responses_only(
    trainer=trainer,
    tokenizer=tokenizer,
    instruction_part="<｜User｜>",
    response_part="<｜Assistant｜>",
)

# Verify masking on trainer dataset
print("Decoding sample 5 input_ids after masking:")
print(tokenizer.decode(trainer.train_dataset[0]["input_ids"], skip_special_tokens=False))
if "labels" in trainer.train_dataset[0]:  
    label_ids = [tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[0]["labels"]]
    print("Decoded labels:", tokenizer.decode(label_ids, skip_special_tokens=False))

# Train LoRA adapters
print("Starting training...")
trainer.train()

# Save LoRA weights
print(f"Saving LoRA weights to {lora_weights_output_dir}...")
trainer.save_model(lora_weights_output_dir)

# Merge LoRA back into base model
print("Merging LoRA adapters...")
base_model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    load_in_4bit=False,
    dtype=dtype,
    max_seq_length=max_seq_length
)
peft_model = PeftModel.from_pretrained(base_model, lora_weights_output_dir)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(merged_model_dir, max_shard_size="2GB")

# Optimize for inference
print("Optimizing model for inference...")
FastLanguageModel.for_inference(merged_model)

print(f"Merged model saved to {merged_model_dir} and ready for inference.")

from huggingface_hub import HfApi, create_repo

# Define your Hugging Face repo ID
hf_repo_id = "Farmerobot/deepseek-r1-among-them"

# Directory where your merged model is saved
local_model_dir = merged_model_dir

print(f"\nUploading merged model from {local_model_dir} to {hf_repo_id}...")

# Create the repository on the Hub (if it doesn't exist)
try:
    create_repo(hf_repo_id, private=False, exist_ok=True)
    print(f"Repository {hf_repo_id} ensured on Hugging Face Hub.")
except Exception as e:
    print(f"Could not create or ensure repository {hf_repo_id}. Error: {e}")
    # Handle error appropriately, e.g., raise SystemExit(1)

# Upload the folder contents
api = HfApi()
api.upload_folder(
    folder_path=local_model_dir,
    repo_id=hf_repo_id,
    repo_type="model",
    commit_message="Upload merged Unsloth fine-tuned model via script",
)

print(f"Successfully uploaded model and tokenizer files to {hf_repo_id}")
