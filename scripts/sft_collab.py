# Setup SFT trainer (only train on the assistant's outputs)
print("Setting up trainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Ensure eval_dataset is passed
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        output_dir=lora_weights_output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,        # Can increase if memory allows
        gradient_accumulation_steps=8,       # Increased effective batch size (1*8=8)
        gradient_checkpointing=True,         # Enable gradient checkpointing
        max_steps=1000,                      # Rely on early stopping
        learning_rate=2e-5,                  # Keep the lowered LR
        warmup_steps=50,                     # Reduced slightly (~5% of max_steps)
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,                    # Keep logging frequent
        
        # --- Parameters for Early Stopping & Regularization ---
        eval_strategy="steps",       # Evaluate periodically
        eval_steps=50,                     # How often to evaluate
        save_strategy="steps",             # Corresponds to eval_strategy
        save_steps=50,                     # Match eval_steps
        load_best_model_at_end=True,       # Load the best checkpoint at the end
        metric_for_best_model="eval_loss", # Monitor validation loss
        greater_is_better=False,           # Lower loss is better
        save_total_limit=2,                # Keep only the best and the latest checkpoints
        # ------------------------------------------------------

        optim="adamw_8bit",
        weight_decay=0.05,                   # Increased weight decay (from 0.01)
        lr_scheduler_type="cosine",        # Changed scheduler (from linear)
        seed=42,
        report_to="tensorboard",             # Changed from "none"
    ),
    # Add the EarlyStoppingCallback
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.005)] # Stop if no improvement for 5 evals
)

# Mask loss to only train on assistant responses
# Remember to use the correct unicode character.
# Deepseek tokenizer uses a different one than the standard vertical line on a QWERTY keyboard
# ｜ - U+FF5C : FULLWIDTH VERTICAL LINE
# | - U+007C : VERTICAL LINE {vertical bar, pipe}
trainer = train_on_responses_only(
    trainer=trainer,
    tokenizer=tokenizer,
    instruction_part="<｜User｜>",
    response_part="<｜Assistant｜>",
)

# Verify masking on trainer dataset
# This should include these special tokens:
# {
#     "stop": [
#         "<｜begin▁of▁sentence｜>",
#         "<｜end▁of▁sentence｜>",
#         "<｜User｜>",
#         "<｜Assistant｜>"
#     ]
# }
# Source: https://ollama.com/library/deepseek-r1/blobs/f4d24e9138dd
print("Decoding sample 0 input_ids after masking:")
print(tokenizer.decode(trainer.train_dataset[0]["input_ids"], skip_special_tokens=False))

# This should print WITHOUT the special tokens above and instead with some padding tokens
# It should only print the output and omit the prompt.
if "labels" in trainer.train_dataset[0]:
    label_ids = [tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[0]["labels"]]
    print("Decoded labels:", tokenizer.decode(label_ids, skip_special_tokens=False))