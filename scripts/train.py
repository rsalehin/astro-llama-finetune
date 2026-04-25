import os
import sys

def main():
    print("--- 🚀 Astro-LLaMA Training Script ---")
    
    # Safety catch for local Windows execution
    if os.name == "nt":
        print("⚠️  Windows environment detected.")
        print("⏸️  Halting execution locally. Unsloth requires Linux/Colab and an NVIDIA GPU.")
        print("   File successfully generated! Please run this on Google Colab.")
        return

    try:
        from datasets import load_dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from unsloth import FastLanguageModel, is_bfloat16_supported
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Make sure you are in your Colab environment with Unsloth installed.")
        return

    print("✅ Dependencies loaded. Initializing model...")

    # 1. Load Model with 4-bit quantization
    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    # 2. Configure LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # 3. Load Dataset and apply Llama 3.1 Chat Template
    dataset = load_dataset("json", data_files="data/train_sample.jsonl", split="train")
    
    def formatting_prompts_func(examples):
        texts = []
        for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
            messages = [
                {"role": "system", "content": inst},
                {"role": "user", "content": inp},
                {"role": "assistant", "content": out}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return { "text" : texts }

    dataset = dataset.map(formatting_prompts_func, batched=True)

    # 4. Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60, # Keeping it small for the sample run
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
        ),
    )
    
    print("🔥 Starting training loop...")
    trainer.train()
    
    # 5. Save the fine-tuned adapters
    model.save_pretrained("astro-llama-lora")
    tokenizer.save_pretrained("astro-llama-lora")
    print("💾 LoRA adapters saved successfully to 'astro-llama-lora'")

if __name__ == "__main__":
    main()
