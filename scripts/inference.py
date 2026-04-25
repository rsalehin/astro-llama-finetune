import os
import json
from pathlib import Path

def main():
    print("--- 🧠 Astro-LLaMA Inference Generation ---")
    
    if os.name == "nt":
        print("⚠️  Windows environment detected. Halting execution locally.")
        print("   Please run this on Google Colab where the trained model lives.")
        return

    try:
        from unsloth import FastLanguageModel
        import torch
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return

    print("✅ Loading fine-tuned model and base model simultaneously...")
    max_seq_length = 2048
    
    # This automatically loads the base Llama 3.1 model WITH your LoRA adapters applied
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="astro-llama-lora",
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    # Enable 2x faster inference
    FastLanguageModel.for_inference(model)

    # Two unseen test examples simulating AstroBench data
    test_cases = [
        {
            "id": 1,
            "title": "Observational Signatures of Supermassive Black Hole Mergers",
            "intro": "The detection of low-frequency gravitational waves has opened a new window into the universe. In this paper, we explore the electromagnetic counterparts associated with the final parsec problem in binary supermassive black hole systems."
        },
        {
            "id": 2,
            "title": "Atmospheric Characterization of Hot Jupiters via Transit Spectroscopy",
            "intro": "Exoplanet atmospheres provide crucial clues about planetary formation and migration. We present new JWST NIRSpec observations of the hot Jupiter WASP-39b, focusing on the carbon-to-oxygen ratio."
        }
    ]

    instruction = "You are an expert astronomer. Complete the abstract/conclusion of the following paper based on its introduction."
    results = []

    for idx, case in enumerate(test_cases):
        print(f"\n🧪 Processing Test Case {idx + 1}...")
        
        prompt_text = f"Title: {case['title']}\n\nIntroduction: {case['intro']}"
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt_text}
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")

        # 1. Generate Fine-Tuned Output
        print("   Generating Fine-Tuned response...")
        ft_outputs = model.generate(input_ids=inputs, max_new_tokens=150, use_cache=True, pad_token_id=tokenizer.eos_token_id)
        # Extract only the newly generated tokens
        ft_text = tokenizer.decode(ft_outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()

        # 2. Generate Base Model Output (By disabling the LoRA adapters)
        print("   Generating Base Model response...")
        with model.disable_adapter():
            base_outputs = model.generate(input_ids=inputs, max_new_tokens=150, use_cache=True, pad_token_id=tokenizer.eos_token_id)
            base_text = tokenizer.decode(base_outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()

        results.append({
            "instruction": instruction,
            "input": prompt_text,
            "output_base": base_text,
            "output_ft": ft_text
        })

    # Save the results for the Gemini Judge
    output_dir = Path("eval")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "inference_results.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n✅ Inference complete! Results saved to {output_file}")

if __name__ == "__main__":
    main()
