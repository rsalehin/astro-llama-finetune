import json
from pathlib import Path

def main():
    print("--- 🧪 TDD Step 1: Data Preparation ---")
    
    # We generate 100 synthetic astronomy records to test the pipeline mechanics
    # This isolates our test from Hugging Face API errors or dataset naming issues
    sample_data = []
    for i in range(1, 101):
        sample_data.append({
            "instruction": "You are an expert astronomer. Write a short abstract based on the title.",
            "input": f"Title: Astrophysical Observations of Phenomenon {i}",
            "output": f"This paper presents novel observations of Phenomenon {i} using spectroscopic analysis, revealing key insights into galactic evolution."
        })
        
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "train_sample.jsonl"
    
    with open(output_file, "w", encoding="utf-8") as f:
        for item in sample_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"✅ Generated {len(sample_data)} records.")
    
    # TDD Assertion: Hard fail if the file wasn't created
    assert output_file.exists(), "❌ TDD Failure: train_sample.jsonl was not created!"
    print(f"✅ TDD Pass: File verified at {output_file}")

if __name__ == "__main__":
    main()
