import json
import sys
from datasets import load_dataset
from pathlib import Path

def format_astro_data(example):
    """
    Dynamically finds the best text columns to form the instruction format.
    """
    # UniverseTBD datasets often use 'text', 'abstract', 'summary', or 'content'
    text_content = example.get("text", example.get("abstract", example.get("content", "")))
    title = example.get("title", "Unknown Title")
    
    # We will use the first half of the text as "input" and the second half as "output"
    # to simulate predicting the conclusion/abstract from the introduction
    words = text_content.split()
    midpoint = len(words) // 2
    
    intro = " ".join(words[:midpoint])
    conclusion = " ".join(words[midpoint:])
    
    return {
        "instruction": "You are an expert astronomer. Complete the abstract/conclusion of the following paper based on its introduction.",
        "input": f"Title: {title}\n\nIntroduction: {intro}",
        "output": conclusion
    }

def main():
    print("--- 🌌 Step 2: Real Data Preparation ---")
    
    # Using the real, verified UniverseTBD dataset
    dataset_id = "UniverseTBD/arxiv-abstracts-large" 
    
    try:
        print(f"Downloading sample from {dataset_id}...")
        # streaming=True grabs a slice without downloading the whole massive dataset
        dataset = load_dataset(dataset_id, split="train", streaming=True)
        
        sample_size = 500
        sample_data = list(dataset.take(sample_size))
        
        print(f"Formatting {len(sample_data)} records...")
        
        formatted_data = [format_astro_data(row) for row in sample_data]
        
        output_dir = Path("data")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "train_sample.jsonl"
        
        with open(output_file, "w", encoding="utf-8") as f:
            for item in formatted_data:
                f.write(json.dumps(item) + "\n")
                
        print(f"✅ Real data saved to {output_file}")
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
