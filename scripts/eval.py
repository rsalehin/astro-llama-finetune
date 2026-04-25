import os
import json
from pathlib import Path

# Explicit import from the google namespace
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("❌ SDK not found. Trying alternative import...")
    import google.genai as genai
    from google.genai import types

def evaluate_models(client, instruction, input_text, output_base, output_ft):
    prompt = f"""You are an expert astronomer and machine learning judge evaluating two AI models.
    The task was to complete a scientific abstract based on the paper's title and introduction.

    Instruction: {instruction}
    Input: {input_text}

    Model A (Base) Output: 
    {output_base}

    Model B (Fine-Tuned) Output: 
    {output_ft}

    Evaluate which model produces a response that better matches the style, tone, and formatting of a real astronomy paper. 
    Crucial Grading Rubric: Real dataset completions typically do NOT include chatty pleasantries or markdown headers like "**Abstract:**". They just seamlessly continue the scientific text.
    
    Respond strictly with a JSON object containing two keys:
    "winner": either "Model A", "Model B", or "Tie"
    "justification": a brief 1-2 sentence explanation.
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )
    return json.loads(response.text)

def main():
    print("--- ⚖️ AstroBench Evaluation: Running Gemini Judge ---")
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  GEMINI_API_KEY environment variable not found.")
        return

    client = genai.Client(api_key=api_key)
    
    results_file = Path("eval/inference_results.json")
    if not results_file.exists():
        print(f"❌ Cannot find {results_file}.")
        return
        
    with open(results_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    scorecard = {"Model A": 0, "Model B": 0, "Tie": 0}
    eval_log = []

    for idx, case in enumerate(test_cases):
        print(f"   Judging Case {idx+1}...")
        try:
            judgement = evaluate_models(
                client, 
                case["instruction"], 
                case["input"], 
                case["output_base"], 
                case["output_ft"]
            )
            
            winner = judgement.get("winner", "Tie")
            if "Model A" in winner:
                scorecard["Model A"] += 1
            elif "Model B" in winner:
                scorecard["Model B"] += 1
            else:
                scorecard["Tie"] += 1
            
            eval_log.append({
                "case": idx + 1,
                "winner": winner,
                "justification": judgement.get("justification", "No justification provided.")
            })
        except Exception as e:
            print(f"   ❌ Error on Case {idx+1}: {e}")

    # Save Scorecard
    scorecard_file = Path("eval/scorecard.md")
    with open(scorecard_file, "w", encoding="utf-8") as f:
        f.write("# Astro-LLaMA Evaluation Scorecard\n\n")
        f.write("## Final Score\n")
        f.write(f"- **Base Model Wins:** {scorecard['Model A']}\n")
        f.write(f"- **Fine-Tuned Model Wins:** {scorecard['Model B']}\n")
        f.write(f"- **Ties:** {scorecard['Tie']}\n\n")
        f.write("## Detailed Judgments\n")
        for log in eval_log:
            f.write(f"### Case {log['case']}\n")
            f.write(f"- **Winner:** {log['winner']}\n")
            f.write(f"- **Justification:** {log['justification']}\n\n")

    print(f"\n✅ Evaluation complete! Scorecard saved to {scorecard_file}")

if __name__ == "__main__":
    main()
