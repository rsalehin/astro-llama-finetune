import os
from google import genai
from google.genai import types

def evaluate_models(client, instruction, output_base, output_ft):
    prompt = f"""You are an expert astronomer evaluating two AI models based on a prompt.
    
    Instruction: {instruction}
    
    Model A (Base) Output: 
    {output_base}
    
    Model B (Fine-Tuned) Output: 
    {output_ft}
    
    Evaluate which model produces a more scientifically accurate, hallucination-free, and relevant response. 
    Respond with a JSON object containing two keys:
    "winner": either "Model A", "Model B", or "Tie"
    "justification": a brief 1-2 sentence explanation.
    """
    
    # Using the new SDK structure
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )
    return response.text

def main():
    print("--- ⚖️ AstroBench Evaluation Script (Gemini Judge) ---")
    
    # Check for API Key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  GEMINI_API_KEY environment variable not found.")
        print("   Set it in PowerShell using: $env:GEMINI_API_KEY=\"your_api_key\"")
        print("   File successfully generated! Exiting for now.")
        return

    # The new SDK automatically picks up the GEMINI_API_KEY environment variable
    client = genai.Client()
    print("✅ Gemini API configured successfully.")
    
    # Dummy test to verify the logic works if a key is present
    print("🧪 Running a quick test evaluation...")
    try:
        result = evaluate_models(
            client=client,
            instruction="Explain the difference between a pulsar and a quasar.",
            output_base="A pulsar is a star. A quasar is a galaxy thing.",
            output_ft="A pulsar is a highly magnetized, rotating neutron star that emits beams of electromagnetic radiation. A quasar is an extremely luminous active galactic nucleus powered by a supermassive black hole."
        )
        print(f"Result: {result}")
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")

if __name__ == "__main__":
    main()
