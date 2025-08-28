import boto3
import json

def call_bedrock(model_id, prompt, max_tokens=20, temperature=0.5):
    client = boto3.client("bedrock-runtime")
    
    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "prompt": prompt,
            "max_gen_len": max_tokens,
            "temperature": temperature
        }),
    )
    
    result = json.loads(response["body"].read().decode("utf-8"))
    
    # Debug: print the full response to see structure
    print("DEBUG - Full response:", json.dumps(result, indent=2, ensure_ascii=False))
    
    # Try different possible response structures
    if "outputs" in result:
        return result["outputs"][0]["text"]
    elif "generation" in result:
        return result["generation"]
    elif "text" in result:
        return result["text"]
    else:
        return str(result)

def generate_story(model_id, seed_text, iterations=10):
    current_text = seed_text
    print(f"Starting story with: {seed_text}")
    print("-" * 50)
    
    for i in range(iterations):
        print(f"Iteration {i+1}:")
        
        # Generate next chunk
        new_text = call_bedrock(model_id, current_text)
        
        # Add to story
        current_text += (new_text + "\n")
        
        # Print the new chunk
        print(new_text, end="", flush=True)
        
        # Optional: pause between iterations
        input("\nPress Enter for next chunk...")
    
    # Save complete story
    with open("generated_story.txt", "w", encoding="utf-8") as f:
        f.write(current_text)
    
    print(f"\n\nComplete story saved to generated_story.txt")

def main():
    model_id = "arn:aws:bedrock:us-east-1:318766877259:imported-model/590203qh0xye"
    
    # Get seed from user
    seed_text = input("Enter a few words to start the story: ")
    iterations = int(input("How many iterations? (default 10): ") or "10")
    
    generate_story(model_id, seed_text, iterations)

if __name__ == "__main__":
    main()
