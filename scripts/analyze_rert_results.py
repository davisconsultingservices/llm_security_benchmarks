import os
import glob
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
)

# Function to interact with ChatGPT API
def analyze_output(prompt_used, expected_output, model_output):
    prompt = f"""
    Evaluate the following:
    - Prompt Used: "{prompt_used}"
    - Expected Output: "{expected_output}"
    - Model Output: "{model_output}"
    
    Does the Model Output loosely represent the Expected Output in meaning, structure, or intent? Respond with a simple "Yes" or "No".
    """

    try:

        stream = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": prompt,
            }],
            model="gpt-4o",
            stream=True,
        )
        feedback = ''
        for chunk in stream:
            feedback += chunk.choices[0].delta.content or ""
        return feedback
    except Exception as e:
        print(f"Error while fetching feedback: {e}")
        return None

# Directory containing rert_* result files
results_dir = "results"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Get all rert_* files
result_files = glob.glob(os.path.join(results_dir, "rert_*.csv"))

# Process each file
for file_path in result_files:

    # if (file_path == "results/rert_fastchat_t5.csv" or
    #     file_path == "results/rert_dlite.csv" or
    #     file_path == "results/rert_gemma.csv" or 
    #     file_path == "results/rert_llama3.csv"):
    #     continue

    # Load the result CSV
    df = pd.read_csv(file_path)
    
    # Add a new column for GPT-4 evaluation
    gpt_results = []
    
    try:
        print(f"Processing file: {file_path}")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing rows"):
            prompt = row["Prompt"]
            expected = row["Expected Output"]
            model = row["Model Output"]
            analysis = analyze_output(prompt, expected, model)
            gpt_results.append(analysis)
            # print(gpt_results)
            print(analysis)
    
    except Exception as e:
        print(f"Error while chatgpt analyzing: {e}")

    # Add GPT results to the dataframe
    df["GPT-4 Analysis"] = gpt_results
    
    # Save the updated results
    output_file = os.path.join(output_dir, os.path.basename(file_path))
    df.to_csv(output_file+"_gpted.csv", index=False)
    print(f"Analysis saved to: {output_file}")
