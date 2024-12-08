import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import yaml
import os

# Load configurations
with open("config/model_config.yaml", "r") as f:
    model_configs = yaml.safe_load(f)

with open("config/task_config.yaml", "r") as f:
    task_configs = yaml.safe_load(f)

def evaluate_model(task, model_name, config):
    print(f"Evaluating {model_name} on {task}...")
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model"]) if config["type"] == "seq2seq" else AutoModelForCausalLM.from_pretrained(config["model"])

    # Load dataset
    dataset_path = task_configs[task]["dataset_path"]
    data = pd.read_csv(dataset_path, sep='\t')

    # Evaluate
    results = []
    for _, row in data.iterrows():
        input_text = row[task_configs[task]["input_column"]]
        expected_output = row[task_configs[task]["expected_column"]]

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
        outputs = model.generate(**inputs)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        results.append({
            "Input": input_text,
            "Model_Output": output_text,
            "Expected_Output": expected_output,
            "Correct": output_text.strip() == expected_output.strip()
        })

    # Save results
    os.makedirs("results", exist_ok=True)
    results_path = f"results/{task}_{model_name}.csv"
    pd.DataFrame(results).to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

# Run evaluations
for task in task_configs:
    for model_name, model_config in model_configs.items():
        evaluate_model(task, model_name, model_config)
