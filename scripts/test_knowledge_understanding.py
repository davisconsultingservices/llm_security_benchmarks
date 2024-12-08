import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import yaml
import os

# Load configurations
with open("config/model_config.yaml", "r") as f:
    models = yaml.safe_load(f)

with open("config/task_config.yaml", "r") as f:
    tasks = yaml.safe_load(f)

datasets = tasks["knowledge_understanding"]

def evaluate_model(task, dataset_path, model_name, config):
    print(f"Evaluating {model_name} on {task}...")
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
    model = (
        AutoModelForSeq2SeqLM.from_pretrained(config["model"])
        if config["type"] == "seq2seq"
        else AutoModelForCausalLM.from_pretrained(config["model"])
    )

    # Load dataset
    data = pd.read_csv(dataset_path, sep='\t')
    results = []

    for _, row in data.iterrows():
        input_text = row[tasks["information_extraction"][task]["input_column"]]
        expected_output = row[tasks["information_extraction"][task]["expected_column"]]

        # Tokenize input and generate output
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
        outputs = model.generate(**inputs)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Record the result
        results.append({
            "Task": task,
            "Input": input_text,
            "Expected Output": expected_output,
            "Model Output": output_text,
            "Correct": output_text.strip() == expected_output.strip(),
        })

    # Save results
    os.makedirs("results", exist_ok=True)
    results_path = f"results/{task}_{model_name}.csv"
    pd.DataFrame(results).to_csv(results_path, index=False)
    print(f"Results for {model_name} on {task} saved to {results_path}")

# Run evaluations for each task and model
for task, dataset_path in datasets.items():
    for model_name, config in models.items():
        evaluate_model(task, datasets[task]["dataset_path"], model_name, config)
