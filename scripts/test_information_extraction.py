import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers.utils import is_torch_available, is_tf_available, is_flax_available
import yaml
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load configurations
with open("config/model_config.yaml", "r") as f:
    models = yaml.safe_load(f)

with open("config/task_config.yaml", "r") as f:
    tasks = yaml.safe_load(f)

datasets = tasks["information_extraction"]

def load_model_and_tokenizer(config):
    # Check for available backends
    if not any([is_torch_available(), is_tf_available(), is_flax_available()]):
        raise RuntimeError(
            "No backend (PyTorch, TensorFlow, or Flax) is available. "
            "Please install at least one framework. For example, to install PyTorch for CPU:\n"
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"], token=True)

    # Load model based on type and available backend
    if config["type"] == "seq2seq":
        if is_torch_available():
            model = AutoModelForSeq2SeqLM.from_pretrained(config["model"], token=True)
        else:
            raise RuntimeError("PyTorch is required for seq2seq models.")
    else:  # Causal model
        if is_torch_available():
            model = AutoModelForCausalLM.from_pretrained(config["model"], token=True)
        else:
            raise RuntimeError("PyTorch is required for causal models.")

    return tokenizer, model


def evaluate_model(task, dataset_path, model_name, config):
    logging.info(f"Evaluating {model_name} on {task}...")
    tokenizer, model = load_model_and_tokenizer(config)

    # Load dataset
    data = pd.read_csv(dataset_path, sep='\t')
    results = []

    for _, row in data.iterrows():
        try:
            # Construct input with fixed structure
            input_text = (
                f"{row['Prompt']}\n"
                f"A) {row['Option A']}\n"
                f"B) {row['Option B']}\n"
                f"C) {row['Option C']}\n"
                f"D) {row['Option D']}"
            )
            expected_output = row["Correct Answer"]

            # Tokenize input and generate output
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
            outputs = model.generate(**inputs)
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Record the result
            results.append({
                "Task": task,
                "Prompt": input_text,
                "Expected Output": expected_output,
                "Model Output": output_text.strip(),
                "Correct": str(output_text).strip() == str(expected_output).strip(),
            })
        except KeyError as e:
            logging.error(f"Missing key in dataset: {e}")
            continue

    # Save results
    os.makedirs("results", exist_ok=True)
    results_path = f"results/{task}_{model_name}.csv"
    pd.DataFrame(results).to_csv(results_path, index=False)
    logging.info(f"Results for {model_name} on {task} saved to {results_path}")


# Run evaluations for each task and model
for task, dataset_path in datasets.items():
    for model_name, config in models.items():
        evaluate_model(task, datasets[task]["dataset_path"], model_name, config)
