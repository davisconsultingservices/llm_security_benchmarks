import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers.utils import is_torch_available, is_tf_available, is_flax_available
import logging

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

    # Load the tokenizer and model
    tokenizer, model = load_model_and_tokenizer(config)

    # Load the dataset
    data = pd.read_csv(dataset_path, sep='\t')

    # Determine the maximum prompt length in tokens
    max_prompt_length = data["Prompt"].apply(lambda x: len(tokenizer(x)["input_ids"])).max()

    # Use max_new_tokens from config if provided; otherwise, use max_prompt_length
    max_new_tokens = config.get("max_new_tokens", max_prompt_length)
    logging.info(f"Using max_new_tokens: {max_new_tokens}")

    results = []

    for _, row in data.iterrows():
        try:
            # Construct input with fixed structure
            input_text = f"{row['Prompt']}"
            expected_output = row["Correct Answer"]

            # Tokenize input and generate output
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

            # Decode the output
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            output_text = output_text.replace(input_text, "").strip()
            if not output_text:
                output_text = "X"

            # Record the result
            results.append({
                "Task": task,
                "Prompt": input_text,
                "Expected Output": expected_output,
                "Model Output": output_text,
                "Correct": output_text == str(expected_output).strip(),
            })
        except KeyError as e:
            logging.error(f"Missing key in dataset: {e}")
            continue

    # Save results
    os.makedirs("results", exist_ok=True)
    results_path = f"results/{task}_{model_name}.csv"
    pd.DataFrame(results).to_csv(results_path, index=False)
    logging.info(f"Results for {model_name} on {task} saved to {results_path}")
