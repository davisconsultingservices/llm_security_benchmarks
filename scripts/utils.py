import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import logging

def load_model_and_tokenizer(config):
    """
    Load the model and tokenizer with GPU support if available.
    """
    # Configure logging
    logging.info("Initializing model and tokenizer...")

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logging.warning("GPU not available, falling back to CPU.")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"], token=True)

    # Load model
    if config["type"] == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(config["model"], token=True).to(device)
    else:  # Causal model
        model = AutoModelForCausalLM.from_pretrained(config["model"], token=True).to(device)

    return tokenizer, model, device


def evaluate_model(task, dataset_path, model_name, config):
    """
    Evaluate the model on a specific task using the provided dataset.
    """
    logging.info(f"Evaluating {model_name} on {task}...")

    # Load tokenizer, model, and device
    tokenizer, model, device = load_model_and_tokenizer(config)

    # Load dataset
    data = pd.read_csv(dataset_path, sep='\t')

    # Ensure all values in the "Prompt" column are strings
    data["Prompt"] = data["Prompt"].astype(str)

    # Calculate maximum expected output length
    max_expected_length = data["Correct Answer"].apply(lambda x: len(tokenizer(x)["input_ids"])).max()

    # Add a small padding of 10 tokens
    max_new_tokens = max_expected_length + 10

    # Ensure max_new_tokens does not exceed model's capacity
    model_max_tokens = model.config.max_position_embeddings
    if max_new_tokens > model_max_tokens:
        logging.warning(f"Calculated max_new_tokens ({max_new_tokens}) exceeds model's capacity ({model_max_tokens}). Using model max.")
        max_new_tokens = model_max_tokens

    logging.info(f"Final max_new_tokens: {max_new_tokens}")

    results = []

    for idx, row in data.iterrows():
        try:
            # Construct input with fixed structure
            input_text = f"{row['Prompt']}"
            expected_output = row["Correct Answer"]

            # Tokenize input and generate output
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
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
        except Exception as e:
            logging.error(f"Error processing row {idx}: {e}")
            continue

    # Save results
    os.makedirs("results", exist_ok=True)
    results_path = f"results/{task}_{model_name}.csv"
    pd.DataFrame(results).to_csv(results_path, index=False)
    logging.info(f"Results for {model_name} on {task} saved to {results_path}")
