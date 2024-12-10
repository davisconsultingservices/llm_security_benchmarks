import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def load_model_and_tokenizer(config):
    """
    Load the model and tokenizer with GPU support if available.
    """
    logging.info("Initializing model and tokenizer...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logging.warning("GPU not available, falling back to CPU.")

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"], use_fast=True)
    if config["type"] == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(config["model"]).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(config["model"]).to(device)

    return tokenizer, model, device

def evaluate_model(task, dataset_path, model_name, config):
    """
    Evaluate the model on a specific task using the provided dataset.
    Results are saved iteratively.
    """
    logging.info(f"Evaluating {model_name} on {task}...")

    tokenizer, model, device = load_model_and_tokenizer(config)
    data = pd.read_csv(dataset_path, sep="\t")

    data["Prompt"] = data["Prompt"].astype(str)
    data["Correct Answer"] = data["Correct Answer"].fillna("").astype(str)

    max_expected_length = data["Correct Answer"].apply(
        lambda x: len(tokenizer.encode(x, add_special_tokens=True))
    ).max()

    max_new_tokens = max_expected_length + 10
    model_max_tokens = getattr(model.config, 'max_position_embeddings', config['max_new_tokens'])

    if max_new_tokens > model_max_tokens:
        logging.warning(f"max_new_tokens ({max_new_tokens}) exceeds model's capacity ({model_max_tokens}). Using model max.")
        max_new_tokens = model_max_tokens

    logging.info(f"Final max_new_tokens: {max_new_tokens}")

    # Initialize results and output path
    results = []
    os.makedirs("results", exist_ok=True)
    results_path = os.path.join("results", f"{task}_{model_name}.csv")

    # # Check if partial results exist
    # if os.path.exists(results_path):
    #     logging.info(f"Partial results found. Resuming from {results_path}.")
    #     results = pd.read_csv(results_path).to_dict("records")

    # processed_indices = {r["Index"] for r in results}

    # Process each row and save iteratively
    for idx, row in data.iterrows():
        # if idx in processed_indices:
        #     continue  # Skip already processed rows

        try:
            input_text = row["Prompt"]
            expected_output = row["Correct Answer"]

            inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
            outputs = model.generate(inputs["input_ids"], max_new_tokens=max_new_tokens)

            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            output_text = output_text.replace(input_text, "").strip()
            if not output_text:
                output_text = "X"

            results.append({
                "Task": task,
                "Prompt": input_text,
                "Expected Output": expected_output,
                "Model Output": output_text,
                "Correct": output_text == expected_output.strip(),
            })

            # Save results iteratively
            pd.DataFrame(results).to_csv(results_path, index=False)
        except Exception as e:
            logging.error(f"Error processing row {idx}: {e}")
            continue

    logging.info(f"Final results for {model_name} on {task} saved to {results_path}")

# def evaluate_model(task, dataset_path, model_name, config):
#     """
#     Evaluate the model on a specific task using the provided dataset.
#     """
#     logging.info(f"Evaluating {model_name} on {task}...")

#     tokenizer, model, device = load_model_and_tokenizer(config)
#     data = pd.read_csv(dataset_path, sep="\t")

#     data["Prompt"] = data["Prompt"].astype(str)
#     data["Correct Answer"] = data["Correct Answer"].fillna("").astype(str)

#     max_expected_length = data["Correct Answer"].apply(
#         lambda x: len(tokenizer.encode(x, add_special_tokens=True))
#     ).max()

#     max_new_tokens = max_expected_length + 10
#     model_max_tokens = getattr(model.config, 'max_position_embeddings', config['max_new_tokens'])

#     if max_new_tokens > model_max_tokens:
#         logging.warning(f"max_new_tokens ({max_new_tokens}) exceeds model's capacity ({model_max_tokens}). Using model max.")
#         max_new_tokens = model_max_tokens

#     logging.info(f"Final max_new_tokens: {max_new_tokens}")

#     results = []
#     for idx, row in data.iterrows():
#         try:
#             input_text = row["Prompt"]
#             expected_output = row["Correct Answer"]

#             inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
#             outputs = model.generate(inputs["input_ids"], max_new_tokens=max_new_tokens)

#             output_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
#             output_text = output_text.replace(input_text, "").strip()
#             if not output_text:
#                 output_text = "X"

#             results.append({
#                 "Task": task,
#                 "Prompt": input_text,
#                 "Expected Output": expected_output,
#                 "Model Output": output_text,
#                 "Correct": output_text == expected_output.strip(),
#             })
#         except Exception as e:
#             logging.error(f"Error processing row {idx}: {e}")
#             continue

#     os.makedirs("results", exist_ok=True)
#     results_path = os.path.join("results", f"{task}_{model_name}.csv")
#     pd.DataFrame(results).to_csv(results_path, index=False)
#     logging.info(f"Results for {model_name} on {task} saved to {results_path}")
