import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers.utils import is_torch_available, is_tf_available, is_flax_available
import logging

logging.basicConfig(
    level=logging.INFO,  # Ensure it's set to INFO or DEBUG
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

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

    # Load tokenizer and model
    tokenizer, model = load_model_and_tokenizer(config)

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

    for _, row in data.iterrows():
        try:
            # Construct input with fixed structure
            input_text = f"{row['Prompt']}"
            expected_output = row["Correct Answer"]

            logging.info(f"**Prompt: {input_text}")

            # Tokenize input and generate output
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

            # Decode the output
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            output_text = output_text.replace(input_text, "").strip()
            if not output_text:
                output_text = "X"

            logging.info(f"**Response: {output_text}")

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
            logging.error(f"Error processing row: {e}")
            continue

    # Save results
    os.makedirs("results", exist_ok=True)
    results_path = f"results/{task}_{model_name}.csv"
    pd.DataFrame(results).to_csv(results_path, index=False)
    logging.info(f"Results for {model_name} on {task} saved to {results_path}")
