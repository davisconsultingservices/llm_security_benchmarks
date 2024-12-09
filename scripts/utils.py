import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(config):
    """
    Load the model and tokenizer with GPU support if available.
    """
    logger.info("Initializing model and tokenizer...")

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("GPU not available, falling back to CPU.")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"], use_fast=True)

    # Load model
    if config["type"] == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(config["model"]).to(device)
    elif config["type"] == "causal":
        model = AutoModelForCausalLM.from_pretrained(config["model"]).to(device)
    else:
        raise ValueError(f"Unsupported model type: {config['type']}")

    logger.info("Model and tokenizer loaded successfully.")
    return model, tokenizer, device


def load_dataset(file_path):
    """
    Load dataset from a TSV file.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        data = pd.read_csv(file_path, sep="\t", encoding="utf-8")
        logger.info(f"Dataset loaded successfully from {file_path}. Shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Failed to load dataset from {file_path}: {e}")
        raise


def preprocess_column(data, column_name):
    """
    Preprocess a column in the dataset to ensure all values are strings.
    """
    try:
        if column_name not in data.columns:
            raise ValueError(f"Column '{column_name}' not found in dataset.")
        
        data[column_name] = data[column_name].fillna("").astype(str)
        logger.info(f"Preprocessed column '{column_name}'.")
        return data
    except Exception as e:
        logger.error(f"Error preprocessing column '{column_name}': {e}")
        raise


def evaluate_model(task, dataset_path, model_name, config):
    """
    Evaluate the model on the specified task and dataset.
    """
    logger.info(f"Evaluating model '{model_name}' on task '{task}'...")

    # Load dataset
    try:
        data = load_dataset(dataset_path)
    except Exception as e:
        logger.error(f"Dataset loading failed: {e}")
        return

    # Preprocess dataset
    try:
        data = preprocess_column(data, "Correct Answer")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return

    # Load model and tokenizer
    try:
        model, tokenizer, device = load_model_and_tokenizer(config)
    except Exception as e:
        logger.error(f"Model or tokenizer initialization failed: {e}")
        return

    # Tokenization and evaluation
    try:
        max_expected_length = data["Correct Answer"].apply(
            lambda x: len(tokenizer.encode(x, add_special_tokens=True))
        ).max()
        logger.info(f"Max expected token length: {max_expected_length}")
    except Exception as e:
        logger.error(f"Error during tokenization: {e}")
        return

    # Placeholder for additional evaluation logic
    logger.info(f"Model '{model_name}' evaluated successfully on task '{task}'.")


