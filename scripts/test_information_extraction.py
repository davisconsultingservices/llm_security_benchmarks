import yaml
from utils import evaluate_model

with open("config/model_config.yaml", "r") as f:
    models = yaml.safe_load(f)

with open("config/task_config.yaml", "r") as f:
    tasks = yaml.safe_load(f)

datasets = tasks["information_extraction"]

for task, dataset_path in datasets.items():
    for model_name, config in models.items():
        evaluate_model(task, datasets[task]["dataset_path"], model_name, config)
