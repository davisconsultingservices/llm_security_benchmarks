import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the directory containing the CSV files
results_dir = 'results'

# Define the mapping of datasets to their respective task categories
task_mapping = {
    'cpst': 'Understanding',
    'cwet': 'Extraction',
    'kcv': 'Understanding',
    'maet': 'Extraction',
    'rert': 'Reasoning'
}

# Ensure output directory exists
output_dir = os.path.join(results_dir, "plots")
os.makedirs(output_dir, exist_ok=True)

# Dictionary to store data for each model
model_task_data = {}

# Process each CSV file
for file_name in os.listdir(results_dir):
    if file_name.endswith('.csv'):
        # Extract dataset and model identifiers from the file name
        parts = file_name.split('_')
        dataset_id = parts[0]
        model_name = parts[1]
        
        # Determine the task category
        task_category = task_mapping.get(dataset_id)
        if task_category:
            # Initialize the model's task data dictionary if not present
            if model_name not in model_task_data:
                model_task_data[model_name] = []
            
            # Load the CSV file
            file_path = os.path.join(results_dir, file_name)
            df = pd.read_csv(file_path)
            
            # Ensure Correct column is boolean
            df['Correct'] = df['Correct'].astype(bool)
            
            # Add task category and dataset info for plotting
            df['TaskCategory'] = task_category
            df['Dataset'] = dataset_id.upper()
            df['Model'] = model_name
            
            # Append the data for the model
            model_task_data[model_name].append(df)


# Generate a heatmap for each model
for model, data_frames in model_task_data.items():
    # Combine all data frames for this model
    combined_data = pd.concat(data_frames, ignore_index=True)
    
    # Calculate mean accuracy by dataset and task category
    heatmap_data = combined_data.groupby(['TaskCategory', 'Dataset'])['Correct'].mean().unstack()
    
    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt=".2f", cbar_kws={'label': 'Accuracy'})
    plt.title(f'Heatmap of Accuracy by Dataset and Task Category for {model}')
    plt.ylabel('Task Category')
    plt.xlabel('Dataset')
    plt.tight_layout()
    
    # Save the heatmap
    heatmap_path = os.path.join(output_dir, f'{model}_accuracy_heatmap.png')
    plt.savefig(heatmap_path)
    plt.close()
    
    print(f"Heatmap saved for {model}: {heatmap_path}")
