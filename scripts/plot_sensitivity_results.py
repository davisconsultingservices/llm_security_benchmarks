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



# Combine data across all models for sensitivity analysis
sensitivity_data = []
for model, data_frames in model_task_data.items():
    combined_data = pd.concat(data_frames, ignore_index=True)
    combined_data['Model'] = model
    sensitivity_data.append(combined_data)

sensitivity_df = pd.concat(sensitivity_data, ignore_index=True)

# Plot box plot for sensitivity analysis
plt.figure(figsize=(12, 8))
sns.boxplot(x='Dataset', y='Correct', data=sensitivity_df)
plt.title('Dataset Sensitivity Analysis')
plt.ylabel('Correct (1 = True, 0 = False)')
plt.xlabel('Dataset')
plt.ylim(-0.1, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
sensitivity_path = os.path.join(output_dir, 'dataset_sensitivity_analysis.png')
plt.savefig(sensitivity_path)
plt.close()

print(f"Dataset sensitivity analysis plot saved: {sensitivity_path}")
