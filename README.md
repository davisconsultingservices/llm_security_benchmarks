# **LLM Security Guidance Benchmarks**

**WORK IN PROGRESS**

This repository is dedicated to benchmarking lightweight, open-source Large Language Models (LLMs) for their effectiveness in providing security guidance. Our work builds upon the [SECURE Benchmark](https://arxiv.org/pdf/2405.20441) to evaluate selected models across predefined cybersecurity tasks using external configuration files for flexibility and scalability.

---

## **Scope**

**Evaluate the following LLMs against the SECURE benchmark dataset:**

- [**FastChat-T5**](https://huggingface.co/lmsys/fastchat-t5): Lightweight T5 variant for sequence-to-sequence tasks.
- [**DLite**](https://huggingface.co/aisquared/dlite-v2-1_5b): Lightweight GPT-based model for causal tasks.
- [**ZySec-AI/SecurityLLM**](https://huggingface.co/ZySec-AI/SecurityLLM): Specialized LLM for security-specific tasks.
- [**LLaMA 2**](https://huggingface.co/meta-llama/Llama-2-7b-hf): Lightweight model for reasoning and causal tasks.
- [**LLaMA 3.2**](https://huggingface.co/meta-llama/Llama-3.2-3B): Advanced model for causal and sequence-to-sequence tasks.
- [**Gemma**](https://huggingface.co/gemma-ai): Lightweight model for cybersecurity reasoning.

---

## **Plotting Functions**

The repository includes the following plotting scripts to visualize benchmarking results. The generated plots are saved in the `results/plots/` directory:

1. **`plot_density_results.py`**  
   - **Description**: Plots the density of correct vs. incorrect predictions for each model.
   - **Outputs**: 
     - `fastchat_correct_vs_incorrect_density.png`
     - `gemma.csv_correct_vs_incorrect_density.png`
     - `securityllm.csv_correct_vs_incorrect_density.png`
   - **Directory**: [results/plots/](results/plots/)

2. **`plot_heatmap_results.py`**  
   - **Description**: Creates heatmaps to show the accuracy of models across datasets and tasks.
   - **Outputs**: 
     - `fastchat_accuracy_heatmap.png`
     - `gemma.csv_accuracy_heatmap.png`
     - `securityllm.csv_accuracy_heatmap.png`
   - **Directory**: [results/plots/](results/plots/)

3. **`plot_violin_results.py`**  
   - **Description**: Generates violin plots to illustrate the performance distribution of models across tasks and datasets.
   - **Outputs**: 
     - `fastchat_violinplot_by_task_and_dataset.png`
     - `gemma.csv_violinplot_by_task_and_dataset.png`
     - `securityllm.csv_violinplot_by_task_and_dataset.png`
   - **Directory**: [results/plots/](results/plots/)

4. **`plot_performance_results.py`**  
   - **Description**: Compares task performance across models using bar plots.
   - **Output**: 
     - `task_performance_comparison_across_models.png`
   - **Directory**: [results/plots/](results/plots/)

5. **`plot_sensitivity_results.py`**  
   - **Description**: Visualizes the sensitivity analysis of models for datasets/tasks.
   - **Output**: 
     - `dataset_sensitivity_analysis.png`
   - **Directory**: [results/plots/](results/plots/)

---

## **Getting Started**

### **1. Clone the Repository**
```bash
git clone git@github.com:davisconsultingservices/llm_security_benchmarks.git
cd llm_security_benchmarks
```

### **2. Initialize Submodules**
If datasets are managed as submodules, initialize and update them:
```bash
git submodule update --init --recursive
```

### **3. Set Up the Environment**
Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### **4. Run Evaluations**
Execute the evaluation scripts for each research category:
```bash
python scripts/test_information_extraction.py
python scripts/test_knowledge_understanding.py
python scripts/test_reasoning_and_problem_solving.py
```

### **5. Generate Plots**
After evaluations, generate visualizations using the plotting scripts:
```bash
python scripts/plot_density_results.py
python scripts/plot_heatmap_results.py
python scripts/plot_violin_results.py
python scripts/plot_performance_results.py
python scripts/plot_sensitivity_results.py
```

### **6. View Results**
Plots and visualizations are saved in the `results/plots/` directory. Example outputs include:
- `fastchat_correct_vs_incorrect_density.png`
- `fastchat_accuracy_heatmap.png`
- `task_performance_comparison_across_models.png`

---

## **References**

- **SECURE Benchmark Paper**: [https://arxiv.org/pdf/2405.20441](https://arxiv.org/pdf/2405.20441)
- **SECURE Dataset Repository**: [https://github.com/aiforsec/SECURE](https://github.com/aiforsec/SECURE)

For more details, refer to the [SECURE Benchmark Paper](https://arxiv.org/pdf/2405.20441).

---

## **License**

This project is licensed under the [Apache-2.0 License](LICENSE).
