# **LLM Security Guidance Benchmarks**

This repository is dedicated to benchmarking lightweight, open-source Large Language Models (LLMs) for their effectiveness in providing security guidance. Our work builds upon the [SECURE Benchmark](https://arxiv.org/pdf/2405.20441) to evaluate selected models across predefined cybersecurity tasks using external configuration files for flexibility and scalability.


See the [RESULTS](results/README.md)

---

## **Scope**

**Evaluate the following LLMs against the SECURE benchmark dataset:**

- [**DLite**](https://huggingface.co/aisquared/dlite-v2-1_5b): Lightweight GPT-based model for causal tasks.
- [**FastChat-T5**](https://huggingface.co/lmsys/fastchat-t5): Lightweight T5 variant for sequence-to-sequence tasks.
- [**Gemma**](https://huggingface.co/gemma-ai): Lightweight model for cybersecurity reasoning.
- [**LLaMA 2**](https://huggingface.co/meta-llama/Llama-2-7b-hf): Lightweight model for reasoning and causal tasks.
- [**LLaMA 3.2**](https://huggingface.co/meta-llama/Llama-3.2-3B): Advanced model for causal and sequence-to-sequence tasks.
- [**ZySec-AI/SecurityLLM**](https://huggingface.co/ZySec-AI/SecurityLLM): Specialized LLM for security-specific tasks.


---


## **Tests**

1. **`test_information_extraction.py`**  
   - **Description**: Tests the ability of models to extract information such as MITRE ATT&CK tactics and CWE weaknesses.
   - **Dataset**: SECURE - MAET.tsv, CWET.tsv

2. **`test_knowledge_understanding.py`**  
   - **Description**: Evaluates models on understanding cybersecurity concepts and known vulnerabilities.
   - **Dataset**: SECURE - KCV.tsv

3. **`test_reasoning_and_problem_solving.py`**  
   - **Description**: Assesses reasoning about cybersecurity risks and solving CVSS-related problems.
   - **Dataset**: SECURE - RERT.tsv, CPST.tsv

---

## **Plotting Functions**

The repository includes scripts to visualize the results. Each script generates plots that can be accessed directly below:

1. **`plot_density_results.py`**  
   - **Description**: Plots the density of correct vs. incorrect predictions for each model.  

2. **`plot_heatmap_results.py`**  
   - **Description**: Creates heatmaps to visualize model accuracy across datasets and tasks.  

3. **`plot_violin_results.py`**  
   - **Description**: Generates violin plots to illustrate performance distribution across tasks and datasets.  

4. **`plot_performance_results.py`**  
   - **Description**: Compares task performance across models using bar plots.  

5. **`plot_sensitivity_results.py`**  
   - **Description**: Visualizes sensitivity analysis of models for datasets/tasks.  

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
Run the plotting scripts to visualize the results:
```bash
python scripts/plot_density_results.py
python scripts/plot_heatmap_results.py
python scripts/plot_violin_results.py
python scripts/plot_performance_results.py
python scripts/plot_sensitivity_results.py
```

---

## **References**

- **SECURE Benchmark Paper**: [https://arxiv.org/pdf/2405.20441](https://arxiv.org/pdf/2405.20441)
- **SECURE Dataset Repository**: [https://github.com/aiforsec/SECURE](https://github.com/aiforsec/SECURE)

For more details, refer to the [SECURE Benchmark Paper](https://arxiv.org/pdf/2405.20441).

---

## **License**

This project is licensed under the [Apache-2.0 License](LICENSE).
