# **LLM Security Benchmarks**

**WORK IN PROGRESS**

This repository is dedicated to benchmarking lightweight, open-source Large Language Models (LLMs) for their effectiveness in providing security guidance. Our work builds upon the [SECURE Benchmark](https://arxiv.org/pdf/2405.20441) to evaluate selected models across predefined cybersecurity tasks using external configuration files for flexibility and scalability.

---

## **Scope**

**Evaluate the following LLMs against the SECURE benchmark dataset.**

- [**FastChat-T5**](https://huggingface.co/lmsys/fastchat-t5): Lightweight T5 variant for seq2seq tasks.
- [**DLite**](https://huggingface.co/aisquared/dlite-v2-1_5b): Lightweight GPT-based model for causal tasks.
- [**Gemma**](https://huggingface.co/gemma-ai): Lightweight model for cybersecurity reasoning.
- [**ZySec-AI/SecurityLLM**](https://huggingface.co/ZySec-AI/SecurityLLM): Specialized LLM for security-specific tasks.
- [**LLaMA 2**](https://huggingface.co/meta-llama/Llama-2-7b-hf): Lightweight model for reasoning and causal tasks.
- [**LLaMA 3.2**](https://huggingface.co/meta-llama/Llama-3.2-3B): Advanced model for causal and seq2seq tasks.


---

## **Mapping of SECURE Datasets to Research Categories**

| **Research Category**        | **SECURE Task**                   | **Dataset**               | **Objective**                                        |
|-------------------------------|------------------------------------|---------------------------|-----------------------------------------------------|
| **Information Extraction**    | MAET (MITRE ATT&CK Extraction)    | `SECURE - MAET.tsv`       | Extract tactics, techniques, and procedures.       |
|                               | CWET (Common Weakness Extraction) | `SECURE - CWET.tsv`       | Extract weaknesses from the CWE database.          |
| **Knowledge Understanding**   | KCV (Knowledge Test)              | `SECURE - KCV.tsv`        | Assess understanding of known vulnerabilities.     |
|                               | VOOD (Vulnerability Out-of-Distribution) | `SECURE - VOOD.tsv` | Test knowledge on new vulnerabilities.             |
| **Reasoning and Problem-Solving** | RERT (Risk Evaluation Reasoning)| `SECURE - RERT.tsv`       | Evaluate reasoning about cybersecurity risks.       |
|                               | CPST (CVSS Problem-Solving)       | `SECURE - CPST.tsv`       | Solve CVSS-related problems.                       |

---

## **Approach**

Our methodology involves:

1. **Dataset Utilization**:
   - Leveraging the SECURE dataset, with tasks grouped under three research categories:
     - **Information Extraction**
     - **Knowledge Understanding**
     - **Reasoning and Problem-Solving**
2. **Model Evaluation**:
   - Running all models on each dataset using external configuration files to define model settings (`model_config.yaml`) and task datasets (`task_config.yaml`).
3. **Performance Analysis**:
   - Evaluating performance metrics such as accuracy, precision, recall, and F1-score for each model and task. (TODO)

---

## **Repository Structure**

```
.
├── config/                     # Configuration files for models and tasks
│   ├── model_config.yaml       # Model definitions (e.g., tokenizer, model, type)
│   └── task_config.yaml        # Task-specific datasets and column mappings
├── datasets/SECURE/            # SECURE benchmark datasets (e.g., MAET.tsv, CWET.tsv)
│   └── Dataset/
├── results/                    # Generated results for model evaluations
├── scripts/                    # Evaluation scripts for each research category
│   ├── test_information_extraction.py
│   ├── test_knowledge_understanding.py
│   └── test_reasoning_and_problem_solving.py
└── README.md                   # Project documentation
```

---

## **Getting Started**

### **1. Clone the Repository**
```bash
git clone git@github.com:davisconsultingservices/llm_security_benchmarks.git
cd llm_security_benchmarks
```

### **2. Initialize Submodules**
If datasets are managed as a submodule, initialize and update them:
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

### **4. Configure Models and Tasks**
- Define models in [`config/model_config.yaml`](config/model_config.yml):
  ```yaml
  fastchat_t5:
    tokenizer: t5-small
    model: t5-small
    type: seq2seq
  ```
- Map datasets and columns in [`config/task_config.yaml`](config/task_config.yaml):
  ```yaml
  information_extraction:
    maet:
      dataset_path: datasets/SECURE/Dataset/SECURE - MAET.tsv
      input_column: Prompt       
      expected_column: Correct Answer 
  ```

### **5. Run Evaluations**
Run the evaluation scripts for each research category:
```bash
python scripts/test_information_extraction.py
python scripts/test_knowledge_understanding.py
python scripts/test_reasoning_and_problem_solving.py
```

### **6. View Results**
Results are saved in the `results/` directory as task-specific CSV files:
```bash
results/
├── maet_fastchat_t5.csv
├── maet_llama3.csv
├── ...
```

---

## **References**

- **SECURE Benchmark Paper**: [https://arxiv.org/pdf/2405.20441](https://arxiv.org/pdf/2405.20441)
- **SECURE Dataset Repository**: [https://github.com/aiforsec/SECURE](https://github.com/aiforsec/SECURE)

For more details, refer to the [SECURE Benchmark Paper](https://arxiv.org/pdf/2405.20441).

---

## **License**

This project is licensed under the [Apache-2.0 License](LICENSE). 
