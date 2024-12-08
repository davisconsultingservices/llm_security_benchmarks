# LLM Security Benchmarks

This repository is dedicated to benchmarking lightweight, open-source Large Language Models (LLMs) for their effectiveness in providing security guidance. Our work builds upon the [SECURE Benchmark](https://arxiv.org/pdf/2405.20441) to evaluate selected models across predefined cybersecurity tasks.

## Scope

We aim to:

1. **Replicate and extend the SECURE Benchmark** by assessing the performance of various LLMs in realistic cybersecurity scenarios.
2. **Evaluate four selected models**:
   - **FastChat-T5**
   - **LLaMA 3**
   - **DLite**
   - **Gemma**

## Approach

Our methodology involves:

1. **Dataset Utilization**: Employing the SECURE dataset, which includes tasks like threat identification, security best practices, and compliance guidance.
2. **Model Evaluation**: Running each model against the dataset to assess their ability to provide accurate and relevant security guidance.
3. **Performance Analysis**: Comparing outputs to expected results to determine each model's effectiveness.

## Repository Structure

- `scripts/`: Contains evaluation scripts.
- `config/`: Holds configuration files for models and tasks.
- `datasets/SECURE/`: Includes the SECURE dataset.
- `results/`: Stores evaluation outcomes.

## Getting Started

1. **Clone the Repository**:
   ```bash
   git clone git@github.com:davisconsultingservices/llm_security_benchmarks.git
   cd llm_security_benchmarks
   ```
2. **Initialize Submodules**:
   ```bash
   git submodule update --init --recursive
   ```
3. **Set Up Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
4. **Configure Models and Tasks**:
   - Edit `config/model_config.yaml` to specify model details.
   - Edit `config/task_config.yaml` to define task parameters.
5. **Run Evaluations**:
   ```bash
   python scripts/test_models.py
   ```

## References

- **SECURE Benchmark Paper**: [https://arxiv.org/pdf/2405.20441](https://arxiv.org/pdf/2405.20441)
- **SECURE Dataset Repository**: [https://github.com/aiforsec/SECURE](https://github.com/aiforsec/SECURE)

For more details, please refer to the [SECURE Benchmark Paper](https://arxiv.org/pdf/2405.20441).

---

*Note: This project is licensed under the [Apache-2.0 License](LICENSE).* 