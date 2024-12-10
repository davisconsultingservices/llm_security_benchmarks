## **Runtime Environment**

All tests and benchmarks were conducted on **Google Colab**, leveraging an **NVIDIA A100 Tensor Core GPU**. This high-performance setup allows for efficient evaluation of large language models and reduces the time required for processing large datasets. Key environment details:

- **GPU**: NVIDIA A100 Tensor Core (40GB memory)
- **Frameworks**: 
  - PyTorch with CUDA support
  - Transformers library for model evaluation
- **Runtime Configuration**:
  - Python 3.x
  - Batch processing with iterative result saving for long-running tasks

---


## Notes

1) rert reporting for the dlite model had no output as the model could not read the input columns.
```
Token indices sequence length is longer than the specified maximum sequence length for this model (2140 > 1024). Running this sequence through the model will result in indexing errors
```