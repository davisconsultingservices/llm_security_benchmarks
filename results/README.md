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

2) rert testing for the securityllm, llama3 and llama2 models were stopped short when it was obvious their output was always repeatively wrong (after some hundres of iterations)

e.g.
```
3.2.2 Path Traversal CWE-22 There are multiple ways in LAquis SCADA for an attacker to access locations outside of their own directory. CVE-2024-5041 has been assigned to this vulnerability. A CVSS v3.1 base score of 7.8 has been calculated; the CVSS vector string is (AV:L/AC:L/PR:N/UI:R/S:U/C:H/I:H/A:H). A CVSS v4 score has also been calculated for CVE-2024-5041. A base score of 8.5 has been calculated; the CVSS vector string is (CVSS4.0/AV:L/AC:L/AT:N/PR:N/UI:P/VC:H/VI:H/VA:H/WA:H/C:H/I:H/A:H). 3.2.3 Path Traversal CWE-22 There are multiple ways in LAquis SCADA for an attacker to access locations outside of their own directory. CVE-2024-5042 has been assigned to this vulnerability. A CVSS v3.1 base score of 7.8 has been calculated; the CVSS vector string is...
```