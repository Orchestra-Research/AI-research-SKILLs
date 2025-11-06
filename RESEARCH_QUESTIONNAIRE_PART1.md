# AI Research Skills Discovery Questionnaire - Part 1

## Model Training & Infrastructure

**Purpose:** Guide literature research to identify critical topics, libraries, and best practices for model training and infrastructure.

**Instructions for Research Team:**
- Answer each question with specific library names, paper citations, and current best practices
- Prioritize by adoption rate and production readiness
- Include version numbers and last update dates
- Note if a tool/practice is emerging vs. established

---

## 1. Model Architecture & Design

### 1.1 Foundation Models
- **Q1.1:** What are the current state-of-the-art architectures for LLMs? (e.g., Transformer variants, Mamba, RWKV)
- **Q1.2:** Which model architectures are optimized for specific tasks? (long-context, multimodal, code, math)
- **Q1.3:** What are the key papers/implementations for each architecture?
- **Q1.4:** Which frameworks are used to implement custom architectures? (e.g., Megatron-Core, NeoX, LitGPT)

### 1.2 Model Initialization & Pretraining
- **Q2.1:** What are the current best practices for model initialization?
- **Q2.2:** Which pretraining libraries/frameworks are most used? (e.g., Megatron-LM, GPT-NeoX, MosaicML Composer)
- **Q2.3:** What tokenization libraries and strategies are standard? (e.g., SentencePiece, tiktoken, custom tokenizers)
- **Q2.4:** What datasets and data processing pipelines are used for pretraining?

---

## 2. Fine-Tuning & Adaptation

### 2.1 Supervised Fine-Tuning (SFT)
- **Q3.1:** What are the standard libraries for SFT? (e.g., Axolotl, TRL, LLaMA-Factory, Unsloth)
- **Q3.2:** What are the best practices for instruction formatting and prompt engineering?
- **Q3.3:** Which dataset formats are standard? (e.g., ShareGPT, Alpaca, chat templates)
- **Q3.4:** What tools exist for data quality assessment and filtering?

### 2.2 Parameter-Efficient Fine-Tuning (PEFT)
- **Q4.1:** Which PEFT methods are production-ready? (LoRA, QLoRA, Adapters, Prefix Tuning, IA3, DoRA)
- **Q4.2:** What libraries implement PEFT? (HuggingFace PEFT, LitGPT adapters)
- **Q4.3:** What are the tradeoffs between PEFT methods? (memory, speed, quality)
- **Q4.4:** Which PEFT methods work best for different model sizes?

### 2.3 Continued Pretraining & Domain Adaptation
- **Q5.1:** What are best practices for continued pretraining on domain-specific data?
- **Q5.2:** Which tools help with domain data curation and filtering?
- **Q5.3:** How do researchers handle catastrophic forgetting during adaptation?

---

## 3. Post-Training & Alignment

### 3.1 Preference Optimization
- **Q6.1:** Which preference optimization methods are most used? (DPO, RLHF, PPO, IPO, KTO, ORPO, SimPO)
- **Q6.2:** What libraries implement these methods? (TRL, trlX, OpenRLHF)
- **Q6.3:** How do researchers generate preference datasets? (AI feedback, human feedback, synthetic data)
- **Q6.4:** What are the emerging alternatives to RLHF?

### 3.2 Reinforcement Learning for LLMs
- **Q7.1:** Which RL algorithms are used for LLM training? (PPO, GRPO, RLOO, ReMax)
- **Q7.2:** What reward modeling techniques are standard?
- **Q7.3:** Which libraries specialize in RL for LLMs? (TRL, trlX, RL4LMs)
- **Q7.4:** How do researchers debug and monitor RL training?

### 3.3 Constitutional AI & Safety
- **Q8.1:** What methods exist for AI safety and alignment? (Constitutional AI, RLHF with safety, red teaming)
- **Q8.2:** Which libraries/frameworks support safety-focused training?
- **Q8.3:** What evaluation benchmarks exist for safety and alignment?
- **Q8.4:** How do researchers implement guardrails and content filtering?

---

## 4. Distributed Training & Optimization

### 4.1 Parallelism Strategies
- **Q9.1:** Which parallelism methods are standard? (Data Parallel, Pipeline Parallel, Tensor Parallel, Sequence Parallel, FSDP, ZeRO)
- **Q9.2:** What libraries implement these strategies? (DeepSpeed, FSDP, Megatron-LM, Accelerate, PyTorch DDP)
- **Q9.3:** What are the tradeoffs between parallelism methods?
- **Q9.4:** Which parallelism strategies work best for different model sizes?

### 4.2 Memory Optimization
- **Q10.1:** What memory optimization techniques are used? (gradient checkpointing, mixed precision, ZeRO stages, CPU offloading)
- **Q10.2:** Which libraries provide memory optimization? (DeepSpeed, bitsandbytes, FSDP)
- **Q10.3:** What are best practices for training on limited GPU memory?
- **Q10.4:** Which quantization methods work during training? (QLoRA, 8-bit optimizers)

### 4.3 Training Infrastructure
- **Q11.1:** Which cloud platforms are most used? (Modal, Lambda Labs, RunPod, vast.ai, AWS, GCP)
- **Q11.2:** What orchestration tools manage multi-node training? (Ray, SLURM, Kubernetes)
- **Q11.3:** Which frameworks abstract infrastructure complexity? (Accelerate, Lightning, Composer)
- **Q11.4:** What are best practices for checkpointing and fault tolerance?

---

## 5. Model Evaluation & Analysis

### 5.1 Benchmark Evaluation
- **Q12.1:** Which evaluation frameworks are standard? (lm-evaluation-harness, HELM, OpenCompass, AlpacaEval)
- **Q12.2:** What benchmark suites are used? (MMLU, HumanEval, GSM8K, TruthfulQA, MT-Bench)
- **Q12.3:** How do researchers evaluate domain-specific capabilities?
- **Q12.4:** What tools exist for custom benchmark creation?

### 5.2 Model Interpretability
- **Q13.1:** Which interpretability methods are used? (attention visualization, probing, mechanistic interpretability)
- **Q13.2:** What libraries support model analysis? (TransformerLens, Captum, Inseq)
- **Q13.3:** How do researchers debug model failures?
- **Q13.4:** What tools visualize model behavior?

### 5.3 Performance Profiling
- **Q14.1:** Which profiling tools measure training performance? (PyTorch Profiler, NVIDIA Nsight, TensorBoard)
- **Q14.2:** What metrics do researchers track? (throughput, MFU, memory bandwidth)
- **Q14.3:** How do researchers identify bottlenecks?

---

## 6. Model Serving & Inference

### 6.1 Inference Optimization
- **Q15.1:** Which inference engines are production-ready? (vLLM, TensorRT-LLM, TGI, SGLang, llama.cpp)
- **Q15.2:** What optimization techniques are used? (continuous batching, PagedAttention, quantization, speculative decoding)
- **Q15.3:** Which quantization methods work for inference? (GPTQ, AWQ, GGUF, SmoothQuant)
- **Q15.4:** What are the tradeoffs between inference engines?

### 6.2 Serving Infrastructure
- **Q16.1:** Which serving frameworks are most used? (vLLM, TorchServe, Ray Serve, TGI, Triton)
- **Q16.2:** What are best practices for API design and rate limiting?
- **Q16.3:** How do researchers implement model versioning and A/B testing?
- **Q16.4:** Which monitoring tools track inference performance? (Prometheus, Grafana, W&B)

### 6.3 Edge & Mobile Deployment
- **Q17.1:** Which frameworks support edge deployment? (ONNX Runtime, TFLite, llama.cpp, MLC LLM)
- **Q17.2:** What compression techniques enable mobile deployment?
- **Q17.3:** How do researchers optimize for latency and battery life?

---

## 7. Data Engineering & Management

### 7.1 Dataset Creation & Curation
- **Q18.1:** Which tools help with data collection? (Common Crawl tools, scrapy, synthetic data generation)
- **Q18.2:** What data filtering and deduplication methods are used? (fuzzy dedup, MinHash, Bloom filters)
- **Q18.3:** Which quality assessment tools exist? (perplexity filtering, classifier-based filtering)
- **Q18.4:** What libraries manage large-scale datasets? (Hugging Face Datasets, WebDataset, Ray Data)

### 7.2 Synthetic Data Generation
- **Q19.1:** What methods generate synthetic training data? (self-instruct, Evol-Instruct, distillation)
- **Q19.2:** Which libraries support synthetic data pipelines?
- **Q19.3:** How do researchers validate synthetic data quality?
- **Q19.4:** What are best practices for mixing synthetic and real data?

### 7.3 Data Versioning & Lineage
- **Q20.1:** Which tools track dataset versions? (DVC, Pachyderm, LakeFS)
- **Q20.2:** How do researchers ensure reproducibility?
- **Q20.3:** What metadata standards exist for ML datasets?

---

## 8. Experiment Tracking & MLOps

### 8.1 Experiment Management
- **Q21.1:** Which experiment tracking tools are standard? (Weights & Biases, MLflow, TensorBoard, Neptune.ai)
- **Q21.2:** What metrics do researchers track during training?
- **Q21.3:** How do researchers organize hyperparameter sweeps?
- **Q21.4:** Which tools support collaborative experiment tracking?

### 8.2 Model Registry & Versioning
- **Q22.1:** What model registry solutions exist? (MLflow Model Registry, HuggingFace Hub, W&B Registry)
- **Q22.2:** How do researchers version models and artifacts?
- **Q22.3:** What metadata should be tracked with models?

---

## Output Format

For each question, provide:

1. **Answer:** Specific libraries/tools/papers with brief descriptions
2. **Priority:** High/Medium/Low (based on adoption and production readiness)
3. **Skill Potential:** Yes/No (should we create a Claude skill for this?)
4. **Documentation Quality:** Rate 1-5 (5 = excellent docs available for scraping)
5. **Notes:** Any additional context (emerging vs. established, alternatives, gotchas)

---

## Example Answer Format

**Q1.1: What are the current state-of-the-art architectures for LLMs?**

| Library/Tool | Description | Priority | Skill Potential | Docs Quality | Notes |
|--------------|-------------|----------|-----------------|--------------|-------|
| Llama 3 | Meta's open-source LLM architecture | High | Yes | 4/5 | Well-documented, widely adopted |
| Mistral | MoE-based efficient architecture | High | Yes | 4/5 | Good docs, strong community |
| Mamba | State-space model alternative to Transformers | Medium | Maybe | 3/5 | Emerging, needs more production use |

---

**Deadline:** [Specify date]

**Contact:** [Your contact info for questions]

---

*This questionnaire will guide the creation of a comprehensive AI research skill library for Claude Code.*
