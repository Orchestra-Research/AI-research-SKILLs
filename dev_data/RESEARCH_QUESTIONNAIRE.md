# AI Research Skills Discovery Questionnaire

**Purpose:** Guide literature research to identify critical topics, libraries, and best practices needed for full-stack AI researchers.

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

## 9. Multimodal & Specialized Models

### 9.1 Vision-Language Models
- **Q24.1:** Which VLM architectures are current? (LLaVA, Flamingo, BLIP, GPT-4V style)
- **Q24.2:** What libraries train vision-language models? (LLaVA, OpenFlamingo)
- **Q24.3:** How do researchers align vision and language encoders?
- **Q24.4:** What evaluation benchmarks exist for VLMs?

### 9.2 Code & Math Models
- **Q25.1:** What specialized techniques improve code generation? (execution feedback, unit test generation)
- **Q25.2:** Which libraries support math reasoning training? (NuminaMath, Lean integration)
- **Q25.3:** What evaluation frameworks exist for code/math? (HumanEval+, MATH, APPS)

### 9.3 Audio & Speech Models
- **Q26.1:** Which speech-to-text models are state-of-the-art? (Whisper, wav2vec 2.0)
- **Q26.2:** What text-to-speech models are production-ready? (Bark, VALL-E, Tortoise)
- **Q26.3:** Which libraries support audio model training?

---

## 10. Emerging Techniques & Research Frontiers

### 10.1 Long-Context Models
- **Q27.1:** What techniques extend context length? (RoPE extensions, ALiBi, Flash Attention)
- **Q27.2:** Which models support 100K+ context windows?
- **Q27.3:** How do researchers evaluate long-context understanding?

### 10.2 Mixture of Experts (MoE)
- **Q28.1:** Which MoE architectures are production-ready? (Mixtral, Switch Transformers)
- **Q28.2:** What libraries support MoE training? (Megablocks, DeepSpeed-MoE)
- **Q28.3:** What are the engineering challenges of MoE?

### 10.3 Test-Time Compute & Inference Scaling
- **Q29.1:** What methods improve inference-time reasoning? (chain-of-thought, tree-of-thoughts, self-consistency)
- **Q29.2:** Which libraries implement advanced inference strategies?
- **Q29.3:** How do researchers balance compute cost and quality?

### 10.4 Model Merging & Composition
- **Q30.1:** What model merging techniques exist? (SLERP, TIES, DARE, task arithmetic)
- **Q30.2:** Which tools merge models? (mergekit, model soups)
- **Q30.3:** When is model merging effective vs. multi-task training?

---

## 11. Domain-Specific Considerations

### 11.1 Scientific Research
- **Q31.1:** Which models/tools support scientific domains? (biology, chemistry, physics)
- **Q31.2:** What specialized pretraining datasets exist?
- **Q31.3:** How do researchers integrate domain knowledge?

### 11.2 Enterprise & Production
- **Q32.1:** What privacy-preserving training methods exist? (federated learning, differential privacy)
- **Q32.2:** Which tools support on-premise deployment?
- **Q32.3:** How do enterprises handle model governance and compliance?

### 11.3 Low-Resource Settings
- **Q33.1:** What techniques work with limited data? (few-shot learning, meta-learning, data augmentation)
- **Q33.2:** Which methods work with limited compute? (distillation, pruning, efficient architectures)
- **Q33.3:** What multilingual techniques support low-resource languages?

---

## 12. Tooling & Development Environment

### 12.1 Development Tools
- **Q34.1:** Which IDEs/editors are used for ML research? (VSCode extensions, JupyterLab, Google Colab)
- **Q34.2:** What debugging tools help with distributed training?
- **Q34.3:** Which visualization tools are standard?

### 12.2 Prototyping & Rapid Experimentation
- **Q35.1:** Which frameworks enable fast prototyping? (Lightning, Composer, Keras)
- **Q35.2:** What notebook environments support GPU access? (Colab, Kaggle, SageMaker)
- **Q35.3:** How do researchers transition from prototype to production?

---

## 13. Agent Frameworks & Orchestration

### 13.1 Agent Frameworks
- **Q36.1:** Which agent frameworks are production-ready? (LangChain, LlamaIndex, AutoGPT, CrewAI, Semantic Kernel)
- **Q36.2:** What multi-agent coordination patterns exist?
- **Q36.3:** Which frameworks support tool-use and function calling?
- **Q36.4:** How do agent frameworks handle memory management?

### 13.2 Agent Reasoning & Planning
- **Q37.1:** What reasoning frameworks are used? (ReAct, Reflexion, Tree-of-Thoughts)
- **Q37.2:** Which planning algorithms work for agent tasks?
- **Q37.3:** How do agents decompose complex tasks?
- **Q37.4:** What error recovery strategies do agents use?

### 13.3 Tool Integration
- **Q38.1:** How do agents execute code safely? (sandboxed environments, E2B, Modal)
- **Q38.2:** What web search integrations are standard? (Serper, Tavily, Bing API)
- **Q38.3:** Which calculator/math tools do agents use?
- **Q38.4:** How do agents orchestrate multiple API calls?

---

## 14. RAG (Retrieval-Augmented Generation)

### 14.1 Vector Databases
- **Q39.1:** Which vector databases are production-ready? (Pinecone, Weaviate, Milvus, Chroma, Qdrant, FAISS)
- **Q39.2:** What are the tradeoffs between vector databases? (latency, scale, features)
- **Q39.3:** Which databases support hybrid search? (vector + keyword)
- **Q39.4:** How do teams handle vector database scaling?

### 14.2 Embeddings & Retrieval
- **Q40.1:** Which embedding models are standard? (sentence-transformers, OpenAI, Cohere, BGE)
- **Q40.2:** What chunking strategies work best? (recursive, semantic, sliding window)
- **Q40.3:** How do teams implement reranking? (Cohere rerank, cross-encoders)
- **Q40.4:** What metadata filtering strategies are used?

### 14.3 Document Processing
- **Q41.1:** Which document loaders are used? (unstructured.io, LlamaIndex, LangChain loaders)
- **Q41.2:** How do teams handle multi-modal documents? (PDFs with images, tables)
- **Q41.3:** What OCR tools are integrated with RAG pipelines?
- **Q41.4:** How do teams update vector stores incrementally?

---

## 15. Prompt Engineering & Management

### 15.1 Prompt Templates & Versioning
- **Q42.1:** Which prompt management tools exist? (PromptLayer, Helicone, LangSmith)
- **Q42.2:** How do teams version and test prompts?
- **Q42.3:** What templating systems are used? (Jinja2, f-strings, LangChain PromptTemplate)
- **Q42.4:** How do teams A/B test prompts in production?

### 15.2 Prompt Optimization
- **Q43.1:** Which prompt optimization techniques exist? (DSPy, PromptPerfect, few-shot selection)
- **Q43.2:** How do teams automate few-shot example selection?
- **Q43.3:** What chain-of-thought strategies are standard?
- **Q43.4:** How do teams handle prompt length optimization?

### 15.3 Context Management
- **Q44.1:** What context compression techniques are used? (summarization, pruning)
- **Q44.2:** How do teams manage long conversation histories?
- **Q44.3:** Which memory systems preserve context across sessions? (Redis, PostgreSQL)
- **Q44.4:** What entity extraction methods track conversation state?

---

## 16. Structured Output & Parsing

### 16.1 Schema Enforcement
- **Q45.1:** Which libraries enforce JSON/schema output? (instructor, Pydantic, guidance, outlines)
- **Q45.2:** What constrained decoding methods exist? (guidance, lm-format-enforcer)
- **Q45.3:** How do teams handle schema validation failures?
- **Q45.4:** Which tools support complex nested schemas?

### 16.2 Output Parsing
- **Q46.1:** What parsing strategies handle malformed LLM output?
- **Q46.2:** How do teams extract structured data from unstructured text?
- **Q46.3:** Which regex/parser libraries are commonly used?
- **Q46.4:** What retry strategies work for parsing failures?

---

## 17. LLM Application Observability

### 17.1 Monitoring & Tracing
- **Q47.1:** Which monitoring tools track LLM applications? (LangSmith, Phoenix, Weights & Biases)
- **Q47.2:** How do teams trace multi-step agent workflows? (OpenTelemetry, LangChain callbacks)
- **Q47.3:** What latency monitoring strategies are used?
- **Q47.4:** How do teams debug production LLM failures?

### 17.2 Cost & Usage Tracking
- **Q48.1:** Which tools track token usage and costs?
- **Q48.2:** How do teams implement cost budgets and alerts?
- **Q48.3:** What strategies reduce API costs? (caching, prompt optimization)
- **Q48.4:** How do teams forecast LLM infrastructure costs?

### 17.3 Quality Metrics
- **Q49.1:** How do teams detect hallucinations? (self-consistency, fact-checking)
- **Q49.2:** What relevance scoring methods are used for RAG?
- **Q49.3:** Which tools measure response quality? (RAGAS, LLM-as-judge)
- **Q49.4:** How do teams monitor model drift in production?

---

## 18. LLM Application Security & Safety

### 18.1 Prompt Injection Defense
- **Q50.1:** What prompt injection defense techniques exist?
- **Q50.2:** Which guardrail frameworks are used? (NeMo Guardrails, Guardrails AI, LlamaGuard)
- **Q50.3:** How do teams sanitize user inputs?
- **Q50.4:** What adversarial testing methods detect vulnerabilities?

### 18.2 Content Moderation & Filtering
- **Q51.1:** Which content moderation APIs are used? (OpenAI Moderation, Perspective API)
- **Q51.2:** How do teams detect and filter PII?
- **Q51.3:** What output filtering strategies are standard?
- **Q51.4:** How do teams handle toxic or harmful outputs?

### 18.3 Access Control & Rate Limiting
- **Q52.1:** What authentication methods secure LLM APIs? (API keys, OAuth, JWT)
- **Q52.2:** How do teams implement rate limiting? (token budgets, request limits)
- **Q52.3:** Which API gateway solutions are used?
- **Q52.4:** How do teams prevent abuse and misuse?

---

## 19. Application Development & Deployment

### 19.1 API Development
- **Q53.1:** Which frameworks serve LLM APIs? (FastAPI, Flask, Express.js)
- **Q53.2:** What streaming response patterns are used? (Server-Sent Events, WebSockets)
- **Q53.3:** How do teams handle API versioning?
- **Q53.4:** What load balancing strategies work for LLM services?

### 19.2 Testing & Validation
- **Q54.1:** Which testing frameworks exist for LLM apps? (pytest, unittest, LangChain eval)
- **Q54.2:** How do teams implement unit tests for LLM logic?
- **Q54.3:** What integration testing strategies are used?
- **Q54.4:** How do teams detect regression in LLM behavior?

### 19.3 Frontend Integration
- **Q55.1:** Which UI libraries integrate with LLM backends? (React, Streamlit, Gradio)
- **Q55.2:** What chat UI components are standard? (Vercel AI SDK, ChatGPT UI patterns)
- **Q55.3:** How do teams handle streaming UI updates?
- **Q55.4:** What accessibility standards apply to LLM interfaces?

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
