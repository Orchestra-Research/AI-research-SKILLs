# 🗺️ Roadmap

## Vision

Build the most comprehensive open-source library of AI research skills, enabling AI agents to autonomously conduct experiments from hypothesis to deployment.

**Target**: 70 comprehensive skills by month 6

## Progress Overview

| Metric | Current | Target |
|--------|---------|--------|
| **Skills** | **43** (high-quality) | 70 |
| **Avg Lines/Skill** | **380 lines** (focused + progressive disclosure) | 200-500 lines |
| **Documentation** | **~65,000 lines** total (SKILL.md + references) | 70,000+ lines |
| **Gold Standard Skills** | **35** with comprehensive references | 50+ |
| **Coverage** | Architecture, Tokenization, Fine-Tuning, Data Processing, Post-Training, Safety, Distributed, Optimization, Evaluation, Inference, Agents, RAG, Multimodal | Full Lifecycle |

## Development Phases

### ✅ Phase 1: Model Architecture (COMPLETE - 5 skills)
**Status**: Core model architectures covered

**Completed Skills**:
- ✅ **Megatron-Core** - NVIDIA's framework for training 2B-462B param models
- ✅ **LitGPT** - Lightning AI's 20+ clean LLM implementations
- ✅ **Mamba** - State-space models with O(n) complexity
- ✅ **RWKV** - RNN+Transformer hybrid, infinite context
- ✅ **NanoGPT** - Educational GPT in ~300 lines by Karpathy

### ✅ Phase 2: Tokenization (COMPLETE - 2 skills)
**Status**: Essential tokenization frameworks covered

**Completed Skills**:
- ✅ **HuggingFace Tokenizers** - Rust-based, BPE/WordPiece/Unigram
- ✅ **SentencePiece** - Language-independent tokenization

### ✅ Phase 3: Fine-Tuning (COMPLETE - 3 skills)
**Status**: Core fine-tuning frameworks covered

**Completed Skills**:
- ✅ **Axolotl** - YAML-based fine-tuning with 100+ models
- ✅ **LLaMA-Factory** - WebUI no-code fine-tuning
- ✅ **Unsloth** - 2x faster QLoRA fine-tuning

### ✅ Phase 4: Data Processing (COMPLETE - 2 skills)
**Status**: Distributed data processing covered

**Completed Skills**:
- ✅ **Ray Data** - Distributed ML data processing
- ✅ **NeMo Curator** - GPU-accelerated data curation

### ✅ Phase 5: Post-Training (COMPLETE - 4 skills)
**Status**: RLHF and alignment techniques covered

**Completed Skills**:
- ✅ **TRL Fine-Tuning** - Transformer Reinforcement Learning
- ✅ **GRPO-RL-Training** - Group Relative Policy Optimization (gold standard)
- ✅ **OpenRLHF** - Full RLHF pipeline with Ray + vLLM
- ✅ **SimPO** - Simple Preference Optimization

### ✅ Phase 6: Safety & Alignment (COMPLETE - 3 skills)
**Status**: Core safety frameworks covered

**Completed Skills**:
- ✅ **Constitutional AI** - AI-driven self-improvement via principles
- ✅ **LlamaGuard** - Safety classifier for LLM inputs/outputs
- ✅ **NeMo Guardrails** - Programmable guardrails with Colang

### ✅ Phase 7: Distributed Training (COMPLETE - 5 skills)
**Status**: Major distributed training frameworks covered

**Completed Skills**:
- ✅ **DeepSpeed** - Microsoft's ZeRO optimization
- ✅ **PyTorch FSDP** - Fully Sharded Data Parallel
- ✅ **Accelerate** - HuggingFace's distributed training API
- ✅ **PyTorch Lightning** - High-level training framework
- ✅ **Ray Train** - Multi-node orchestration

### ✅ Phase 8: Optimization (COMPLETE - 3 skills)
**Status**: Core optimization techniques covered

**Completed Skills**:
- ✅ **Flash Attention** - 2-4x faster attention with memory efficiency
- ✅ **bitsandbytes** - 8-bit/4-bit quantization
- ✅ **GPTQ** - 4-bit post-training quantization

### ✅ Phase 9: Evaluation (COMPLETE - 1 skill)
**Status**: Standard benchmarking framework available

**Completed Skills**:
- ✅ **lm-evaluation-harness** - EleutherAI's standard for benchmarking LLMs

### ✅ Phase 10: Inference & Serving (COMPLETE - 4 skills)
**Status**: Production inference frameworks covered

**Completed Skills**:
- ✅ **vLLM** - High-throughput LLM serving with PagedAttention
- ✅ **TensorRT-LLM** - NVIDIA's fastest inference
- ✅ **llama.cpp** - CPU/Apple Silicon inference
- ✅ **SGLang** - Structured generation with RadixAttention

### ✅ Phase 11: Agents (COMPLETE - 2 skills)
**Status**: Major agent frameworks covered

**Completed Skills**:
- ✅ **LangChain** - Most popular agent framework, 500+ integrations
- ✅ **LlamaIndex** - Data framework for LLM apps, 300+ connectors

### ✅ Phase 12: RAG (COMPLETE - 4 skills)
**Status**: Core RAG and vector database skills covered

**Completed Skills**:
- ✅ **Chroma** - Open-source embedding database
- ✅ **FAISS** - Facebook's similarity search, billion-scale
- ✅ **Sentence Transformers** - 5000+ embedding models
- ✅ **Pinecone** - Managed vector database

### ✅ Phase 13: Multimodal (COMPLETE - 3 skills)
**Status**: Core multimodal frameworks covered

**Completed Skills**:
- ✅ **CLIP** - OpenAI's vision-language model
- ✅ **Whisper** - Robust speech recognition, 99 languages
- ✅ **LLaVA** - Vision-language assistant, GPT-4V level

### 🚧 Phase 14: Advanced Optimization (HIGH PRIORITY) - 5 skills
**Status**: In progress - Advanced quantization techniques

- [ ] **AWQ** - Activation-aware quantization
- [ ] **HQQ** - Half-Quadratic Quantization
- [ ] **AutoAWQ** - Automated AWQ implementation
- [ ] **GGUF** - Quantization format for llama.cpp
- [ ] **Activation Offloading** - Memory optimization techniques

### 🚧 Phase 15: MLOps & Observability (MEDIUM-HIGH PRIORITY) - 6 skills
**Status**: Planned - Production monitoring and tracking

- [ ] **Weights & Biases** - Experiment tracking
- [ ] **MLflow** - Open-source MLOps
- [ ] **TensorBoard** - Visualization
- [ ] **LangSmith** - LLM observability
- [ ] **Phoenix** - LLM monitoring
- [ ] **HuggingFace Hub** - Model registry

### 🚧 Phase 16: Advanced Applications (MEDIUM PRIORITY) - 6 skills
**Status**: Planned - Specialized application frameworks

- [ ] **DSPy** - Prompt optimization
- [ ] **Instructor** - Structured output
- [ ] **CrewAI** - Multi-agent collaboration
- [ ] **AutoGPT** - Autonomous agents
- [ ] **Haystack** - NLP framework
- [ ] **Semantic Kernel** - Microsoft's AI orchestration

### 🚧 Phase 17: Extended Multimodal (MEDIUM PRIORITY) - 5 skills
**Status**: Planned - Additional multimodal capabilities

- [ ] **Stable Diffusion** - Image generation
- [ ] **BLIP-2** - Vision-language pretraining
- [ ] **SAM** - Segment Anything Model
- [ ] **AudioCraft** - Audio generation
- [ ] **ImageBind** - Multi-modal embeddings

### 🚧 Phase 18: Emerging Techniques (LOW-MEDIUM PRIORITY) - 6 skills
**Status**: Planned - Cutting-edge research techniques

- [ ] **MoE Training** - Mixture of Experts
- [ ] **Model Merging** - mergekit
- [ ] **Long Context** - RoPE extensions, ALiBi
- [ ] **Speculative Decoding** - Faster inference
- [ ] **Distillation** - Knowledge distillation
- [ ] **Pruning** - Model compression

## Contributing to the Roadmap

Want to help us achieve these goals?

1. **Pick a skill from the roadmap** - Comment on [GitHub Discussions](https://github.com/zechenzhangAGI/claude-ai-research-skills/discussions) to claim it
2. **Follow the [contribution guide](CONTRIBUTING.md)** - Use our template and quality standards
3. **Submit your PR** - We review within 48 hours

## Priority Areas

If you're looking to make the biggest impact, focus on:

1. **Phase 14: Advanced Optimization** (HIGH PRIORITY) - AWQ, HQQ, activation offloading for better model compression
2. **Phase 15: MLOps & Observability** (MEDIUM-HIGH PRIORITY) - Production monitoring, experiment tracking
3. **Phase 16: Advanced Applications** (MEDIUM PRIORITY) - DSPy, Instructor, CrewAI for specialized workflows

## Philosophy

**Quality over Quantity**: Each skill must provide real value with comprehensive guidance, not just links to docs. We aim for 300+ lines of expert-level content per skill, with real code examples, troubleshooting guides, and production-ready workflows.
