# 🎓 Claude AI Research Skills

> **The most comprehensive open-source library of AI research skills for Claude Code**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Skills](https://img.shields.io/badge/Skills-15-blue.svg)](.)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

Empower Claude Code with deep expertise across the entire AI research lifecycle—from model architecture to production deployment. Each skill provides comprehensive documentation, real-world examples, and battle-tested best practices.

---

## 🌟 Why This Project?

**Problem**: AI researchers juggle 100+ frameworks, libraries, and tools. Documentation is scattered. Best practices are tribal knowledge.

**Solution**: Curated, comprehensive skills that make Claude Code your AI research copilot.

**Impact**: 
- ⚡ **10x faster prototyping** - Skip docs hunting, get straight to coding
- 🎯 **Production-ready patterns** - Learn from real GitHub issues & releases
- 🤝 **Community-driven** - Open-source knowledge base that grows with the field

---

## 📚 Available Skills (16/70 roadmap)

### 🏗️ Model Architecture & Training
Build and train models from scratch or customize existing architectures.

**01-model-architecture/** (2 skills)
- **[Megatron-Core](01-model-architecture/megatron-core/)** - NVIDIA's industry-standard large-scale training (2,342 files, 32 releases)
- **[LitGPT](01-model-architecture/litgpt/)** - Lightning AI's modular LLM implementations (299 files, 27 releases)

**02-tokenization/** (1 skill)
- **[HuggingFace Tokenizers](02-tokenization/huggingface-tokenizers/)** - Fast tokenizers in Rust

### 🎯 Fine-Tuning & Adaptation
Adapt pre-trained models to your specific needs with state-of-the-art methods.

**03-fine-tuning/** (4 skills) ✓ **COMPLETE**
- **[Axolotl](03-fine-tuning/axolotl/)** - YAML-based fine-tuning with 100+ models (185 pages)
- **[TRL](03-fine-tuning/trl-fine-tuning/)** - Transformer Reinforcement Learning (67 releases)
- **[LLaMA-Factory](03-fine-tuning/llama-factory/)** - WebUI no-code fine-tuning (25 pages)
- **[Unsloth](03-fine-tuning/unsloth/)** - 2x faster QLoRA fine-tuning (172 pages)

**04-peft/** (1 skill) ✓ **COMPLETE**
- **[HuggingFace PEFT](04-peft/huggingface-peft/)** - LoRA, QLoRA, AdaLoRA, IA3, DoRA (805 files)

### 🎓 Post-Training & Alignment
Improve models after initial training with reinforcement learning and human feedback.

**06-post-training/** (1 skill)
- **[GRPO-RL-Training](06-post-training/grpo-rl-training/)** - Group Relative Policy Optimization with TRL

### 📊 Data & Safety
Process data at scale and ensure model safety.

**05-data-processing/** (1 skill)
- **[NeMo Curator](05-data-processing/nemo-curator/)** - NVIDIA's distributed data curation

**07-safety-alignment/** (1 skill)
- **[NeMo Guardrails](07-safety-alignment/nemo-guardrails/)** - LLM safety & jailbreak protection (1,887 files)

### ⚡ Distributed Training & Infrastructure
Scale training across multiple GPUs and nodes.

**08-distributed-training/** (3 skills)
- **[DeepSpeed](08-distributed-training/deepspeed/)** - Microsoft's ZeRO optimization (144 pages)
- **[PyTorch FSDP](08-distributed-training/pytorch-fsdp/)** - Fully Sharded Data Parallel (15 pages)
- **[Accelerate](08-distributed-training/huggingface-accelerate/)** - HuggingFace distributed training (400 files)

**09-infrastructure/** (2 skills)
- **[PyTorch Lightning](09-infrastructure/pytorch-lightning/)** - Production training framework (1,238 files, 170 releases)
- **[Ray Train](09-infrastructure/ray-train/)** - Distributed training with Ray (10,892 files, 115 releases)

---

## 🚀 Quick Start

### For Claude Code Users

1. **Clone this repository**:
```bash
git clone https://github.com/YOUR_USERNAME/claude-ai-research-skills.git
cd claude-ai-research-skills
```

2. **Browse skills** - Each directory contains:
   - `SKILL.md` - Quick reference & metadata
   - `references/` - Comprehensive documentation (300KB+ per skill)
   - Real code examples, GitHub issues, and release notes

3. **Use with Claude Code** - Point Claude to any skill directory:
```
"I need help with distributed training using DeepSpeed"
→ Claude reads: 08-distributed-training/deepspeed/
```

### For Skill Creators

1. **Read [CONTRIBUTING.md](CONTRIBUTING.md)** - Step-by-step guide
2. **Use [SKILL_TEMPLATE.md](SKILL_TEMPLATE.md)** - Copy-paste scaffold
3. **Run validation**: `python scripts/validate_skill.py your-skill/`
4. **Submit PR** - We review within 48 hours

---

## 📖 Skill Structure

Each skill follows a battle-tested format for maximum usefulness:

```
skill-name/
├── SKILL.md                    # Quick reference (50-150 lines)
│   ├── Metadata (name, description, version)
│   ├── When to use this skill
│   ├── Quick patterns & examples
│   └── Links to references
│
├── references/                 # Deep documentation (300KB+)
│   ├── README.md              # From GitHub/official docs
│   ├── api.md                 # API reference
│   ├── tutorials.md           # Step-by-step guides
│   ├── issues.md              # Real GitHub issues & solutions
│   ├── releases.md            # Version history & breaking changes
│   └── file_structure.md      # Codebase navigation
│
├── scripts/                    # Helper scripts (optional)
└── assets/                     # Templates & examples (optional)
```

**Quality Standards**:
- ✅ 300KB+ documentation from official sources
- ✅ Real GitHub issues & solutions (when available)
- ✅ Code examples with language detection
- ✅ Version history & breaking changes
- ✅ Links to official docs

---

## 🗺️ Roadmap

### ✅ Phase 1: Training Foundation (COMPLETE - 15 skills)
**Status**: 70% of training lifecycle covered

### 🚧 Phase 2: Complete Training Stack (Weeks 1-2) - 5 Skills
**Priority**: HIGH - Finish what we started

- [ ] **OpenRLHF** - Open-source RLHF implementation
- [ ] **VERL** - VolcEngine RL for LLMs  
- [ ] **Flash Attention** - Tri Dao's kernel optimization
- [ ] **bitsandbytes** - 8-bit training/inference
- [ ] **RWKV** - State-space model alternative

### 🎯 Phase 3: Inference & Serving (Weeks 3-4) - 6 Skills
**Priority**: CRITICAL - Enable production deployment

- [ ] **vLLM** - PagedAttention, continuous batching
- [ ] **TensorRT-LLM** - NVIDIA inference optimization
- [ ] **llama.cpp** - CPU/edge inference
- [ ] **SGLang** - Fast structured generation
- [ ] **GPTQ** - Post-training quantization
- [ ] **AWQ** - Activation-aware quantization

### 📊 Phase 4: Evaluation & Data (Weeks 5-6) - 5 Skills
**Priority**: HIGH - Complete research lifecycle

- [ ] **lm-evaluation-harness** - EleutherAI benchmarks
- [ ] **HELM** - Stanford evaluation
- [ ] **AlpacaEval** - Instruction-following eval
- [ ] **Ray Data** - Distributed data processing
- [ ] **HuggingFace Datasets** - Dataset management

### 📈 Phase 5: MLOps (Weeks 7-8) - 4 Skills
**Priority**: MEDIUM-HIGH - Production readiness

- [ ] **Weights & Biases** - Experiment tracking
- [ ] **MLflow** - Open-source MLOps
- [ ] **TensorBoard** - Visualization
- [ ] **HuggingFace Hub** - Model registry

### 🤖 Phase 6: Applications (Weeks 9-12) - 12 Skills
**Priority**: CRITICAL - Enable AI applications

**Agents**:
- [ ] **LangChain** - Most popular agent framework
- [ ] **LlamaIndex** - Data-focused agents
- [ ] **CrewAI** - Multi-agent collaboration
- [ ] **AutoGPT** - Autonomous agents

**RAG**:
- [ ] **Pinecone** - Vector database
- [ ] **ChromaDB** - Open-source vector DB
- [ ] **LlamaIndex RAG** - RAG pipelines
- [ ] **Sentence Transformers** - Embeddings

**Tools**:
- [ ] **DSPy** - Prompt optimization
- [ ] **Instructor** - Structured output
- [ ] **LangSmith** - Observability
- [ ] **Guardrails AI** - Output validation

### 🔬 Phase 7: Specialized (Weeks 13-16) - 8 Skills
**Priority**: MEDIUM - Cutting-edge techniques

- [ ] **LLaVA** - Vision-language models
- [ ] **Whisper** - Speech-to-text
- [ ] **Stable Diffusion** - Image generation
- [ ] **MoE Training** - Mixture of Experts
- [ ] **Model Merging** - mergekit
- [ ] **Long Context** - RoPE extensions
- [ ] **Speculative Decoding**
- [ ] **Distillation**

**Target**: 70 comprehensive skills by month 6

---

## 🤝 Contributing

We welcome contributions from the AI research community! 

### Ways to Contribute

1. **Add a new skill** - See [CONTRIBUTING.md](CONTRIBUTING.md)
2. **Improve existing skills** - Update docs, add examples
3. **Report issues** - Outdated info, broken links
4. **Share feedback** - What skills do you need?

### Quick Contribution Guide

```bash
# 1. Fork & clone
git clone https://github.com/YOUR_USERNAME/claude-ai-research-skills.git

# 2. Create skill from template
cp SKILL_TEMPLATE.md 12-inference-serving/vllm/SKILL.md

# 3. Fill in documentation
# ... edit SKILL.md and add references/

# 4. Validate
python scripts/validate_skill.py 12-inference-serving/vllm/

# 5. Submit PR
git checkout -b add-vllm-skill
git add 12-inference-serving/vllm/
git commit -m "Add vLLM inference skill"
git push origin add-vllm-skill
```

**Recognition**: All contributors are featured in our [Contributors Hall of Fame](CONTRIBUTORS.md) 🌟

---

## 📊 Project Stats

| Metric | Current | Target (6 months) |
|--------|---------|-------------------|
| **Skills** | 16 | 70 |
| **Documentation** | 4.5MB | 20MB+ |
| **GitHub Issues Captured** | 143 | 1000+ |
| **Release Notes Tracked** | 562 | 3000+ |
| **Contributors** | 1 | 100+ |
| **Coverage** | Training (70%) | Full Lifecycle (100%) |

---

## 🏗️ Repository Structure

```
claude-ai-research-skills/
├── README.md                    ← You are here
├── CONTRIBUTING.md              ← Contribution guide
├── SKILL_TEMPLATE.md            ← Skill scaffold
├── PROJECT_ANALYSIS.md          ← Strategic roadmap
├── RESEARCH_QUESTIONNAIRE.md    ← Coverage checklist
│
├── 01-model-architecture/       (2 skills)
├── 02-tokenization/             (1 skill)
├── 03-fine-tuning/              (4 skills ✓)
├── 04-peft/                     (1 skill ✓)
├── 05-data-processing/          (1 skill)
├── 06-post-training/            (planned)
├── 07-safety-alignment/         (1 skill)
├── 08-distributed-training/     (3 skills)
├── 09-infrastructure/           (2 skills)
├── 10-optimization/             (planned)
├── 11-evaluation/               (planned)
├── 12-inference-serving/        (planned - CRITICAL)
├── 13-mlops/                    (planned)
├── 14-agents/                   (planned - CRITICAL)
├── 15-rag/                      (planned - CRITICAL)
├── 16-prompt-engineering/       (planned)
├── 17-observability/            (planned)
├── 18-multimodal/               (planned)
└── 19-emerging-techniques/      (planned)
```

---

## 🎯 Use Cases

### For Researchers
"I need to fine-tune Llama 3 with custom data"
→ **03-fine-tuning/axolotl/** - YAML configs, 100+ model support

### For ML Engineers  
"How do I optimize inference latency?"
→ **12-inference-serving/vllm/** (coming soon) - PagedAttention, batching

### For Students
"I want to learn how transformers work"
→ **01-model-architecture/litgpt/** - Clean implementations

### For Teams
"We need to scale training to 100 GPUs"
→ **08-distributed-training/deepspeed/** - ZeRO stages, 3D parallelism

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

**Note**: Individual skills may reference libraries with different licenses. Please check each project's license before use.

---

## 🙏 Acknowledgments

Built with:
- 💜 **[Claude Code](https://claude.ai/claude-code)** - AI pair programming
- ⚡ **[Skill Seeker](https://github.com/yusufkaraaslan/skill-seeker)** - Automated doc scraping
- 🤗 **Open Source AI Community** - For amazing tools and docs

Special thanks to:
- EleutherAI, HuggingFace, NVIDIA, Lightning AI, Meta AI, Anthropic
- All researchers who maintain excellent documentation

---

## 📞 Get Involved

- **🐛 Issues**: [Report bugs or request skills](https://github.com/YOUR_USERNAME/claude-ai-research-skills/issues)
- **💬 Discussions**: [Ask questions & share ideas](https://github.com/YOUR_USERNAME/claude-ai-research-skills/discussions)
- **🌟 Star**: Help others discover this project
- **🔀 Fork**: Build your own skill collection

---

## 📈 Recent Updates

**November 6, 2025** - v0.2.0
- ✅ Added 10 skills from GitHub (Megatron-Core, Lightning, Ray Train, etc.)
- ✅ Improved skill structure with comprehensive references
- ✅ Created strategic roadmap to 70 skills
- ✅ Added contribution guidelines

**October 2025** - v0.1.0
- 🎉 Initial release with 5 fine-tuning skills

---

<p align="center">
  <strong>Help us build the definitive AI research skill library</strong><br>
  <a href="CONTRIBUTING.md">Contribute</a> •
  <a href="PROJECT_ANALYSIS.md">Roadmap</a> •
  <a href="https://github.com/YOUR_USERNAME/claude-ai-research-skills/discussions">Community</a>
</p>

<p align="center">
  Made with 💜 by the AI research community
</p>
