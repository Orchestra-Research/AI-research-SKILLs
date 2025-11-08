# AI Research Skills Library

> **The most comprehensive open-source library of AI research skills for AI models**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Skills](https://img.shields.io/badge/Skills-24-blue.svg)](.)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

## Our Mission

**Enable AI agents to autonomously conduct AI research**—from hypothesis to experimental verification. We provide the foundational skills that empower **AI research agents** to conduct experiments, including preparing datasets, executing training pipelines, deploying models, and validating scientific hypotheses.

 

<p align="center">
  <img src="docs/skills.png" alt="AI Research Agent System" width="50%">
</p>
  

## Path Towards AI Research Agent

**Problem**: Conducting AI research requires mastering 100+ frameworks and tools. Experimentation is slow, error-prone, and requires deep expertise in each component of the stack.

**Solution**: Provide AI agents with comprehensive, battle-tested skills for every stage of AI research—enabling autonomous hypothesis testing through systematic experimentation.

**Impact**:
- 🔬 **Hypothesis-Driven Research** - Agents design and execute experiments to test scientific ideas
- ⚡ **Automated Experimentation** - From data processing to model deployment without manual intervention
- 🎯 **Production-Ready Workflows** - Real-world patterns from GitHub issues and release notes
- 🤝 **Reproducible Science** - Standardized, documented experimentation protocols

## 📚 Available Skills (24/70 roadmap)

**Quality over quantity**: Each skill provides comprehensive, expert-level guidance with real code examples, troubleshooting guides, and production-ready workflows.

### 🏗️ Model Architecture (5 skills)
- **[Megatron-Core](01-model-architecture/megatron-core/)** - NVIDIA's framework for training 2B-462B param models with 47% MFU on H100 (359 lines + 4 refs)
- **[LitGPT](01-model-architecture/litgpt/)** - Lightning AI's 20+ clean LLM implementations with production training recipes (462 lines + 4 refs)
- **[Mamba](01-model-architecture/mamba/)** - State-space models with O(n) complexity, 5× faster than Transformers (253 lines + 3 refs)
- **[RWKV](01-model-architecture/rwkv/)** - RNN+Transformer hybrid, infinite context, Linux Foundation project (253 lines + 3 refs)
- **[NanoGPT](01-model-architecture/nanogpt/)** - Educational GPT in ~300 lines by Karpathy (283 lines + 3 refs)

### 🎯 Fine-Tuning (4 skills)
- **[Axolotl](03-fine-tuning/axolotl/)** - YAML-based fine-tuning with 100+ models (156 lines + 4 refs)
- **[TRL](03-fine-tuning/trl-fine-tuning/)** - Transformer Reinforcement Learning (447 lines + 4 refs)
- **[LLaMA-Factory](03-fine-tuning/llama-factory/)** - WebUI no-code fine-tuning (78 lines + 5 refs)
- **[Unsloth](03-fine-tuning/unsloth/)** - 2x faster QLoRA fine-tuning (75 lines + 4 refs)

### 🎓 Post-Training (3 skills)
- **[GRPO-RL-Training](06-post-training/grpo-rl-training/)** - Group Relative Policy Optimization with TRL (569 lines, **gold standard**)
- **[OpenRLHF](06-post-training/openrlhf/)** - Full RLHF pipeline with Ray + vLLM (241 lines + 4 refs)
- **[SimPO](06-post-training/simpo/)** - Simple Preference Optimization, no reference model needed (211 lines + 3 refs)

### 🛡️ Safety & Alignment (3 skills)
- **[Constitutional AI](07-safety-alignment/constitutional-ai/)** - AI-driven self-improvement via principles (282 lines)
- **[LlamaGuard](07-safety-alignment/llamaguard/)** - Safety classifier for LLM inputs/outputs (329 lines)
- **[NeMo Guardrails](07-safety-alignment/nemo-guardrails/)** - Programmable guardrails with Colang (289 lines)

### ⚡ Distributed Training (2 skills)
- **[DeepSpeed](08-distributed-training/deepspeed/)** - Microsoft's ZeRO optimization (137 lines + 9 refs)
- **[PyTorch FSDP](08-distributed-training/pytorch-fsdp/)** - Fully Sharded Data Parallel (124 lines + 2 refs)

### 🔧 Infrastructure (3 skills)
- **[Accelerate](09-infrastructure/accelerate/)** - HuggingFace's 4-line distributed training API (324 lines + 3 refs)
- **[PyTorch Lightning](09-infrastructure/pytorch-lightning/)** - High-level training framework with Trainer class (339 lines + 3 refs)
- **[Ray Train](09-infrastructure/ray-train/)** - Multi-node orchestration and hyperparameter tuning (399 lines + 1 ref)

### 🚀 Optimization (2 skills)
- **[Flash Attention](10-optimization/flash-attention/)** - 2-4x faster attention with memory efficiency (359 lines + 2 refs)
- **[bitsandbytes](10-optimization/bitsandbytes/)** - 8-bit/4-bit quantization for 50-75% memory reduction (403 lines + 3 refs)

### 📊 Evaluation (1 skill)
- **[lm-evaluation-harness](11-evaluation/lm-evaluation-harness/)** - EleutherAI's standard for benchmarking LLMs across 60+ tasks (482 lines + 4 refs)

### 🔥 Inference & Serving (1 skill)
- **[vLLM](12-inference-serving/vllm/)** - High-throughput LLM serving with PagedAttention (356 lines + 4 refs, **production-ready**)

---

## 🚀 Quick Start

### For AI Users

1. **Clone this repository**:
```bash
git clone https://github.com/zechenzhangAGI/claude-ai-research-skills.git
cd claude-ai-research-skills
```

2. **Browse skills** - Each directory contains:
   - `SKILL.md` - Expert-level guidance (100-600 lines)
   - `references/` - Supporting documentation
   - Real code examples, workflows, and troubleshooting

3. **Use with your AI assistant**:
   - **Upload as skill**: Package into `.zip` and upload to any AI assistant that supports custom skills
   - **Attach as context**: Copy-paste `SKILL.md` content or attach as file
   - **RAG knowledge**: Inject into your custom agent's knowledge base
   - **Claude Code**: Point to skill directory for native integration

### For Skill Creators

1. **Read [CONTRIBUTING.md](CONTRIBUTING.md)** - Step-by-step guide
2. **Use [SKILL_TEMPLATE.md](SKILL_TEMPLATE.md)** - Copy-paste scaffold
3. **Run validation**: `python scripts/validate_skill.py your-skill/`
4. **Submit PR** - We review within 48 hours

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
- 300KB+ documentation from official sources
- Real GitHub issues & solutions (when available)
- Code examples with language detection
- Version history & breaking changes
- Links to official docs

## 🗺️ Roadmap

We're building towards 70 comprehensive skills across the full AI research lifecycle. See our [detailed roadmap](ROADMAP.md) for the complete development plan.

**Current Progress**: 24 high-quality skills | **Target**: 70 skills

**Priority Areas**:
- 🎯 **Inference & Serving** (CRITICAL) - vLLM, TensorRT-LLM, llama.cpp
- 🤖 **Applications** (CRITICAL) - LangChain, RAG, agents
- 🚧 **Training Stack** (HIGH) - Complete RLHF, quantization, optimization

[View Full Roadmap →](ROADMAP.md)

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
git clone https://github.com/zechenzhangAGI/claude-ai-research-skills.git

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

## 📊 Project Stats

| Metric | Current | Target |
|--------|---------|--------|
| **Skills** | **24** (high-quality) | 70 |
| **Avg Lines/Skill** | **295 lines** (focused + progressive disclosure) | 200-500 lines |
| **Documentation** | **~40,000 lines** total (SKILL.md + references) | 60,000+ lines |
| **Gold Standard Skills** | **18** with comprehensive references | 50+ |
| **Contributors** | 1 | 100+ |
| **Coverage** | Architecture, Fine-Tuning, Post-Training, Safety, Distributed, Infrastructure, Optimization, Evaluation, Inference | Full Lifecycle |

**Recent Progress**: +11 skills (Mamba, RWKV, NanoGPT, Accelerate, PyTorch Lightning, Ray Train, OpenRLHF, SimPO, Constitutional AI, LlamaGuard, NeMo Guardrails), +30 reference files, comprehensive safety & alignment coverage

**Philosophy**: Quality > Quantity. Following [Anthropic official best practices](anthropic_official_docs/best_practices.md) - each skill provides 200-500 lines of focused, actionable guidance with progressive disclosure.

## 🏗️ Repository Structure

```
claude-ai-research-skills/
├── README.md                    ← You are here
├── CONTRIBUTING.md              ← Contribution guide
├── SKILL_TEMPLATE.md            ← Skill scaffold
├── PROJECT_ANALYSIS.md          ← Strategic roadmap
├── RESEARCH_QUESTIONNAIRE.md    ← Coverage checklist
│
├── 01-model-architecture/       (5 skills ✓ - Megatron, LitGPT, Mamba, RWKV, NanoGPT)
├── 02-tokenization/             (planned)
├── 03-fine-tuning/              (4 skills ✓ - Axolotl, TRL, LLaMA-Factory, Unsloth)
├── 04-peft/                     (planned)
├── 05-data-processing/          (planned)
├── 06-post-training/            (3 skills ✓ - GRPO, OpenRLHF, SimPO)
├── 07-safety-alignment/         (3 skills ✓ - Constitutional AI, LlamaGuard, NeMo Guardrails)
├── 08-distributed-training/     (2 skills ✓ - DeepSpeed, FSDP)
├── 09-infrastructure/           (3 skills ✓ - Accelerate, Lightning, Ray Train)
├── 10-optimization/             (2 skills ✓ - Flash Attention, bitsandbytes)
├── 11-evaluation/               (1 skill ✓ - lm-evaluation-harness)
├── 12-inference-serving/        (1 skill ✓ - vLLM)
├── 13-mlops/                    (planned)
├── 14-agents/                   (planned - CRITICAL)
├── 15-rag/                      (planned - CRITICAL)
├── 16-prompt-engineering/       (planned)
├── 17-observability/            (planned)
├── 18-multimodal/               (planned)
└── 19-emerging-techniques/      (planned)
```

## 🎯 Use Cases

### For Researchers
"I need to fine-tune Llama 3 with custom data"
→ **03-fine-tuning/axolotl/** - YAML configs, 100+ model support

### For ML Engineers
"How do I optimize inference latency?"
→ **12-inference-serving/vllm/** - PagedAttention, batching

### For Students
"I want to learn how transformers work"
→ **01-model-architecture/litgpt/** - Clean implementations

### For Teams
"We need to scale training to 100 GPUs"
→ **08-distributed-training/deepspeed/** - ZeRO stages, 3D parallelism

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

**Note**: Individual skills may reference libraries with different licenses. Please check each project's license before use.

## 🙏 Acknowledgments

Built with:
- **[Claude Code](https://www.claude.com/product/claude-code)** - AI pair programming
- **[Skill Seeker](https://github.com/yusufkaraaslan/Skill_Seekers)** - Automated doc scraping
- **Open Source AI Community** - For amazing tools and docs

Special thanks to:
- EleutherAI, HuggingFace, NVIDIA, Lightning AI, Meta AI, Anthropic
- All researchers who maintain excellent documentation

## 📞 Get Involved

- **Issues**: [Report bugs or request skills](https://github.com/zechenzhangAGI/claude-ai-research-skills/issues)
- **Discussions**: [Ask questions & share ideas](https://github.com/zechenzhangAGI/claude-ai-research-skills/discussions)
- **Star**: Help others discover this project
- **Fork**: Build your own skill collection

## 📈 Recent Updates

**November 6, 2025** - v0.2.0
- Added 10 skills from GitHub (Megatron-Core, Lightning, Ray Train, etc.)
- Improved skill structure with comprehensive references
- Created strategic roadmap to 70 skills
- Added contribution guidelines

**November 3, 2025** - v0.1.0
- 🎉 Initial release with 5 fine-tuning skills
 
