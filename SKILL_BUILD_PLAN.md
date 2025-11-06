# AI Research Skills Build Plan

Based on deep_research_report_1.md analysis - 25+ skills identified from 100+ tools

---

## Priority Matrix: Documentation Quality + Production Readiness

### Tier 1: VERY HIGH Priority + 5/5 Documentation (13 skills)

**Ready for immediate scraping:**

#### 1. Model Architecture (3 skills)
- **megatron-core** - https://docs.nvidia.com/megatron-core/
- **litgpt** - https://github.com/Lightning-AI/litgpt (comprehensive docs)
- **nanogpt** - https://github.com/karpathy/nanoGPT (educational)

#### 2. Tokenization (1 skill)
- **huggingface-tokenizers** - https://huggingface.co/docs/tokenizers/

#### 3. Fine-Tuning (2 skills)
- **axolotl** - https://docs.axolotl.ai
- **trl** - https://huggingface.co/docs/trl
- **llama-factory** - https://llamafactory.readthedocs.io

#### 4. PEFT (1 skill)
- **huggingface-peft** - https://huggingface.co/docs/peft

#### 5. Data Processing (1 skill)
- **nemo-curator** - https://developer.nvidia.com/nemo-curator

#### 6. Safety & Alignment (2 skills)
- **nemo-guardrails** - https://docs.nvidia.com/nemo/guardrails/
- **perspective-api** - https://perspectiveapi.com/

#### 7. Distributed Training (3 skills)
- **deepspeed** - https://www.deepspeed.ai/
- **pytorch-fsdp** - https://pytorch.org/docs/stable/fsdp.html
- **accelerate** - https://huggingface.co/docs/accelerate

#### 8. Infrastructure (2 skills)
- **pytorch-lightning** - https://lightning.ai/
- **ray-train** - https://www.ray.io/

---

### Tier 2: HIGH Priority + 4-5/5 Documentation (8 skills)

#### 1. Model Architecture (2 skills)
- **rwkv** - https://wiki.rwkv.com/ (4.5/5)
- **gpt-neox** - https://github.com/EleutherAI/gpt-neox (4.5/5)

#### 2. Tokenization (1 skill)
- **tiktoken** - https://github.com/openai/tiktoken (4/5)

#### 3. Fine-Tuning (1 skill)
- **unsloth** - https://docs.unsloth.ai (4/5)

#### 4. Post-Training (2 skills)
- **openrlhf** - https://github.com/OpenRLHF/OpenRLHF (4/5)
- **verl** - https://github.com/volcengine/verl (4/5)

#### 5. Optimization (1 skill)
- **flash-attention** - https://github.com/Dao-AILab/flash-attention (5/5)

---

## Directory Structure

```
claude-ai-research-skills/
├── 1-model-architecture/
│   ├── megatron-core/
│   ├── litgpt/
│   ├── nanogpt/
│   ├── rwkv/
│   └── gpt-neox/
├── 2-tokenization/
│   ├── huggingface-tokenizers/
│   ├── sentencepiece/
│   └── tiktoken/
├── 3-fine-tuning/
│   ├── axolotl/
│   ├── trl/
│   ├── llama-factory/
│   └── unsloth/
├── 4-peft/
│   └── huggingface-peft/
├── 5-data-processing/
│   └── nemo-curator/
├── 6-post-training/
│   ├── trl-alignment/
│   ├── openrlhf/
│   └── verl/
├── 7-safety-alignment/
│   ├── nemo-guardrails/
│   ├── constitutional-ai/
│   └── perspective-api/
├── 8-distributed-training/
│   ├── deepspeed/
│   ├── pytorch-fsdp/
│   ├── megatron-lm/
│   └── accelerate/
├── 9-infrastructure/
│   ├── pytorch-lightning/
│   ├── ray-train/
│   └── composer/
└── 10-optimization/
    ├── flash-attention/
    └── bitsandbytes/
```

---

## Build Sequence

### Phase 1: Fine-Tuning Stack (Most Requested)
1. axolotl
2. trl
3. llama-factory
4. unsloth
5. huggingface-peft

### Phase 2: Distributed Training (Production Critical)
6. deepspeed
7. pytorch-fsdp
8. accelerate
9. megatron-core

### Phase 3: Infrastructure
10. pytorch-lightning
11. ray-train

### Phase 4: Safety & Alignment
12. nemo-guardrails
13. perspective-api

### Phase 5: Architecture & Optimization
14. litgpt
15. flash-attention
16. rwkv
17. gpt-neox

### Phase 6: Specialized
18. nemo-curator
19. openrlhf
20. verl
21. huggingface-tokenizers
22. tiktoken

---

## Skill Seeker MCP Commands

### Generate Config Template
```bash
mcp__skill-seeker__generate_config(
  name="axolotl",
  url="https://docs.axolotl.ai",
  description="Expert guidance for fine-tuning LLMs with Axolotl - YAML configs, LoRA/QLoRA, DPO/GRPO/ORPO support"
)
```

### Estimate Pages (Before Scraping)
```bash
mcp__skill-seeker__estimate_pages(
  config_path="configs/axolotl.json"
)
```

### Scrape Documentation
```bash
mcp__skill-seeker__scrape_docs(
  config_path="configs/axolotl.json"
)
```

### Package Skill
```bash
mcp__skill-seeker__package_skill(
  skill_dir="output/axolotl/"
)
```

---

## Quality Assurance Checklist

For each skill:
- [ ] Config generated with correct selectors
- [ ] Page count estimated (target: 50-500 pages)
- [ ] Documentation scraped successfully
- [ ] SKILL.md generated with examples
- [ ] References organized by category
- [ ] Code examples extracted
- [ ] Packaged as .zip
- [ ] Moved to appropriate directory

---

## Success Metrics

**Target**: 21 skills built in Phase 1-6
**Timeline**: ~2-4 hours of scraping time (parallel execution)
**Expected Size**: 50-500 pages per skill
**Doc Quality**: All 4-5/5 rated sources

---

## Next Steps

1. Create directory structure
2. Generate configs for Tier 1 (13 skills)
3. Run parallel scraping (5-10 at once)
4. Move completed skills to organized directories
5. Create master index/README
6. Generate configs for Tier 2 (8 skills)
7. Repeat scraping and organization

---

**Status**: Ready to execute
**Last Updated**: November 2025
