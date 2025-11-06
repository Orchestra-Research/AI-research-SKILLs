# AI Research Skills Discovery Questionnaire - Part 2

## Deployment & Specialized Applications

**Purpose:** Guide literature research to identify critical topics, libraries, and best practices for specialized models and deployment.

**Instructions for Research Team:**
- Answer each question with specific library names, paper citations, and current best practices
- Prioritize by adoption rate and production readiness
- Include version numbers and last update dates
- Note if a tool/practice is emerging vs. established

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

## Output Format

For each question, provide:

1. **Answer:** Specific libraries/tools/papers with brief descriptions
2. **Priority:** High/Medium/Low (based on adoption and production readiness)
3. **Skill Potential:** Yes/No (should we create a Claude skill for this?)
4. **Documentation Quality:** Rate 1-5 (5 = excellent docs available for scraping)
5. **Notes:** Any additional context (emerging vs. established, alternatives, gotchas)

---

## Example Answer Format

**Q24.1: Which VLM architectures are current?**

| Library/Tool | Description | Priority | Skill Potential | Docs Quality | Notes |
|--------------|-------------|----------|-----------------|--------------|-------|
| LLaVA | Open-source vision-language model | High | Yes | 4/5 | Well-documented, active development |
| OpenFlamingo | Open reproduction of Flamingo | Medium | Yes | 3/5 | Good research use, limited production |
| BLIP-2 | Salesforce vision-language pretraining | High | Yes | 4/5 | Production-ready, HuggingFace integration |

---

**Deadline:** [Specify date]

**Contact:** [Your contact info for questions]

---

*This questionnaire will guide the creation of a comprehensive AI research skill library for Claude Code.*
