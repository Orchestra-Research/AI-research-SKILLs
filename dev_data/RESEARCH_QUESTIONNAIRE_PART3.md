# AI Research Skills Discovery Questionnaire - Part 3

## Agent & Application Engineering

**Purpose:** Guide literature research to identify critical topics, libraries, and best practices for agent frameworks and LLM application development.

**Instructions for Research Team:**
- Answer each question with specific library names, paper citations, and current best practices
- Prioritize by adoption rate and production readiness
- Include version numbers and last update dates
- Note if a tool/practice is emerging vs. established

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

**Q36.1: Which agent frameworks are production-ready?**

| Library/Tool | Description | Priority | Skill Potential | Docs Quality | Notes |
|--------------|-------------|----------|-----------------|--------------|-------|
| LangChain | Most popular agent framework with extensive tools | High | Yes | 5/5 | Excellent docs, massive ecosystem |
| LlamaIndex | Data-focused agent framework for RAG | High | Yes | 4/5 | Great docs, strong RAG focus |
| CrewAI | Multi-agent collaboration framework | Medium | Yes | 3/5 | Growing, good for role-based agents |

---

**Deadline:** [Specify date]

**Contact:** [Your contact info for questions]

---

*This questionnaire will guide the creation of a comprehensive AI research skill library for Claude Code.*
