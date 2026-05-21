# FinSkillsTracer: Trace-to-Skills Mining for Financial AI Agents

Automatic mining of reusable **Agent Skills** from financial retrieval traces. Given a dataset of 7,000 tool-call traces across 14 LLM configurations answering 500 financial queries, FinSkillsTracer discovers, parameterizes, evaluates, and evolves reusable skills that make AI agents faster and more accurate.

**Internship project @ 易方达 (E-Fund) — Return Offer Presentation 2026.**

---

## Overview

Modern AI agents rely on tool-calling patterns to answer financial questions. However, hand-crafting agent skills is expensive and brittle. FinSkillsTracer automates this process:

1. **Mine** frequent, semantically meaningful tool sequences from execution traces
2. **Consensus-validate** skills across multiple LLM configurations
3. **Parameterize** skills with precise data-flow templates
4. **Evaluate** skills on held-out queries (coverage, uniqueness, consensus)
5. **Benchmark** skill-guided agents vs baselines (latency, tool calls, error rate)
6. **Evolve** skills through mutation, crossover, and fitness-based selection

The optimization is organized in three phases, each improving a different aspect of the pipeline.

---

## Live Demos

### v5.3 — Real Financial APIs + Semantic Skill Router (Latest)

**Next-generation financial demo with real API tools.** Replaces v5.2's simulated MCP with actual Alpha Vantage and SerpAPI calls. Streams live financial data step-by-step via SSE.

**Key Features:**
- **12 Real Financial API Tools**: 7 Alpha Vantage (fundamentals, quotes, economic data) + 2 SerpAPI (web/news search) + web fetch + company comparison
- **Real Data Pipeline**: Company overview → income statement → stock quote, all from live APIs
- **Semantic Skill Router**: LLM-powered two-phase routing (keyword pre-filter → semantic re-ranking), retained from v5.2
- **Smart Pipeline Builder**: Intent-based routing — LLM classifier → regex keyword fallback → skill bank mapping
- **TTL Caching + Rate Limiting**: In-memory cache (60s-3600s TTL), shared-bucket rate limiter for Alpha Vantage free tier (5 calls/min)
- **SSE Streaming**: Step-by-step execution with live/cached/error data source indicators
- **EFund Institutional Blue Design**: Glassmorphism nav, metric cards, gold accents, animated pipeline nodes
- **Multi-Provider LLM**: DeepSeek / EFund / OpenAI / Custom switchable via button group
- **i18n**: Full English/中文 toggle
- **Model Switching**: Runtime provider switching via POST /api/llm/switch

#### Starting v5.3

```bash
# Install dependencies
pip install flask flask-cors python-dotenv openai requests

# Configure API keys in .env
# Required: LLM_PROVIDER, DEEPSEEK_API_KEY (or EFund/OpenAI)
# Optional: ALPHAVANTAGE_API_KEY, SERPAPI_KEY (for real financial data)

# Start the server (port 5003)
python web/server_5.3.py
# → http://localhost:5003/api/health

# Open the frontend
start web/v5.3.html      # Windows
open web/v5.3.html        # macOS

# Or use Streamlit version
streamlit run web/app_5.3.py
# → http://localhost:8501
```

#### v5.3 Real Financial Tools

| Tool | API Source | Endpoint | Cache TTL | Description |
|------|-----------|----------|-----------|-------------|
| `get_company_overview` | Alpha Vantage | OVERVIEW | 3600s | Company profile, sector, market cap, P/E, EPS, description |
| `get_stock_quote` | Alpha Vantage | GLOBAL_QUOTE | 60s | Real-time price, change %, volume, day range |
| `get_income_statement` | Alpha Vantage | INCOME_STATEMENT | 3600s | Annual revenue, gross profit, operating income, net income |
| `get_balance_sheet` | Alpha Vantage | BALANCE_SHEET | 3600s | Total assets, liabilities, equity, debt ratios |
| `get_cash_flow` | Alpha Vantage | CASH_FLOW | 3600s | Operating, investing, financing cash flows |
| `get_earnings` | Alpha Vantage | EARNINGS | 1800s | Annual/quarterly EPS, estimates, surprises |
| `get_market_data` | Alpha Vantage | TREASURY_YIELD/CPI/etc | 1800s | Economic indicators: yields, CPI, unemployment, GDP |
| `get_fx_rate` | Alpha Vantage | CURRENCY_EXCHANGE_RATE | 300s | Real-time FX rate between any two currencies |
| `web_search` | SerpAPI | Google Search | 300s | Organic search results: titles, snippets, links |
| `search_news` | SerpAPI | Google News | 300s | Financial news articles with source + date |
| `web_fetch` | requests | Any URL | 600s | Fetch + extract text content from web pages |
| `compare_companies` | Alpha Vantage | Multi-call OVERVIEW+QUOTE | 3600s | Side-by-side comparison of up to 5 companies |

#### v5.3 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Server status, API key validation, rate limits, cache stats |
| `/api/skills/route` | POST | Semantic skill router — LLM-based relevance ranking |
| `/api/trace/execute?query=...` | GET | **SSE stream** — real-time tool execution pipeline |
| `/api/llm/providers` | GET | List available LLM providers + current selection |
| `/api/llm/switch` | POST | Switch LLM provider at runtime |
| `/api/tools` | GET | List all 12 real tools with metadata + rate limits |
| `/api/tools/<tool_id>/test` | GET | Quick test endpoint for a single tool |
| `/api/skills` | GET/POST | Search or list skills bank |
| `/api/skills/<id>` | GET | Skill detail by ID |
| `/api/metrics` | GET | Phase 3 evaluation metrics |

#### v5.3 Example API Calls

```bash
# Health check
curl http://localhost:5003/api/health
# → {"status":"ok","version":"5.3","tools_available":12,"api_keys":{"alphavantage":true,...}}

# Semantic route
curl -X POST http://localhost:5003/api/skills/route \
  -H "Content-Type: application/json" \
  -d '{"query":"compare Tesla and Ford financial performance","limit":5}'

# SSE execution (streaming)
curl "http://localhost:5003/api/trace/execute?query=What+is+Apple+revenue+and+PE+ratio"
# → SSE events: meta → entities → tool_start/tool_done × 3 → answer → done

# Switch model
curl -X POST http://localhost:5003/api/llm/switch \
  -H "Content-Type: application/json" \
  -d '{"provider":"deepseek"}'
```

#### v5.3 Architecture

```
User Query → Semantic Router (keyword pre-filter → LLM re-ranking)
           → Smart Pipeline Builder (LLM intent → regex → skill bank)
           → Entity Extraction (LLM + COMPANY_TICKER_MAP ~250 entries)
           → RealToolExecutor
              ├── CACHE CHECK (TTL-based, per-tool config)
              ├── RATE LIMIT CHECK (shared bucket per api_source)
              └── API CALL (Alpha Vantage / SerpAPI / requests)
           → Answer Synthesis (LLM from accumulated real data)
           → SSE Stream to Frontend
```

---

### v4.0 — Skill Router + Streaming + Comparison (Legacy)

---

## Dataset

**FinRetrieval** (from HuggingFace `daloopa/finretrieval`):
- **7,000 traces** (500 queries × 14 model configurations)
- **14 model configs**: 8 full-tool (gemini3pro, gpt5.2, opus4.5, sonnet4.5 + reasoning variants) + 6 web-only
- **17 tool types**: Daloopa financial API (discover_companies, discover_company_series, get_company_fundamentals + MCP variants), WebSearch, google_search, WebFetch, Read, Grep, Glob, Bash, Task

Download:
```bash
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
for f in ['tool_traces.parquet', 'scores.parquet', 'questions.parquet']:
    hf_hub_download('daloopa/finretrieval', f, repo_type='dataset', local_dir='data')
"
```

---

## Quick Start

### 1. Setup

```bash
# Clone
git clone https://github.com/betalau/FinSkillsTracer.git
cd FinSkillsTracer

# Install dependencies
pip install pandas numpy scikit-learn python-dotenv openai

# Download dataset (see above)

# Configure API key (optional — falls back to templates/regex if not set)
cp .env.sample .env
# Edit .env with your EFundGPT credentials
```

### 2. Run Phase 1 — Skill Mining + Evaluation

```bash
cd experiments
python run_ablation.py              # Ablation study (all strategies)
cd ..
python src/main_v2.py               # Full Phase 1 pipeline
```

Outputs: `skills_bank_v2.json`, `evaluation_metrics.json`, `experiments/ablation_results.json`

### 3. Run Phase 2 — Error Mining + Agent Benchmark

```bash
cd experiments
python run_phase2.py
```

Outputs: `../phase2_results.json`

### 4. Run Phase 3 — Deep Optimization

```bash
cd experiments
python run_phase3.py
```

Outputs: `../phase3_results.json`

---

## Architecture

```
FinSkillsTracer/
├── src/
│   ├── models.py                     # Skill, HierarchicalSkill, ParameterizedTemplate
│   ├── main_v2.py                    # Phase 1 enhanced pipeline entry point
│   │
│   ├── ingestion/
│   │   └── normalization.py          # Data ingestion (parquet → normalized traces)
│   │
│   ├── mining/
│   │   ├── fsp.py                    # Strategy A: Frequent Sequence Pattern mining
│   │   ├── semantic.py               # Strategy B: Semantic clustering + LLM descriptions
│   │   ├── hierarchical.py           # Strategy C / P3.1: Tree-structured boundary detection
│   │   ├── trace2skill.py            # Strategy D: Trace-to-skill extraction
│   │   ├── skillclaw.py              # Strategy E / P3.3: Evolutionary optimization
│   │   ├── consensus.py              # P1.1: Cross-model consensus scoring
│   │   └── error_miner.py            # P2.1: Error pattern mining + real confidence
│   │
│   ├── parameterization/
│   │   └── generalization.py         # P3.2: Parameter type inference + data flow
│   │
│   ├── evaluation/
│   │   ├── llm_evaluator.py          # P1.2/P1.3: SkillEvaluatorV2 + LLM judge
│   │   └── utility_evaluator.py      # P2.2: Agent-vs-Agent benchmark runner
│   │
│   └── agents/
│       └── agent_simulator.py        # P2.2/P2.3: SkillAwareAgent, ReActAgent, RandomAgent
│
├── web/                              # Live demo frontend + backend
│   ├── v5.3.html                     # V5.3: Real APIs + SSE + EFund blue design
│   ├── server_5.3.py                 # V5.3: Flask API (12 real tools + SSE streaming)
│   ├── app_5.3.py                    # V5.3: Streamlit app (real APIs + model switching)
│   ├── v4.html                       # V4.0: Skill Router + Streaming + 3-Column Compare
│   ├── v2.html                       # V2.0: Enterprise presentation edition
│   ├── index.html                    # V1.0: Dark theme presentation
│   ├── server.py                     # V4.0 Flask API server
│   └── app.py                        # Streamlit v3.0 demo
│
├── experiments/
│   ├── run_ablation.py               # Ablation study (all strategies × sizes)
│   ├── run_phase2.py                 # Phase 2 experiment runner
│   ├── run_phase3.py                 # Phase 3 experiment runner
│   ├── plot_ablation.py              # Ablation visualization
│   └── visualize_results.py          # Scaling law visualization
│
├── skills_bank_v2.json               # 225 mined skills
├── phase3_results.json               # Phase 3 evaluation metrics
├── .env.sample                       # API key template
├── .gitignore
└── README.md
```

---

## Phase Details

### Phase 1: Multi-Model Consensus Mining

| Component | Description | Key Metric |
|-----------|-------------|------------|
| **P1.1** Consensus Mining | Tool-category-aware cross-model agreement scoring | Consensus score (0-1) |
| **P1.2** Train/Test Split | Coverage evaluated on held-out queries (no leakage) | Coverage(test) |
| **P1.3** LLM Descriptions | EFundGPT-generated skill names and descriptions | Semantic quality |

**Mining Strategies:**
- **A (FSP)**: Frequent sub-sequence mining with min_support threshold
- **B (Semantic)**: K-means clustering of tool sequences + LLM description generation
- **C (Hierarchical)**: Recursive boundary detection (category transitions, duration spikes, I/O chains)
- **D (Trace2Skill)**: Direct skill extraction from annotated trace segments
- **E (SkillClaw)**: Merge identical sequences + filter by consensus score and support

### Phase 2: Real Utility Evaluation

| Component | Description | Key Metric |
|-----------|-------------|------------|
| **P2.1** Error Mining | Tool-level vs answer-level error separation on all 7,000 traces | Real confidence |
| **P2.2** Agent Benchmark | SkillAwareAgent vs ReActAgent vs RandomAgent on 50 held-out queries | Latency, tool calls, error rate |
| **P2.3** LLM Skill Quality | EFundGPT scores skills on Usefulness/Completeness/Generality | LLM judge scores (1-10) |

### Phase 3: Deep Optimization

| Component | Description | Key Metric |
|-----------|-------------|------------|
| **P3.1** Tree Hierarchy | 5-category taxonomy, 3 depth levels, parent-child links via subsequence matching | Tree depth, categories |
| **P3.2** Parameterization | Cross-instance voting, JSON-parsed data flow, variable/constant/derived classification | Data flow edges |
| **P3.3** Real Evolution | 5 mutation ops, crossover, tournament selection, elitism, multi-round iteration | Skill reduction, consensus gain |

---

## Current Results (May 2026)

### Skill Quality Progression

| Metric | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|
| Skills Discovered | 225 | 225 | **181** |
| Coverage (test) | 96.04% | 96.04% | **96.04%** |
| Uniqueness | 99.19% | 98.53% | 98.83% |
| Confidence | 1.00* | 0.5881 | **0.6458** |
| Consensus Mean | 0.2791 | 0.2791 | **0.3088** |
| High Consensus | 66/225 | 66/225 | 7/181 |

> *Phase 1 confidence was a placeholder (golden-only). Phase 2 introduced real error-based confidence from all 7,000 traces. Phase 3 improved real confidence by 9.8% and consensus mean by 10.6% through namespace merging and evolution.

### Agent-vs-Agent Benchmark (50 test queries)

| Agent | Task Success | Avg Tool Calls | Avg Latency | Error Rate |
|-------|-------------|----------------|-------------|------------|
| **SkillAwareAgent** | 74% | 6.0 | 3,251 ms | 0% |
| ReActAgent | 90% | 14.0 | 41,263 ms | 2.9% |
| RandomAgent | 60% | 5.0 | 21,574 ms | 0% |

**Comparative vs ReActAgent:** 92.1% latency reduction, 57.1% tool call reduction, 100% error rate reduction.

### Ablation Study

| Strategy | Skills | Coverage(test) | Uniqueness | Consensus Rate |
|----------|--------|----------------|------------|----------------|
| A: FSP | 177 | 96.04% | 99.19% | 24.9% |
| B: Semantic | 10 | 22.03% | 100% | 10.0% |
| C: Hierarchical (P3.1) | 87 | 27.11% | 96.55% | 0% |
| D: Trace2Skill | 21 | 16.85% | 95.24% | 0% |
| E: SkillClaw (V1) | 206 | 96.04% | 98.53% | 28.2% |
| E: Evolutionary (P3.3) | 181 | 96.04% | 98.83% | 3.9% |

### P3.1 Tree Structure

5 category roots auto-detected via tool-category classification:
- `financial_direct` (27 skills) — Daloopa API without MCP prefix
- `web_search` (47 skills) — WebSearch/google_search loops of varying lengths
- `financial_mcp` (14 skills) — Daloopa MCP variants
- `web_fetch` (2 skills) — WebFetch extraction patterns
- `file_ops` (2 skills) — Read/Grep/Glob/Bash combinations

### P3.2 Data Flow (Daloopa Pipeline, 2,960 instances)

```
discover_companies(keywords) → outputs: {company_id, company_name, ticker}
         │
         └── company_id (76.4% match) → discover_company_series(company_id, keywords, periods)
                    │
                    └── company_id (76.4% match) → get_company_fundamentals(company_id, periods, series_ids)
```

Parameters classified: `company_id` = **derived**, `keywords` = **variable**, `periods` = **variable**.

---

## Environment

Create `.env` from the template:

```env
LLM_PROVIDER=deepseek

# DeepSeek (default for v5.3)
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# EFund (易方达) — enterprise LLM
EFUNDS_BASE_URL=https://aigc.efunds.com.cn/v1
EFUNDS_API_KEY=your_efund_api_key_here
EFUNDS_USER=SX-your_username

# OpenAI (optional)
# OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_API_KEY=your_openai_key

# Real Financial Data APIs (v5.3)
ALPHAVANTAGE_API_KEY=your_alphavantage_key    # Free tier: 5 calls/min
SERPAPI_KEY=your_serpapi_key                  # 100 calls/month free
```

**API Key Notes for v5.3:**
- **LLM key** (DeepSeek/EFund/OpenAI) — required for skill routing + answer synthesis
- **Alpha Vantage** — free tier at [alphavantage.co](https://www.alphavantage.co/support/#api-key), 5 calls/min shared across all tool types
- **SerpAPI** — free tier at [serpapi.com](https://serpapi.com/), 100 searches/month

The server degrades gracefully without financial API keys — tools requiring missing keys return config errors rather than crashing.

---

## Dependencies

**Core pipeline:**
```
pandas>=1.5
numpy>=1.24
scikit-learn>=1.2
python-dotenv>=1.0
openai>=1.0
```

**Web demo (v5.3):**
```
flask>=2.3
flask-cors>=4.0
requests>=2.28
```

**Web demo (v4.0):**
```
flask>=2.3
flask-cors>=4.0
```

**Streamlit demo (v5.3 / v3.0):**
```
streamlit>=1.28
pandas>=1.5
```

**Visualization (optional):**
```
matplotlib, seaborn
```

---

## Key Design Decisions

### Mining Pipeline (v1-v4)
1. **Train/test split by query index** — Same query's 14 traces stay together to prevent cross-configuration leakage
2. **Tool-category-aware consensus** — Full-tool skills max at 8 configs, web skills at 14
3. **Canonical tool names** — `mcp__daloopa__X` normalized to `X` for coverage matching across namespace variants
4. **Recursive JSON parsing in P3.2** — Handles outputs nested as `{"type":"text","text":"{...}"}` in Daloopa API responses
5. **Elitism + tournament selection in P3.3** — Preserves top 20 skills per generation, k=3 tournament for parent selection
6. **Backward-compatible evolution** — `SkillClawEvolver` (V1 merge) kept for Phase 1/2, `EvolutionarySkillClaw` for Phase 3

### v5.3 Real API Demo
7. **Shared rate limiting bucket** — Alpha Vantage free tier (5 calls/min) shared across ALL tool types, not per-tool. SerpAPI and web fetch have separate buckets
8. **Three-tier pipeline routing** — LLM intent classifier → regex keyword fallback → skill bank OLD_TO_NEW mapping (last resort). Prevents old v5.2 simulated tool names from leaking into v5.3 real pipelines
9. **No serpapi package dependency** — Uses `requests.get()` directly to SerpAPI HTTP API, avoiding pip dependency issues
10. **TTL-based cache with rate-limit fallback** — When rate-limited, accepts stale cached data with extended TTL (99999s) rather than failing
11. **Company name → ticker resolution** — Three-tier: built-in COMPANY_TICKER_MAP (~250 entries) → Alpha Vantage SYMBOL_SEARCH → LLM inference
12. **Direct HTTP (no bs4)** — Web fetch uses regex-based HTML text extraction (strip scripts/styles → strip tags → collapse whitespace), avoiding BeautifulSoup dependency

---

## License

Internal project @ 易方达 (E-Fund). For return offer presentation purposes.
