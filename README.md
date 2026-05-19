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
├── experiments/
│   ├── run_ablation.py               # Ablation study (all strategies × sizes)
│   ├── run_phase2.py                 # Phase 2 experiment runner
│   ├── run_phase3.py                 # Phase 3 experiment runner
│   ├── plot_ablation.py              # Ablation visualization
│   └── visualize_results.py          # Scaling law visualization
│
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

```bash
cp .env.sample .env
```

```env
EFUNDS_BASE_URL=https://aigc.efunds.com.cn/v1
EFUNDS_API_KEY=your_api_key_here
EFUNDS_USER=SX-your_username
```

The project degrades gracefully without an API key:
- Skill descriptions → template-based fallback
- Answer extraction → regex-based fallback
- Skill quality assessment → default score fallback

---

## Dependencies

```
pandas>=1.5
numpy>=1.24
scikit-learn>=1.2
python-dotenv>=1.0
openai>=1.0
```

Optional: `matplotlib`, `seaborn` (for visualization scripts).

---

## Key Design Decisions

1. **Train/test split by query index** — Same query's 14 traces stay together to prevent cross-configuration leakage
2. **Tool-category-aware consensus** — Full-tool skills max at 8 configs, web skills at 14
3. **Canonical tool names** — `mcp__daloopa__X` normalized to `X` for coverage matching across namespace variants
4. **Recursive JSON parsing in P3.2** — Handles outputs nested as `{"type":"text","text":"{...}"}` in Daloopa API responses
5. **Elitism + tournament selection in P3.3** — Preserves top 20 skills per generation, k=3 tournament for parent selection
6. **Backward-compatible evolution** — `SkillClawEvolver` (V1 merge) kept for Phase 1/2, `EvolutionarySkillClaw` for Phase 3

---

## License

Internal project @ 易方达 (E-Fund). For return offer presentation purposes.
