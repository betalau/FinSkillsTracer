"""
FinSkillsTracer v5.2 — Streamlit App
====================================
Semantic Skill Router + MCP-Style Step-by-Step Execution.

Usage:
    cd FinSkillsTracer
    streamlit run web/app_5.2.py
    # Opens http://localhost:8501
"""

import sys
import os
import json
import time
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from src.models import canonical_tool_name, get_tool_category
from src.llm_client import get_client, LLMClient

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FinSkillsTracer v5.2 — MCP Execution",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_skills() -> List[Dict]:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "skills_bank_v2.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

@st.cache_data
def load_metrics() -> Dict:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for fname in ["phase3_results.json", "phase2_results.json"]:
        path = os.path.join(base, fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}

# ---------------------------------------------------------------------------
# Tool Registry (matching server_5.2.py)
# ---------------------------------------------------------------------------
TOOL_REGISTRY = {
    "discover_companies": {
        "category": "financial_direct", "typical_ms": 520,
        "desc": "Discover company IDs by keyword search",
        "input_schema": {"keywords": "string"},
        "output_schema": {"company_id": "int", "company_name": "str", "ticker": "str"},
    },
    "discover_company_series": {
        "category": "financial_direct", "typical_ms": 636,
        "desc": "Find available financial data series for a company",
        "input_schema": {"company_id": "int", "keywords": "string", "periods": "list[str]"},
        "output_schema": {"series_ids": "list[str]", "series_names": "list[str]"},
    },
    "get_company_fundamentals": {
        "category": "financial_direct", "typical_ms": 979,
        "desc": "Retrieve fundamental financial data",
        "input_schema": {"company_id": "int", "series_ids": "list[str]", "periods": "list[str]"},
        "output_schema": {"fundamentals": "dict[str, float|str]"},
    },
    "WebSearch": {
        "category": "web_search", "typical_ms": 3000,
        "desc": "General web search for financial information",
        "input_schema": {"query": "string", "type": "string"},
        "output_schema": {"results": "list[dict]"},
    },
    "google_search": {
        "category": "web_search", "typical_ms": 3000,
        "desc": "Google search for financial information",
        "input_schema": {"query": "string"},
        "output_schema": {"results": "list[dict]"},
    },
    "WebFetch": {
        "category": "web_fetch", "typical_ms": 2000,
        "desc": "Fetch and parse web page content",
        "input_schema": {"url": "string"},
        "output_schema": {"content": "str"},
    },
}

# ---------------------------------------------------------------------------
# Semantic Router (same logic as server_5.2, adapted for Streamlit)
# ---------------------------------------------------------------------------
def semantic_route_skills(query: str, skills: List[Dict], limit: int = 12, llm_client=None) -> List[Dict]:
    """Two-phase semantic routing: keyword pre-filter + LLM re-ranking."""
    if llm_client is None:
        llm_client = get_client()
    if not skills:
        return []

    q_lower = query.lower()
    q_words = set(q_lower.split())

    financial_terms = {
        'revenue', 'eps', 'earnings', 'income', 'growth', 'margin', 'ebitda',
        'fundamental', 'company', 'series', 'discover', 'financial', 'stock',
        'ticker', 'quarter', 'fiscal', 'annual', 'report', 'profit', 'net',
    }
    finance_bonus = len(q_words & financial_terms) * 0.3

    broad = []
    for skill in skills:
        seq = skill.get("trigger_pattern", {}).get("tool_sequence", [])
        search_text = f"{skill.get('name', '')} {skill.get('description', '')} {' '.join(seq)}".lower()
        text_words = set(search_text.split())
        overlap = len(q_words & text_words)
        score = overlap + finance_bonus
        if score > 0.05:
            broad.append((score, skill))

    broad.sort(key=lambda x: x[0], reverse=True)
    broad = broad[:40]

    # Always attempt LLM re-ranking when >1 candidate
    if len(broad) <= 1:
        return [_skill_to_candidate(s, i) for i, (_, s) in enumerate(broad)]

    if not llm_client.available:
        return [_skill_to_candidate(s, i) for i, (_, s) in enumerate(broad[:limit])]

    # Send enough candidates to LLM for meaningful ranking
    top_for_llm = broad[:min(len(broad), max(limit * 3, 15))]

    candidates_text_parts = []
    for llm_idx, (_, skill) in enumerate(top_for_llm):
        seq = skill.get("trigger_pattern", {}).get("tool_sequence", [])
        name = skill.get("name", "Unknown")
        desc = skill.get("description", "")[:120]
        candidates_text_parts.append(
            f"[{llm_idx}] {name}\n    Tools: {' → '.join(seq)}\n    Desc: {desc}"
        )
    candidates_text = "\n".join(candidates_text_parts)

    prompt = f"""You are a financial AI skill router. Rank ALL of these candidate skills by relevance to the query.

Query: {query}

Candidates:
{candidates_text}

Prefer multi-step pipelines (Discover → Series → Fundamentals) over single-tool skills.

Return a JSON array of ranked indices (most relevant first, include ALL indices):
{{"ranking": [3, 0, 7, ...], "reasoning": "one sentence explaining top pick"}}
Return ONLY valid JSON."""

    try:
        result = llm_client.chat_json(prompt, system="You are a financial AI skill router. Output only valid JSON.", temperature=0.1)
    except Exception:
        result = None

    if result and "ranking" in result:
        ranking = result["ranking"]
        reasoning = result.get("reasoning", "")

        candidates = []
        for rank_pos, llm_index in enumerate(ranking):
            if llm_index < len(top_for_llm) and len(candidates) < limit:
                _, skill = top_for_llm[llm_index]
                c = _skill_to_candidate(skill, len(candidates))
                c["semantic_rank"] = rank_pos + 1
                c["llm_reasoning"] = reasoning[:200] if rank_pos == 0 else ""
                candidates.append(c)

        ranked_ids = {c["skill_id"] for c in candidates}
        for _, skill in top_for_llm:
            if len(candidates) >= limit:
                break
            if skill.get("skill_id") not in ranked_ids:
                c = _skill_to_candidate(skill, len(candidates))
                c["semantic_rank"] = len(candidates) + 1
                candidates.append(c)

        return candidates

    return [_skill_to_candidate(s, i) for i, (_, s) in enumerate(broad[:limit])]


def _skill_to_candidate(skill: Dict, rank: int) -> Dict:
    seq = skill.get("trigger_pattern", {}).get("tool_sequence", [])
    canonical_seq = [canonical_tool_name(t) for t in seq]

    tools_detail = []
    total_ms = 0
    for t in canonical_seq:
        info = TOOL_REGISTRY.get(t, {})
        ms = info.get("typical_ms", 500)
        total_ms += ms
        tools_detail.append({"tool": t, "category": info.get("category", "unknown"), "estimated_ms": ms})

    consensus = skill.get("consensus", {})
    cons_score = consensus.get("consensus_score", 0) if isinstance(consensus, dict) else 0

    return {
        "rank": rank + 1,
        "skill_id": skill.get("skill_id", ""),
        "name": skill.get("name", "Unknown"),
        "tool_sequence": canonical_seq,
        "tools_detail": tools_detail,
        "support": skill.get("support", 0),
        "consensus_score": round(cons_score, 4),
        "estimated_total_ms": total_ms,
        "estimated_tool_calls": len(canonical_seq),
        "description": skill.get("description", "")[:150],
    }


# ---------------------------------------------------------------------------
# MCP Tool Executor (simulated)
# ---------------------------------------------------------------------------
class MCPExecutor:
    def __init__(self, query: str, llm_client=None):
        self.query = query
        self.context: List[Dict] = []
        self.client = llm_client or get_client()

    def execute(self, tool_sequence: List[str], progress_placeholder, ctx_placeholder):
        """Execute tool sequence with step-by-step progress display."""
        # Entity extraction
        with progress_placeholder.container():
            st.markdown("**Phase 1: Entity Extraction**")
            entities = self._extract_entities()
            st.json(entities)

        # Tool execution
        total = len(tool_sequence)
        for i, tool_name in enumerate(tool_sequence):
            canonical = canonical_tool_name(tool_name)
            tool_info = TOOL_REGISTRY.get(canonical, {})

            with progress_placeholder.container():
                st.markdown(f"**Step {i+1}/{total}: `{canonical}`**")
                st.caption(f"{tool_info.get('desc', '')} | ~{tool_info.get('typical_ms', '?')}ms")

                tool_input = self._build_input(canonical, entities)

                col_in, col_out = st.columns(2)
                with col_in:
                    st.markdown("*Input:*")
                    st.json(tool_input)
                with col_out:
                    st.markdown("*Output:*")
                    output_placeholder = st.empty()

            start = time.time()
            output = self._execute_tool(canonical, tool_input)
            elapsed_ms = int((time.time() - start) * 1000)

            with output_placeholder:
                st.json(output)
                st.caption(f"Completed in {elapsed_ms}ms")

            self.context.append({
                "tool": canonical, "input": tool_input, "output": output, "duration_ms": elapsed_ms,
            })

            # Update context chain
            with ctx_placeholder.container():
                st.markdown("**Parameter Chain**")
                for j, ctx in enumerate(self.context):
                    out_keys = [k for k in ctx.get("output", {}).keys()
                                if k not in ("error", "message", "status", "currency", "fiscal_year_end")][:3]
                    st.markdown(
                        f"`{ctx['tool']}` → " +
                        ", ".join(f"`{k}`: {str(ctx['output'].get(k, ''))[:40]}" for k in out_keys)
                    )

            time.sleep(0.3)

        # Answer extraction
        with progress_placeholder.container():
            st.markdown("**Final: Answer Extraction**")
            answer, confidence = self._extract_answer()
            st.success(f"**Answer:** {answer}")
            st.metric("Confidence", f"{confidence:.0%}")

        return answer, confidence

    def _extract_entities(self) -> Dict:
        prompt = f"""Extract financial entities from this query. Return JSON.
Query: {self.query}
Return: {{"companies": [{{"name": "...", "ticker": "..."}}], "metrics": [...], "periods": [...], "query_type": "..."}}
Return ONLY valid JSON."""
        result = self.client.chat_json(prompt, system="You are a financial data extraction tool. Output only valid JSON.", temperature=0)
        return result if result else self._regex_extract_entities()

    def _regex_extract_entities(self) -> Dict:
        q = self.query.lower()
        known = {"apple": ("Apple Inc.", "AAPL"), "tesla": ("Tesla Inc.", "TSLA"),
                 "microsoft": ("Microsoft Corp.", "MSFT"), "nvidia": ("NVIDIA Corp.", "NVDA")}
        companies = []
        for key, (name, ticker) in known.items():
            if key in q:
                companies.append({"name": name, "ticker": ticker})
        metrics = [m for m in ["revenue", "eps", "net income", "growth", "margin"] if m in q]
        periods = [p for p in ["FY2024", "FY2023", "Q1", "Q2", "Q3", "Q4"] if p.lower() in q]
        return {"companies": companies, "metrics": metrics, "periods": periods, "query_type": "single_company"}

    def _build_input(self, tool_name: str, entities: Dict) -> Dict:
        company = (entities.get("companies") or [{}])[0]
        for ctx in self.context:
            out = ctx.get("output", {})
            if isinstance(out, dict) and "company_id" in out:
                company["id"] = out["company_id"]
        series_ids = None
        for ctx in self.context:
            if isinstance(ctx.get("output", {}), dict) and "series_ids" in ctx["output"]:
                series_ids = ctx["output"]["series_ids"]

        if "discover_companies" in tool_name:
            return {"keywords": company.get("name", self.query[:80])}
        elif "discover_company_series" in tool_name:
            return {"company_id": company.get("id", 320193), "keywords": ", ".join(entities.get("metrics", ["financials"])), "periods": entities.get("periods", ["FY2024"])}
        elif "get_company_fundamentals" in tool_name:
            return {"company_id": company.get("id", 320193), "series_ids": series_ids or ["RETAIL_REV", "NET_INC_Q"], "periods": entities.get("periods", ["FY2024"])}
        elif tool_name in ("WebSearch", "google_search"):
            return {"query": self.query[:200], "type": "financial"}
        elif tool_name == "WebFetch":
            return {"url": f"https://finance.example.com/company/{company.get('ticker', 'AAPL')}"}
        return {"query": self.query[:200]}

    def _execute_tool(self, tool_name: str, tool_input: Dict) -> Dict:
        tool_info = TOOL_REGISTRY.get(tool_name, {})
        prior = "\n".join(
            f"[{c['tool']}] → {json.dumps(c.get('output', {}), default=str)[:200]}"
            for c in self.context[-3:]
        )
        system = f"""You are the MCP tool '{tool_name}'.
Purpose: {tool_info.get('desc', '')}
Output schema: {json.dumps(tool_info.get('output_schema', {}))}
Return ONLY valid JSON matching the schema. No commentary."""
        prompt = f"""Execute this MCP tool call:
Tool: {tool_name}
Input: {json.dumps(tool_input)}
Query: {self.query}
Prior context: {prior or '(none)'}
Return ONLY the JSON output."""
        try:
            result = self.client.chat_json(prompt, system=system, temperature=0.1)
            if result:
                return result
        except Exception as e:
            st.warning(f"LLM tool call failed: {e}")
        return self._mock_output(tool_name, tool_input)

    def _mock_output(self, tool_name: str, tool_input: Dict) -> Dict:
        kw = str(tool_input.get("keywords", tool_input.get("query", ""))).lower()
        if "discover_companies" in tool_name:
            m = {"apple": (320193, "Apple Inc.", "AAPL"), "nvidia": (219867, "NVIDIA Corp.", "NVDA"),
                 "tesla": (219863, "Tesla Inc.", "TSLA"), "microsoft": (219864, "Microsoft Corp.", "MSFT")}
            for k, v in m.items():
                if k in kw:
                    return {"company_id": v[0], "company_name": v[1], "ticker": v[2]}
            return {"company_id": 320193, "company_name": "Apple Inc.", "ticker": "AAPL"}
        elif "discover_company_series" in tool_name:
            return {"series_ids": ["RETAIL_REV", "NET_INC_Q", "EPS_DILUTED"], "series_names": ["Revenue", "Net Income", "EPS"], "periods_available": ["FY2020-FY2024"]}
        elif "get_company_fundamentals" in tool_name:
            return {"fundamentals": {"Revenue_FY2024": 391035000000, "NetIncome_FY2024": 93736000000, "EPS_FY2024": 6.43}, "currency": "USD"}
        elif tool_name in ("WebSearch", "google_search"):
            return {"results": [{"title": "Financial Results", "snippet": "Revenue: $391.04B", "url": "https://example.com"}]}
        elif tool_name == "WebFetch":
            return {"content": "Revenue: $391.04B. Net Income: $93.74B.", "status_code": 200}
        return {"result": "ok"}

    def _extract_answer(self):
        if not self.client.available:
            return "N/A", 0.0
        outputs_text = "\n".join(
            f"[{c['tool']}]: {json.dumps(c.get('output', {}), default=str)[:300]}"
            for c in self.context
        )
        prompt = f"""Extract the final answer from these MCP tool outputs.
Query: {self.query}
Outputs: {outputs_text}
Return JSON: {{"answer": "...", "confidence": 0.0_to_1.0, "explanation": "brief"}}
Return ONLY valid JSON."""
        result = self.client.chat_json(prompt, system="You are a financial data analyst. Output only valid JSON.", temperature=0)
        if result and "answer" in result:
            return result.get("answer", "N/A"), float(result.get("confidence", 0.5))
        return "N/A", 0.0


# ===========================================================================
# Streamlit UI
# ===========================================================================
st.title("FinSkillsTracer v5.2")
st.caption("Semantic Skill Router + MCP Step-by-Step Execution")

# ---- Sidebar ----
with st.sidebar:
    st.header("Configuration")

    # Provider selector — use session_state to persist across reruns
    if "v52_provider" not in st.session_state:
        st.session_state.v52_provider = get_client().provider

    available = LLMClient.detect_available()
    current = st.session_state.v52_provider
    provider = st.selectbox(
        "LLM Provider", available,
        index=available.index(current) if current in available else 0,
        key="provider_select"
    )
    if provider != current:
        st.session_state.v52_provider = provider
        client = LLMClient(provider=provider)
        st.success(f"Switched to {client.label} ({client.model})")
        st.rerun()

    # Skills info
    skills = load_skills()
    st.metric("Skills Loaded", len(skills))
    metrics_data = load_metrics()
    if metrics_data:
        qm = metrics_data.get("quality_metrics", {})
        if qm:
            st.metric("Coverage", f"{qm.get('coverage', 0):.2%}")
            st.metric("Uniqueness", f"{qm.get('uniqueness', 0):.2%}")

    st.divider()
    st.caption("v5.2 Features:")
    st.caption("- Two-phase LLM semantic routing")
    st.caption("- Step-by-step MCP tool execution")
    st.caption("- Parameter chaining across steps")
    st.caption("- Real-time execution streaming")

# ---- Main Area ----
# Query input
if "v52_query" not in st.session_state:
    st.session_state.v52_query = "What was Apple Inc.'s revenue and net income for fiscal year 2024?"

col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input(
        "Financial Query",
        value=st.session_state.v52_query,
        placeholder="Enter a financial query...",
        label_visibility="collapsed",
        key="query_input",
    )
    st.session_state.v52_query = query
with col2:
    run_btn = st.button("▶ Run MCP Analysis", type="primary", use_container_width=True)

# Presets
presets = {
    "NVIDIA FY2024": "What was NVIDIA's revenue growth rate and earnings per share for fiscal year 2024?",
    "Apple FY2024": "What was Apple Inc.'s revenue and net income for fiscal year 2024?",
    "Tesla Q3/Q4": "What was Tesla's earnings per share for Q3 and Q4 of fiscal year 2024?",
    "FAANG Compare": "Compare revenue growth rates of Apple, Microsoft, Google, Amazon and Meta for FY2023-FY2024",
}
preset_cols = st.columns(len(presets))
for i, (label, q) in enumerate(presets.items()):
    with preset_cols[i]:
        if st.button(label, key=f"preset_{i}", use_container_width=True):
            st.session_state.v52_query = q
            st.rerun()

if not run_btn and "v52_ran" not in st.session_state:
    st.info("Enter a financial query above and click **Run MCP Analysis** to start.")
    st.markdown("---")
    st.markdown("### How It Works")
    st.markdown("""
    1. **Semantic Router** — Keyword pre-filter (top 40) → LLM re-ranking for semantic relevance
    2. **Entity Extraction** — LLM extracts companies, metrics, periods from query
    3. **MCP Execution** — Each tool runs independently with structured I/O schemas
    4. **Parameter Chaining** — Output from step N feeds into step N+1 input
    5. **Answer Extraction** — LLM extracts final answer from all accumulated context
    """)
    st.stop()

if not run_btn:
    st.stop()

st.session_state["v52_ran"] = True

# ---- Execute ----
st.divider()

# Phase 1: Semantic Router
st.subheader("🎯 Semantic Skill Router")
active_client = LLMClient(provider=st.session_state.v52_provider)
with st.spinner("Routing query to skills (keyword pre-filter + LLM re-ranking)..."):
    candidates = semantic_route_skills(query, skills, limit=12, llm_client=active_client)

if candidates:
    best = candidates[0]
    st.success(f"Best route: **{best['name']}** — `{' → '.join(best['tool_sequence'])}`")

    # Show candidates table
    df = pd.DataFrame([{
        "Rank": c["rank"],
        "Skill": c["name"][:40],
        "Tool Sequence": " → ".join(c["tool_sequence"]),
        "Est. ms": c["estimated_total_ms"],
        "Calls": c["estimated_tool_calls"],
        "Support": c.get("support", 0),
    } for c in candidates[:8]])
    st.dataframe(df, use_container_width=True, hide_index=True)

    tool_sequence = best["tool_sequence"]
else:
    st.warning("No matching skills found. Using default WebSearch.")
    tool_sequence = ["WebSearch"]

# Phase 2: MCP Execution
st.divider()
st.subheader("🔧 MCP Step-by-Step Execution")

progress_placeholder = st.empty()
ctx_placeholder = st.empty()

executor = MCPExecutor(query, llm_client=active_client)
answer, confidence = executor.execute(tool_sequence, progress_placeholder, ctx_placeholder)

# Final result
st.divider()
st.subheader("📋 Final Result")
col_a, col_c = st.columns([3, 1])
with col_a:
    st.markdown(f"### {answer}")
with col_c:
    st.metric("Confidence", f"{confidence:.0%}")

st.caption(f"Execution completed with {len(tool_sequence)} MCP tool calls across {len(executor.context)} steps.")
