"""
FinSkillsTracer v5.3 — Streamlit App
====================================
Real Financial APIs: Alpha Vantage + SerpAPI + Semantic Skill Router.

Usage:
    cd FinSkillsTracer
    streamlit run web/app_5.3.py
    # Opens http://localhost:8501
"""

import sys
import os
import json
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from src.models import canonical_tool_name
from src.llm_client import get_client

# Import real tool infrastructure from server_5.3.py (filename has dot — use importlib)
import importlib.util
_server_53_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server_5.3.py")
_spec = importlib.util.spec_from_file_location("server_5_3", _server_53_path)
_server_53 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_server_53)

REAL_TOOL_REGISTRY = _server_53.REAL_TOOL_REGISTRY
REAL_TOOL_EXECUTORS = _server_53.REAL_TOOL_EXECUTORS
COMPANY_TICKER_MAP = _server_53.COMPANY_TICKER_MAP
API_KEYS = _server_53.API_KEYS
_CACHE = _server_53._CACHE
_RATE_LIMITER = _server_53._RATE_LIMITER
RealToolExecutor = _server_53.RealToolExecutor
semantic_route_skills = _server_53.semantic_route_skills
_build_smart_pipeline = _server_53._build_smart_pipeline

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FinSkillsTracer v5.3 — Real Financial APIs",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Institutional Blue Fintech Theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Override Streamlit defaults with v2.html fintech theme */
    :root {
        --primary: #005096;
        --primary-light: #e8f0f8;
        --accent: #1EB9E1;
        --gold: #C8963E;
        --green: #1a8c5e;
        --red: #C41230;
        --bg: #F4F7F8;
    }
    .stApp { background: #F4F7F8; }
    section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #E2E8EE; }
    section[data-testid="stSidebar"] .stMarkdown h2 { color: #005096; font-weight: 700; }
    .stButton>button {
        background: #005096; color: #fff; border: none; border-radius: 6px;
        font-weight: 600; padding: 8px 20px; transition: all .2s;
    }
    .stButton>button:hover { background: #003a6f; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,80,150,0.2); }
    /* Metric cards */
    .metric-card {
        background: #fff; border: 1px solid #EEF2F5; border-radius: 10px;
        padding: 16px; text-align: center; box-shadow: 0 2px 16px rgba(0,80,150,0.07);
    }
    .metric-card .big-num { font-size: 1.6rem; font-weight: 700; color: #005096; font-family: 'JetBrains Mono', monospace; }
    .metric-card .lbl { font-size: 0.7rem; color: #8a9bb0; text-transform: uppercase; letter-spacing: 0.6px; }
    /* Data source badges */
    .source-badge {
        display: inline-block; padding: 3px 10px; border-radius: 12px;
        font-size: 0.7rem; font-weight: 600; margin: 2px;
    }
    .source-badge.alphavantage { background: #e8f0f8; color: #005096; }
    .source-badge.serpapi { background: #e6f7fc; color: #0d7a96; }
    .source-badge.live { color: #1a8c5e; }
    .source-badge.cached { color: #C8963E; }
    .source-badge.error { color: #C41230; }
    /* Tool result cards */
    .tool-card-header { font-family: 'JetBrains Mono', monospace; font-weight: 600; color: #005096; }
</style>
""", unsafe_allow_html=True)

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

# ---------------------------------------------------------------------------
# Session State Init
# ---------------------------------------------------------------------------
if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = os.getenv("LLM_PROVIDER", "deepseek")
if "execution_history" not in st.session_state:
    st.session_state.execution_history = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## FinSkillsTracer")
    st.markdown(f"**v5.3** · Real Financial APIs")
    st.markdown("---")

    # Model Switching
    st.markdown("### Model Provider")
    from src.llm_client import LLMClient
    available_providers = [p["id"] for p in LLMClient.list_providers()]
    for prov in available_providers:
        is_active = st.session_state.llm_provider == prov
        label = f"{'●' if is_active else '○'} {prov.title()}"
        if st.button(label, key=f"model_{prov}", use_container_width=True,
                     type="primary" if is_active else "secondary",
                     disabled=is_active):
            get_client(provider=prov)
            st.session_state.llm_provider = prov
            st.rerun()

    st.markdown("---")

    # Rate Limits
    st.markdown("### API Rate Limits")
    rl = _RATE_LIMITER
    limits = {
        "Alpha Vantage": ("alphavantage", 5),
        "SerpAPI": ("serpapi", 100),
        "Web Fetch": ("requests", 50),
    }
    for label, (bucket, cap) in limits.items():
        remaining = rl.remaining(bucket)
        pct = remaining / cap if cap > 0 else 0
        color = "#1a8c5e" if pct > 0.5 else ("#C8963E" if pct > 0.1 else "#C41230")
        st.markdown(f"**{label}**: {remaining}/{cap}")
        st.progress(pct)

    st.markdown("---")

    # Cache Stats
    st.markdown("### Cache")
    cache_entries = len(_CACHE._cache) if hasattr(_CACHE, '_cache') else 0
    st.markdown(f"Entries: **{cache_entries}**")

    st.markdown("---")
    st.markdown(f"**{len(load_skills())}** skills loaded · **{len(REAL_TOOL_REGISTRY)}** tools")

# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------
st.title("FinSkillsTracer v5.3")
st.caption("Real Financial APIs — Alpha Vantage + SerpAPI + Semantic Skill Router")

# --- Query Input ---
st.markdown("### Query")
presets = {
    "🍎 Apple Financials": "What is Apple Inc revenue, net income, and P/E ratio for the latest fiscal year?",
    "🚗 Tesla Earnings": "What were Tesla earnings and EPS surprise for the latest quarter?",
    "⚔️ MSFT vs AAPL": "Compare Microsoft and Apple — market cap, P/E ratio, and revenue",
    "📈 US Inflation & Yields": "What is the current US inflation rate, treasury yield, and unemployment rate?",
    "💱 USD/CNY Exchange": "What is the USD/CNY exchange rate and how has it trended?",
    "📰 Financial News": "Latest financial news about the tech sector and AI stocks",
}

cols = st.columns(6)
for i, (label, query) in enumerate(presets.items()):
    with cols[i]:
        if st.button(label, key=f"preset_{i}", use_container_width=True):
            st.session_state.query_input = query

query = st.text_input(
    "Enter a financial query...",
    value=st.session_state.get("query_input", ""),
    key="query_main",
    placeholder="e.g. What is Apple's revenue and P/E ratio?",
    label_visibility="collapsed",
)

if st.button("▶ Analyze", type="primary", use_container_width=True, disabled=not query.strip()):
    _execute_query(query)

# --- Results Area ---
if st.session_state.current_result:
    result = st.session_state.current_result
    st.markdown("---")

    # Router info
    st.markdown("### Semantic Router → Pipeline")
    st.info(f"**Intent:** `{result.get('query_type', 'general')}` → "
            f"**Pipeline:** `{' → '.join(result.get('tool_sequence', []))}` "
            f"({len(result.get('tool_sequence', []))} tools)")

    # Execution Metrics
    m1, m2, m3, m4 = st.columns(4)
    metrics = result.get("metrics", {})
    with m1:
        with st.container():
            st.markdown(f"<div class='metric-card'><div class='big-num'>{metrics.get('tools', '-')}</div><div class='lbl'>Tools Called</div></div>", unsafe_allow_html=True)
    with m2:
        with st.container():
            st.markdown(f"<div class='metric-card'><div class='big-num'>{metrics.get('duration', '-')}ms</div><div class='lbl'>Total Duration</div></div>", unsafe_allow_html=True)
    with m3:
        with st.container():
            errs = metrics.get('errors', '-')
            color = "#1a8c5e" if errs == 0 else "#C41230"
            st.markdown(f"<div class='metric-card'><div class='big-num' style='color:{color};'>{errs}</div><div class='lbl'>Errors</div></div>", unsafe_allow_html=True)
    with m4:
        with st.container():
            conf = metrics.get('confidence', 0)
            st.markdown(f"<div class='metric-card'><div class='big-num'>{conf}%</div><div class='lbl'>Confidence</div></div>", unsafe_allow_html=True)

    # Data Sources
    sources = result.get("data_sources", [])
    if sources:
        st.markdown("**Data Sources:** " + " ".join(
            [f"<span class='source-badge {s}'>{s.upper()}</span>" for s in sources]
        ), unsafe_allow_html=True)

    # Step-by-step results
    st.markdown("### Step-by-Step Execution")
    steps = result.get("steps", [])
    for i, step in enumerate(steps):
        ds = step.get("data_source", "live")
        ds_icon = {"live": "🟢", "cached": "🟡", "error": "🔴"}.get(ds, "⚪")

        with st.expander(f"{ds_icon} Step {i+1}: **{step.get('tool', '?')}** — {step.get('tool_label', '')} "
                         f"({step.get('duration_ms', 0)}ms, {ds.upper()})",
                         expanded=(i == 0 or step.get("is_error"))):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Input**")
                st.json(step.get("input", {}))
            with c2:
                st.markdown("**Output**")
                if step.get("is_error"):
                    st.error(step.get("error_message", "Unknown error"))
                else:
                    output = step.get("output", {})
                    if isinstance(output, dict) and not output.get("error"):
                        st.json(output)
                    elif isinstance(output, dict) and output.get("error"):
                        st.warning(str(output.get("error")))
                    else:
                        st.json(output)

    # Synthesized Answer
    st.markdown("### Synthesized Answer")
    answer = result.get("answer", "")
    if answer:
        st.markdown(answer)
    else:
        st.warning("No answer synthesized.")

    # History
    st.markdown("### Execution History")
    hist_df = pd.DataFrame([
        {"Query": h.get("query", "")[:60], "Tools": h.get("metrics", {}).get("tools", 0),
         "Duration": f"{h.get('metrics', {}).get('duration', 0)}ms",
         "Errors": h.get("metrics", {}).get("errors", 0)}
        for h in reversed(st.session_state.execution_history[-10:])
    ])
    if not hist_df.empty:
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Query Execution Logic
# ---------------------------------------------------------------------------
def _execute_query(query: str):
    """Execute a financial query through the v5.3 real API pipeline."""
    llm_client = get_client()

    # Step 1: Semantic Router
    with st.spinner("Running semantic skill router..."):
        skills = load_skills()
        candidates = semantic_route_skills(query, limit=6)
        tool_sequence = _build_smart_pipeline(query, candidates)

    # Step 2: Extract entities
    executor = RealToolExecutor(query)
    entities = executor._extract_entities()
    query_type = entities.get("query_type", "general")

    # Step 3: Execute tools with step-by-step progress
    steps = []
    total_duration = 0
    total_errors = 0
    data_sources_set = set()

    progress_text = st.empty()
    progress_bar = st.progress(0)

    for i, tool_name in enumerate(tool_sequence):
        canonical = canonical_tool_name(tool_name)
        tool_info = REAL_TOOL_REGISTRY.get(canonical, REAL_TOOL_REGISTRY.get(tool_name))
        if not tool_info:
            steps.append({
                "tool": canonical, "tool_label": "Unknown",
                "is_error": True, "error_message": f"Unknown tool: {canonical}",
                "data_source": "error", "duration_ms": 0,
                "input": {}, "output": {},
            })
            total_errors += 1
            continue

        tool_input = executor._build_input(canonical, entities)
        progress_text.markdown(f"**Executing tool {i+1}/{len(tool_sequence)}:** `{canonical}` — {tool_info.get('name', '')}")

        # Check cache
        cache_key = f"{canonical}:{json.dumps(tool_input, sort_keys=True)}"
        cached = _CACHE.get(cache_key, tool_info.get("cache_ttl", 3600))

        rate_limit_bucket = tool_info.get("api_source", canonical)
        data_source = "live"

        start_ms = time.time() * 1000

        if cached is not None:
            output = cached
            data_source = "cached"
        elif not _RATE_LIMITER.can_call(rate_limit_bucket):
            fallback = _CACHE.get(cache_key, 99999)
            if fallback:
                output = fallback
                data_source = "cached"
            else:
                output = {"error": f"Rate limited. Retry later.", "rate_limited": True}
                data_source = "error"
                total_errors += 1
        else:
            api_key = API_KEYS.get(tool_info.get("api_source", ""))
            if not api_key and tool_info.get("api_source") in ("alphavantage", "serpapi"):
                output = {"error": f"No API key for {tool_info['api_source']}"}
                data_source = "error"
                total_errors += 1
            else:
                executor_fn = REAL_TOOL_EXECUTORS.get(canonical)
                if executor_fn:
                    try:
                        output = executor_fn(tool_input, api_key)
                        if output.get("error"):
                            data_source = "error"
                            total_errors += 1
                        else:
                            _RATE_LIMITER.record_call(rate_limit_bucket)
                            _CACHE.set(cache_key, output)
                    except Exception as e:
                        output = {"error": str(e)}
                        data_source = "error"
                        total_errors += 1
                else:
                    output = {"error": f"No executor for {canonical}"}
                    data_source = "error"
                    total_errors += 1

        elapsed_ms = int(time.time() * 1000 - start_ms)
        total_duration += elapsed_ms

        data_sources_set.add(data_source if data_source != "error" and not output.get("error") else "error")

        steps.append({
            "tool": canonical,
            "tool_label": tool_info.get("name", canonical),
            "api_source": tool_info.get("api_source", ""),
            "input": tool_input,
            "output": output,
            "data_source": data_source,
            "duration_ms": elapsed_ms,
            "is_error": data_source == "error",
            "error_message": output.get("error", "") if isinstance(output, dict) else "",
        })

        # Update progress
        progress_bar.progress((i + 1) / len(tool_sequence))

        # Spacing for Alpha Vantage rate limit
        if (tool_info.get("api_source") == "alphavantage" and data_source == "live"
                and i + 1 < len(tool_sequence)):
            next_tool = tool_sequence[i + 1]
            next_info = REAL_TOOL_REGISTRY.get(canonical_tool_name(next_tool), REAL_TOOL_REGISTRY.get(next_tool))
            if next_info and next_info.get("api_source") == "alphavantage":
                time.sleep(1.5)

    # Step 4: Synthesize answer from accumulated data
    progress_text.markdown("**Synthesizing answer from real financial data...**")
    full_context = [
        {"tool": s["tool"], "input": s["input"], "output": s["output"],
         "data_source": s.get("data_source", "live"),
         "is_error": s.get("is_error", False)}
        for s in steps
    ]
    executor.context = full_context
    answer, confidence = executor._synthesize_answer()

    progress_text.empty()
    progress_bar.empty()

    # Build result
    sources = list(data_sources_set) if data_sources_set else ["alphavantage"]
    result = {
        "query": query,
        "query_type": query_type,
        "tool_sequence": tool_sequence,
        "steps": steps,
        "answer": answer,
        "data_sources": sources,
        "metrics": {
            "tools": len(steps),
            "duration": total_duration,
            "errors": total_errors,
            "confidence": int(confidence * 100) if confidence else (95 if total_errors == 0 else max(50, 95 - total_errors * 20)),
        },
    }

    st.session_state.current_result = result
    st.session_state.execution_history.append(result)
    st.rerun()


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("FinSkillsTracer v5.3 · Real Financial APIs: Alpha Vantage + SerpAPI · Semantic Router + Step-by-Step Execution")
