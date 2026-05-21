"""
FinSkillsTracer v5.2 — Semantic Skill Router + MCP-Style Step-by-Step Execution.

Key improvements over v5.1:
  1. Semantic Router: LLM-based relevance ranking instead of support-biased keyword matching
  2. Step-by-step MCP execution: tools are called sequentially with parameter chaining
  3. Streaming SSE: each tool step is streamed to the frontend with real outputs

Usage:
    cd FinSkillsTracer
    python web/server_5.2.py
    # Opens at http://localhost:5002
"""

import os
import sys
import json
import time
import re
from typing import Dict, Any, List, Tuple, Generator

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, jsonify, request, Response
from flask_cors import CORS

from src.models import canonical_tool_name
from src.llm_client import LLMClient, get_client

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------
SKILLS: List[Dict] = []
METRICS: Dict = {}
TRACES: List[Dict] = []

# ---------------------------------------------------------------------------
# Tool Registry (MCP-style metadata)
# ---------------------------------------------------------------------------
TOOL_REGISTRY = {
    "discover_companies": {
        "category": "financial_direct",
        "input_schema": {"keywords": "string — company name, ticker, or industry keywords"},
        "output_schema": {"company_id": "int", "company_name": "str", "ticker": "str"},
        "typical_ms": 520,
        "desc": "Discover company IDs by keyword search (name/ticker)",
        "mcp_endpoint": "daloopa/discover_companies",
    },
    "discover_company_series": {
        "category": "financial_direct",
        "input_schema": {"company_id": "int", "keywords": "string (optional)", "periods": "list[str] (optional)"},
        "output_schema": {"series_ids": "list[str]", "series_names": "list[str]", "periods_available": "list[str]"},
        "typical_ms": 636,
        "desc": "Find available financial data series for a company",
        "mcp_endpoint": "daloopa/discover_company_series",
    },
    "get_company_fundamentals": {
        "category": "financial_direct",
        "input_schema": {"company_id": "int", "series_ids": "list[str]", "periods": "list[str]"},
        "output_schema": {"fundamentals": "dict[str, float|str] — metric_name → value"},
        "typical_ms": 979,
        "desc": "Retrieve fundamental financial data for a company",
        "mcp_endpoint": "daloopa/get_company_fundamentals",
    },
    "mcp__daloopa__discover_companies": {
        "category": "financial_mcp",
        "input_schema": {"keywords": "string"},
        "output_schema": {"company_id": "int", "company_name": "str", "ticker": "str"},
        "typical_ms": 403,
        "desc": "MCP: Discover companies via Daloopa MCP namespace",
        "mcp_endpoint": "mcp/daloopa/discover_companies",
    },
    "mcp__daloopa__discover_company_series": {
        "category": "financial_mcp",
        "input_schema": {"company_id": "int", "keywords": "string", "periods": "list[str]"},
        "output_schema": {"series_ids": "list[str]", "periods_available": "list[str]"},
        "typical_ms": 500,
        "desc": "MCP: Discover company series via Daloopa MCP namespace",
        "mcp_endpoint": "mcp/daloopa/discover_company_series",
    },
    "mcp__daloopa__get_company_fundamentals": {
        "category": "financial_mcp",
        "input_schema": {"company_id": "int", "series_ids": "list[str]", "periods": "list[str]"},
        "output_schema": {"fundamentals": "dict[str, float|str]"},
        "typical_ms": 600,
        "desc": "MCP: Get company fundamentals via Daloopa MCP namespace",
        "mcp_endpoint": "mcp/daloopa/get_company_fundamentals",
    },
    "WebSearch": {
        "category": "web_search",
        "input_schema": {"query": "string", "type": "string (optional: 'news'|'financial'|'general')"},
        "output_schema": {"results": "list[dict] — [{title, snippet, url}]"},
        "typical_ms": 3000,
        "desc": "General web search for financial information",
        "mcp_endpoint": "web/search",
    },
    "google_search": {
        "category": "web_search",
        "input_schema": {"query": "string"},
        "output_schema": {"results": "list[dict] — [{title, snippet, url}]"},
        "typical_ms": 3000,
        "desc": "Google search for financial information",
        "mcp_endpoint": "web/google_search",
    },
    "WebFetch": {
        "category": "web_fetch",
        "input_schema": {"url": "string"},
        "output_schema": {"content": "str — page text content"},
        "typical_ms": 2000,
        "desc": "Fetch and parse web page content",
        "mcp_endpoint": "web/fetch",
    },
}


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_data():
    global SKILLS, METRICS, TRACES

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    skills_path = os.path.join(base, "skills_bank_v2.json")
    if os.path.exists(skills_path):
        with open(skills_path, "r", encoding="utf-8") as f:
            SKILLS = json.load(f)
        print(f"[5.2] Loaded {len(SKILLS)} skills")

    for metrics_file in ["phase3_results.json", "phase2_results.json"]:
        metrics_path = os.path.join(base, metrics_file)
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                md = json.load(f)
            if "quality_metrics" in md:
                METRICS.update(md["quality_metrics"])
            if "utility_benchmark" in md:
                METRICS["utility_benchmark"] = md["utility_benchmark"]
            print(f"[5.2] Loaded metrics from {metrics_file}")
            break


# ===========================================================================
# v5.2 FEATURE 1: Semantic Skill Router
# ===========================================================================

def semantic_route_skills(query: str, limit: int = 12) -> List[Dict]:
    """Route query to skills using LLM-based semantic matching.

    Two-phase approach:
      1. Broad retrieval: keyword-overlap pre-filter (top 40)
      2. Semantic ranking: LLM evaluates relevance of each candidate
    """
    if not SKILLS:
        return []

    q_lower = query.lower()
    q_words = set(q_lower.split())

    # --- Phase 1: Broad retrieval (keyword overlap, no support bias) ---
    financial_terms = {
        'revenue', 'eps', 'earnings', 'income', 'growth', 'margin', 'ebitda',
        'fundamental', 'company', 'series', 'discover', 'financial', 'stock',
        'ticker', 'quarter', 'fiscal', 'annual', 'report', 'balance', 'sheet',
        'cash', 'flow', 'ratio', 'valuation', 'market', 'cap', 'dividend', 'yield',
        'profit', 'net', 'operating', 'compare', 'comparison', 'versus',
    }
    finance_bonus = len(q_words & financial_terms) * 0.3

    broad = []
    for skill in SKILLS:
        seq = skill.get("trigger_pattern", {}).get("tool_sequence", [])
        search_text = f"{skill.get('name', '')} {skill.get('description', '')} {' '.join(seq)}".lower()
        text_words = set(search_text.split())
        overlap = len(q_words & text_words)
        score = overlap + finance_bonus  # keyword only, no support bias
        if score > 0.05:
            broad.append((score, skill))

    broad.sort(key=lambda x: x[0], reverse=True)
    broad = broad[:40]  # top 40 for LLM re-ranking

    # --- Phase 2: LLM semantic ranking ---
    # Always attempt LLM re-ranking when >1 candidate — the keyword pre-filter
    # only uses lexical overlap, but the LLM understands semantic relevance.
    if len(broad) <= 1:
        return [_skill_to_candidate(s, i, "B") for i, (_, s) in enumerate(broad)]

    client = get_client()
    if not client.available:
        return [_skill_to_candidate(s, i, "B") for i, (_, s) in enumerate(broad[:limit])]

    # Send enough candidates to LLM for meaningful ranking
    # Minimum 15 so the LLM has a diverse pool even when limit is small
    top_for_llm = broad[:min(len(broad), max(limit * 3, 15))]

    # Build candidate list for LLM evaluation from top_for_llm
    candidates_text_parts = []
    for llm_idx, (_, skill) in enumerate(top_for_llm):
        seq = skill.get("trigger_pattern", {}).get("tool_sequence", [])
        name = skill.get("name", "Unknown")
        desc = skill.get("description", "")[:120]
        candidates_text_parts.append(
            f"[{llm_idx}] {name}\n    Tools: {' → '.join(seq)}\n    Desc: {desc}"
        )
    candidates_text = "\n".join(candidates_text_parts)

    semantic_prompt = f"""You are a financial AI skill router. Rank ALL of these candidate skills by relevance to the query.

Query: {query}

Candidates:
{candidates_text}

For each candidate, evaluate:
- Does this skill's tool sequence directly answer the query?
- Does the skill handle the specific financial metrics mentioned?
- Would executing this skill produce useful results?
- Prefer multi-step pipelines (Discover → Series → Fundamentals) over single-tool skills for detailed queries

Return a JSON array of ranked indices (most relevant first, include ALL indices):
{{"ranking": [3, 0, 7, 12, ...], "reasoning": "one sentence explaining top pick"}}

Return ONLY valid JSON."""

    try:
        result = client.chat_json(semantic_prompt, system="You are a financial AI skill router. Output only valid JSON.", temperature=0.1)
    except Exception:
        result = None

    if result and "ranking" in result:
        ranking = result["ranking"]
        reasoning = result.get("reasoning", "")

        # Build ordered list from LLM ranking (indices refer to top_for_llm)
        candidates = []
        for rank_pos, llm_index in enumerate(ranking):
            if llm_index < len(top_for_llm) and len(candidates) < limit:
                _, skill = top_for_llm[llm_index]
                total = len(ranking)
                tier = "A" if rank_pos < max(total // 3, 1) else ("B" if rank_pos < max(total * 2 // 3, 2) else "C")
                c = _skill_to_candidate(skill, len(candidates), tier)
                c["semantic_rank"] = rank_pos + 1
                c["llm_reasoning"] = reasoning[:200] if rank_pos == 0 else ""
                candidates.append(c)

        # If LLM didn't rank enough, backfill from keyword-sorted top_for_llm
        ranked_ids = {c["skill_id"] for c in candidates}
        for _, skill in top_for_llm:
            if len(candidates) >= limit:
                break
            if skill.get("skill_id") not in ranked_ids:
                c = _skill_to_candidate(skill, len(candidates), "C")
                c["semantic_rank"] = len(candidates) + 1
                candidates.append(c)

        return candidates

    # Fallback: keyword ranking only
    return [_skill_to_candidate(s, i, "B") for i, (_, s) in enumerate(broad[:limit])]


def _skill_to_candidate(skill: Dict, rank: int, tier: str) -> Dict:
    """Convert a raw skill dict to a router candidate."""
    seq = skill.get("trigger_pattern", {}).get("tool_sequence", [])
    canonical_seq = [canonical_tool_name(t) for t in seq]

    tools_detail = []
    total_ms = 0
    for t in canonical_seq:
        info = TOOL_REGISTRY.get(t, {})
        ms = info.get("typical_ms", 500)
        total_ms += ms
        tools_detail.append({
            "tool": t,
            "category": info.get("category", "unknown"),
            "estimated_ms": ms,
        })

    consensus = skill.get("consensus", {})
    cons_score = consensus.get("consensus_score", 0) if isinstance(consensus, dict) else 0

    return {
        "rank": rank + 1,
        "tier": tier,
        "skill_id": skill.get("skill_id", ""),
        "name": skill.get("name", "Unknown"),
        "tool_sequence": canonical_seq,
        "original_sequence": seq,
        "tools_detail": tools_detail,
        "support": skill.get("support", 0),
        "consensus_score": round(cons_score, 4),
        "config_count": consensus.get("configuration_count", 0) if isinstance(consensus, dict) else 0,
        "estimated_total_ms": total_ms,
        "estimated_tool_calls": len(canonical_seq),
        "description": skill.get("description", "")[:150],
    }


# ===========================================================================
# v5.2 FEATURE 2: MCP-Style Step-by-Step Tool Executor
# ===========================================================================

class MCPToolExecutor:
    """Simulates MCP (Model Context Protocol) tool execution with LLM.

    Each tool call goes through the LLM, which generates realistic financial
    data outputs based on the tool's purpose, the input parameters, and the
    accumulated execution context.

    This mimics how a real MCP server would work — each tool is called
    independently, with structured input/output schemas, and results from
    earlier steps feed into later steps (parameter chaining).
    """

    def __init__(self, query: str):
        self.query = query
        self.context: List[Dict] = []  # accumulated step results
        self.client = get_client()

    def execute_sequence(self, tool_sequence: List[str]) -> Generator[Dict, None, None]:
        """Execute a tool sequence step by step, yielding each step's result.

        This is a generator that yields progress events for SSE streaming.
        """
        # Step 0: Entity extraction from query
        yield {"type": "phase", "phase": "extract", "message": "Extracting entities from query..."}

        entities = self._extract_entities()
        yield {"type": "entities", "entities": entities}

        # Execute each tool in sequence
        for i, tool_name in enumerate(tool_sequence):
            canonical = canonical_tool_name(tool_name)
            tool_info = TOOL_REGISTRY.get(canonical, TOOL_REGISTRY.get(tool_name, {}))

            # Yield tool-start event
            yield {
                "type": "tool_start",
                "index": i,
                "tool": canonical,
                "original": tool_name,
                "category": tool_info.get("category", "unknown"),
                "desc": tool_info.get("desc", ""),
                "mcp_endpoint": tool_info.get("mcp_endpoint", ""),
                "input_schema": tool_info.get("input_schema", {}),
            }

            # Build tool input from context
            tool_input = self._build_input(canonical, entities)

            # Execute the tool (LLM-simulated MCP call)
            start_ms = time.time() * 1000
            output = self._execute_tool(canonical, tool_input)
            elapsed_ms = int(time.time() * 1000 - start_ms)

            # Store in context
            step_result = {
                "index": i,
                "tool": canonical,
                "input": tool_input,
                "output": output,
                "duration_ms": elapsed_ms,
                "is_error": output.get("error") is not None,
            }
            self.context.append(step_result)

            # Yield tool-complete event
            yield {
                "type": "tool_done",
                "index": i,
                "tool": canonical,
                "input": tool_input,
                "output": output,
                "duration_ms": elapsed_ms,
                "is_error": step_result["is_error"],
                "context_size": len(self.context),
            }

        # Final answer extraction
        yield {"type": "phase", "phase": "extract_answer", "message": "Extracting final answer from MCP outputs..."}
        answer, confidence = self._extract_answer()
        yield {
            "type": "answer",
            "answer": answer,
            "confidence": confidence,
            "total_tools": len(tool_sequence),
            "total_context": len(self.context),
        }

    def _extract_entities(self) -> Dict[str, Any]:
        """Extract financial entities from the query using LLM."""
        if not self.client.available:
            return self._regex_extract_entities()

        prompt = f"""Extract financial entities from this query. Return JSON.

Query: {self.query}

Return:
{{"companies": [{{"name": "...", "ticker": "..."}}],
 "metrics": ["revenue", "eps", ...],
 "periods": ["FY2024", "Q3", ...],
 "query_type": "single_company | multi_company | comparison | general"}}

If unsure about ticker, use the most common one. Return ONLY valid JSON."""

        result = self.client.chat_json(prompt, system="You are a financial data extraction tool. Output only valid JSON.", temperature=0)
        if result:
            return result
        return self._regex_extract_entities()

    def _regex_extract_entities(self) -> Dict[str, Any]:
        """Fallback entity extraction with regex."""
        q = self.query.lower()
        companies = []
        known = {
            "apple": ("Apple Inc.", "AAPL"), "tesla": ("Tesla Inc.", "TSLA"),
            "microsoft": ("Microsoft Corp.", "MSFT"), "google": ("Alphabet Inc.", "GOOGL"),
            "alphabet": ("Alphabet Inc.", "GOOGL"), "amazon": ("Amazon.com Inc.", "AMZN"),
            "nvidia": ("NVIDIA Corp.", "NVDA"), "meta": ("Meta Platforms Inc.", "META"),
        }
        for key, (name, ticker) in known.items():
            if key in q:
                companies.append({"name": name, "ticker": ticker})

        metrics = []
        for m in ["revenue", "eps", "earnings per share", "net income", "growth", "margin", "ebitda"]:
            if m in q:
                metrics.append(m)

        periods = []
        for p in ["FY2024", "FY2023", "Q1", "Q2", "Q3", "Q4"]:
            if p.lower() in q:
                periods.append(p)

        return {"companies": companies, "metrics": metrics, "periods": periods, "query_type": "single_company"}

    def _build_input(self, tool_name: str, entities: Dict) -> Dict:
        """Build tool input parameters from extracted entities and prior context."""
        canonical = canonical_tool_name(tool_name)

        # Get company info from entities or previous context
        company = None
        if entities.get("companies"):
            company = entities["companies"][0]

        # Check prior context for company_id
        for ctx in self.context:
            out = ctx.get("output", {})
            if isinstance(out, dict) and "company_id" in out and not out.get("error"):
                company = company or {}
                company["id"] = out["company_id"]
                company["name"] = company.get("name") or out.get("company_name", "")
                company["ticker"] = company.get("ticker") or out.get("ticker", "")

        # Check prior context for series_ids
        series_ids = None
        for ctx in self.context:
            out = ctx.get("output", {})
            if isinstance(out, dict) and "series_ids" in out:
                series_ids = out["series_ids"]

        # Build input based on tool
        if canonical in ("discover_companies", "mcp__daloopa__discover_companies"):
            keywords = entities.get("companies", [{}])[0].get("name", "") if entities.get("companies") else ""
            if not keywords:
                keywords = self.query.split("What was ")[-1].split(" for ")[0] if "What was " in self.query else self.query[:80]
            return {"keywords": keywords}

        elif canonical in ("discover_company_series", "mcp__daloopa__discover_company_series"):
            return {
                "company_id": company.get("id", 320193) if company else 320193,
                "keywords": ", ".join(entities.get("metrics", ["financials"])),
                "periods": entities.get("periods", ["FY2024"]),
            }

        elif canonical in ("get_company_fundamentals", "mcp__daloopa__get_company_fundamentals"):
            return {
                "company_id": company.get("id", 320193) if company else 320193,
                "series_ids": series_ids or ["RETAIL_REV", "NET_INC_Q", "EPS_DILUTED"],
                "periods": entities.get("periods", ["FY2024"]),
            }

        elif canonical in ("WebSearch", "google_search"):
            return {"query": self.query[:200], "type": "financial"}

        elif canonical == "WebFetch":
            return {"url": f"https://finance.example.com/company/{company.get('ticker', 'AAPL') if company else 'AAPL'}"}

        return {"query": self.query[:200]}

    def _execute_tool(self, tool_name: str, tool_input: Dict) -> Dict:
        """Execute a single MCP tool via LLM simulation.

        The LLM generates realistic financial data based on the tool's
        purpose and the input parameters. This simulates what a real
        MCP server would return.
        """
        tool_info = TOOL_REGISTRY.get(tool_name, {})

        # Build context from prior steps
        prior_context = ""
        for ctx in self.context[-3:]:  # last 3 steps
            prior_context += f"\n  [{ctx['tool']}] → {json.dumps(ctx.get('output', {}), default=str)[:200]}"

        system_prompt = f"""You are the MCP tool '{tool_name}' ({tool_info.get('mcp_endpoint', '')}).
Your job is to return realistic financial data as this tool would.

Tool purpose: {tool_info.get('desc', 'Execute tool')}
Expected output schema: {json.dumps(tool_info.get('output_schema', {}))}

Rules:
- Return ONLY valid JSON matching the output schema
- Use realistic financial data (plausible but not guaranteed accurate)
- If the input is invalid, return {{"error": "invalid_input", "message": "..."}}
- Do NOT explain or add commentary — just the JSON output"""

        user_prompt = f"""Execute this MCP tool call:

Tool: {tool_name}
Input: {json.dumps(tool_input)}
Original query: {self.query}
Prior context: {prior_context or '(none — this is the first call)'}

Generate realistic financial data output matching the tool's schema.
Return ONLY the JSON output."""

        try:
            result = self.client.chat_json(
                user_prompt,
                system=system_prompt,
                temperature=0.1,
            )
            if result:
                return result
        except Exception as e:
            print(f"[MCP] Tool execution error for {tool_name}: {e}")

        # Hard fallback: generate basic mock
        return self._mock_output(tool_name, tool_input)

    def _mock_output(self, tool_name: str, tool_input: Dict) -> Dict:
        """Hard fallback mock output when LLM is unavailable."""
        canonical = canonical_tool_name(tool_name)
        keywords = str(tool_input.get("keywords", tool_input.get("query", ""))).lower()

        if canonical in ("discover_companies", "mcp__daloopa__discover_companies"):
            company_map = {
                "apple": (320193, "Apple Inc.", "AAPL"),
                "tesla": (219863, "Tesla Inc.", "TSLA"),
                "microsoft": (219864, "Microsoft Corp.", "MSFT"),
                "google": (219865, "Alphabet Inc.", "GOOGL"),
                "amazon": (219866, "Amazon.com Inc.", "AMZN"),
                "nvidia": (219867, "NVIDIA Corp.", "NVDA"),
                "meta": (219868, "Meta Platforms Inc.", "META"),
            }
            for key, (cid, name, ticker) in company_map.items():
                if key in keywords:
                    return {"company_id": cid, "company_name": name, "ticker": ticker, "status": "active"}
            return {"company_id": 320193, "company_name": "Apple Inc.", "ticker": "AAPL", "status": "active"}

        elif canonical in ("discover_company_series", "mcp__daloopa__discover_company_series"):
            return {
                "series_ids": ["RETAIL_REV", "NET_INC_Q", "EPS_DILUTED", "REV_GROWTH_YOY", "OP_INC_Q"],
                "series_names": ["Revenue", "Net Income", "Diluted EPS", "Revenue Growth YoY", "Operating Income"],
                "periods_available": ["FY2020", "FY2021", "FY2022", "FY2023", "FY2024", "Q1FY2024", "Q2FY2024", "Q3FY2024", "Q4FY2024"],
            }

        elif canonical in ("get_company_fundamentals", "mcp__daloopa__get_company_fundamentals"):
            return {
                "fundamentals": {
                    "Revenue_FY2024": 391035000000,
                    "NetIncome_FY2024": 93736000000,
                    "EPS_Diluted_FY2024": 6.43,
                    "Revenue_Growth_YoY": 0.083,
                    "Operating_Income_FY2024": 119000000000,
                },
                "currency": "USD",
                "fiscal_year_end": "2024-09-28",
            }

        elif canonical in ("WebSearch", "google_search"):
            return {
                "results": [
                    {"title": "Financial Results FY2024", "snippet": "Revenue: $391.04B, Net Income: $93.74B, EPS: $6.43", "url": "https://example.com/financials"},
                    {"title": "Quarterly Earnings Report", "snippet": "Q4 2024 revenue increased 8.3% year-over-year", "url": "https://example.com/earnings"},
                ],
                "total_results": 2,
            }

        elif canonical == "WebFetch":
            return {"content": "Financial data retrieved. Revenue: $391.04B. Net Income: $93.74B. EPS: $6.43.", "status_code": 200}

        return {"result": "ok", "tool": tool_name, "input": tool_input}

    def _extract_answer(self) -> Tuple[str, float]:
        """Extract final answer from all accumulated tool outputs."""
        if not self.client.available:
            return "N/A", 0.0

        outputs_text = "\n".join([
            f"[Step {c['index']}] {c['tool']}: {json.dumps(c.get('output', {}), default=str)[:300]}"
            for c in self.context
        ])

        prompt = f"""Extract the final financial answer from these MCP tool outputs.

Query: {self.query}

MCP Tool Execution Trace:
{outputs_text}

Return JSON:
{{"answer": "the specific numeric or factual answer", "confidence": 0.0_to_1.0, "explanation": "brief (max 1 sentence)"}}

Return ONLY valid JSON."""

        result = self.client.chat_json(prompt, system="You are a financial data analyst. Output only valid JSON.", temperature=0)
        if result and "answer" in result:
            return result.get("answer", "N/A"), float(result.get("confidence", 0.5))

        # Regex fallback
        numbers = re.findall(r'[\d,]+\.?\d*', outputs_text)
        ans = numbers[-1].replace(",", "") if numbers else "N/A"
        return ans, 0.5


# ===========================================================================
# API Endpoints
# ===========================================================================

@app.route("/api/health")
def health():
    client = get_client()
    return jsonify({
        "status": "ok",
        "version": "5.2",
        "skills_loaded": len(SKILLS),
        "llm_available": client.available,
        "llm_provider": client.provider,
        "llm_model": client.model,
        "llm_label": client.label,
        "available_providers": LLMClient.detect_available(),
    })


# ---- v5.2: Semantic Skill Router ----
@app.route("/api/skills/route", methods=["POST"])
def skills_route_v52():
    """Semantic skill router: LLM-based relevance ranking."""
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    limit = min(data.get("limit", 12), 20)

    if not query:
        return jsonify({"error": "Query is required"}), 400

    candidates = semantic_route_skills(query, limit=limit)
    best = candidates[0] if candidates else None

    return jsonify({
        "query": query,
        "method": "semantic_llm",
        "total_candidates": len(candidates),
        "best_route": best,
        "candidates": candidates,
        "route_summary": {
            "tier_a_count": sum(1 for c in candidates if c["tier"] == "A"),
            "tier_b_count": sum(1 for c in candidates if c["tier"] == "B"),
            "tier_c_count": sum(1 for c in candidates if c["tier"] == "C"),
            "semantic_ranked": True,
        } if candidates else {},
    })


# ---- v5.2: MCP-Style Step-by-Step Execution (SSE) ----
@app.route("/api/trace/execute")
def trace_execute_v52():
    """SSE endpoint: step-by-step MCP tool execution.

    Query params: ?query=<query>&skill_id=<optional skill_id>
    """
    query = request.args.get("query", "").strip()
    skill_id = request.args.get("skill_id", "").strip()

    if not query:
        return jsonify({"error": "Query is required"}), 400

    # Resolve tool sequence: explicit skill_id, or auto-route via semantic router
    tool_sequence = None
    if skill_id:
        skill = next((s for s in SKILLS if s.get("skill_id") == skill_id), None)
        if skill:
            seq = skill.get("trigger_pattern", {}).get("tool_sequence", [])
            if seq:
                tool_sequence = [canonical_tool_name(t) for t in seq]

    if not tool_sequence:
        candidates = semantic_route_skills(query, limit=1)
        if candidates:
            tool_sequence = candidates[0].get("tool_sequence", ["WebSearch"])
        else:
            tool_sequence = ["WebSearch"]

    executor = MCPToolExecutor(query)

    def generate():
        """SSE event generator — yields step-by-step execution events."""
        # Send metadata
        yield f"data: {json.dumps({'type': 'meta', 'tool_sequence': tool_sequence, 'tool_count': len(tool_sequence), 'query': query})}\n\n"

        for event in executor.execute_sequence(tool_sequence):
            yield f"data: {json.dumps(event)}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


# ---- LLM Provider Management ----
@app.route("/api/llm/providers")
def llm_providers():
    provider_id = request.args.get("provider", "").strip()
    if provider_id:
        client = LLMClient(provider=provider_id)
        return jsonify({
            "provider": client.provider,
            "label": client.label,
            "model": client.model,
            "available": client.available,
        })
    return jsonify({
        "current_provider": get_client().provider,
        "current_model": get_client().model,
        "current_label": get_client().label,
        "providers": LLMClient.list_providers(),
        "available": LLMClient.detect_available(),
    })


@app.route("/api/llm/switch", methods=["POST"])
def llm_switch():
    data = request.get_json(silent=True) or {}
    new_provider = data.get("provider", "").strip()
    new_model = data.get("model", "").strip() or None
    if not new_provider:
        return jsonify({"error": "provider is required"}), 400
    if new_provider not in LLMClient.get_provider_configs():
        return jsonify({"error": f"Unknown provider: {new_provider}"}), 400

    client = get_client(provider=new_provider, model=new_model)
    return jsonify({
        "switched": True,
        "provider": client.provider,
        "label": client.label,
        "model": client.model,
        "available": client.available,
        "warning": None if client.available else f"No API key set for {new_provider}",
    })


# ---- Backward-compatible: skills list & search ----
@app.route("/api/skills", methods=["GET", "POST"])
def skills_list():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        query = data.get("q", data.get("query", ""))
        limit = data.get("limit", 5)
    else:
        query = request.args.get("q", "")
        limit = int(request.args.get("limit", 5))

    if query:
        results = semantic_route_skills(query, limit=limit)
        return jsonify({"query": query, "count": len(results), "skills": results})

    return jsonify({"count": len(SKILLS), "skills": SKILLS[:limit]})


@app.route("/api/skills/<skill_id>")
def skill_detail(skill_id):
    for s in SKILLS:
        if s.get("skill_id") == skill_id:
            return jsonify(s)
    return jsonify({"error": "Skill not found"}), 404


@app.route("/api/metrics")
def metrics():
    return jsonify(METRICS)


# ===========================================================================
# Startup
# ===========================================================================
if __name__ == "__main__":
    load_data()
    client = get_client()
    print("=" * 60)
    print("  FinSkillsTracer v5.2 — Semantic Router + MCP Executor")
    print("=" * 60)
    print(f"  Skills       : {len(SKILLS)}")
    print(f"  LLM          : {client.label} ({client.model}) {'✓' if client.available else '✗ no key'}")
    print(f"  Features     : Semantic LLM Routing + Step-by-Step MCP")
    available = LLMClient.detect_available()
    print(f"  Providers    : {available if available else 'none'}")
    print("=" * 60)
    print("  Starting server at http://localhost:5002")
    print("  Open web/v5.2.html in your browser")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5002, debug=True)
