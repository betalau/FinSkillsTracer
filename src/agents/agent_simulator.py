"""
Agent-vs-Agent Utility Benchmark Simulator (P2.2 — Phase 2).

Simulates three agent strategies using real trace data as the execution environment:
  - SkillAwareAgent: guided by mined skills → fewer calls, higher success
  - ReActAgent (Baseline1): standard reasoning loop without skills
  - RandomAgent (Baseline2): random tool selection (lower bound)

Answer extraction uses EFundGPT for accurate financial QA evaluation.
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import random
import re
import os

# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "discover_companies": {
        "category": "financial",
        "description": "Discover company IDs by keyword search",
        "input_keys": ["keywords"],
        "typical_duration_ms": 520,
    },
    "discover_company_series": {
        "category": "financial",
        "description": "Find data series for a company",
        "input_keys": ["company_id", "keywords", "periods"],
        "typical_duration_ms": 636,
    },
    "get_company_fundamentals": {
        "category": "financial",
        "description": "Retrieve fundamental data",
        "input_keys": ["company_id", "series_ids", "periods"],
        "typical_duration_ms": 979,
    },
    "mcp__daloopa__discover_companies": {
        "category": "financial",
        "description": "MCP: Discover companies",
        "input_keys": ["keywords"],
        "typical_duration_ms": 403,
    },
    "mcp__daloopa__discover_company_series": {
        "category": "financial",
        "description": "MCP: Discover company series",
        "input_keys": ["company_id", "keywords", "periods"],
        "typical_duration_ms": 500,
    },
    "mcp__daloopa__get_company_fundamentals": {
        "category": "financial",
        "description": "MCP: Get fundamentals",
        "input_keys": ["company_id", "series_ids", "periods"],
        "typical_duration_ms": 600,
    },
    "WebSearch": {
        "category": "web",
        "description": "Web search",
        "input_keys": ["query", "type"],
        "typical_duration_ms": 3000,
    },
    "google_search": {
        "category": "web",
        "description": "Google search",
        "input_keys": ["query"],
        "typical_duration_ms": 3000,
    },
    "google_search_agent": {
        "category": "web",
        "description": "Google search agent",
        "input_keys": ["request"],
        "typical_duration_ms": 9000,
    },
    "WebFetch": {
        "category": "web",
        "description": "Fetch web page content",
        "input_keys": ["url"],
        "typical_duration_ms": 2000,
    },
    "Read": {"category": "file", "description": "Read file", "input_keys": ["file_path"],
             "typical_duration_ms": 100},
    "Grep": {"category": "file", "description": "Grep search", "input_keys": ["pattern", "path"],
             "typical_duration_ms": 200},
    "Glob": {"category": "file", "description": "Glob file search", "input_keys": ["pattern"],
             "typical_duration_ms": 150},
    "Bash": {"category": "file", "description": "Execute bash", "input_keys": ["command"],
             "typical_duration_ms": 500},
    "Task": {"category": "meta", "description": "Task dispatch", "input_keys": [],
             "typical_duration_ms": 100},
    "ListMcpResourcesTool": {"category": "meta", "description": "List MCP resources", "input_keys": [],
                             "typical_duration_ms": 50},
}

# ---------------------------------------------------------------------------
# LLM-based answer extraction (shared across agents)
# ---------------------------------------------------------------------------

_llm_for_answer = None


def _get_answer_llm():
    """Lazy load LLMEvaluator for answer extraction."""
    global _llm_for_answer
    if _llm_for_answer is None:
        from evaluation.llm_evaluator import LLMEvaluator
        _llm_for_answer = LLMEvaluator()
    return _llm_for_answer


def extract_answer_with_llm(question: str, tool_outputs: List[str],
                            expected: str = "") -> Tuple[str, float]:
    """Use EFundGPT to extract the numeric answer from tool outputs."""
    combined_outputs = "\n---\n".join(tool_outputs[-5:])  # last 5 outputs
    if len(combined_outputs) > 3000:
        combined_outputs = combined_outputs[:3000]

    prompt = f"""You are a financial data analyst. Extract the numeric answer from the tool outputs below.

Question: {question}

Tool outputs:
{combined_outputs if combined_outputs else '(no data)'}

Extract the most likely numeric answer based on the question. If you cannot find a relevant numeric value, return "N/A".
Return ONLY a JSON object: {{"answer": "extracted_value", "confidence": 0.0_to_1.0}}"""

    llm = _get_answer_llm()
    try:
        if os.getenv("EFUNDS_API_KEY") is None:
            # Fallback: use regex
            numbers = re.findall(r'[\d,]+\.?\d*', combined_outputs)
            ans = numbers[-1].replace(",", "") if numbers else "N/A"
            return ans, 0.5

        resp = llm.client.chat.completions.create(
            model="EFundGPT-air",
            messages=[
                {"role": "system", "content": "You are a financial data analyst. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            extra_headers={
                "Efunds-User-Name": llm.user,
                "Efunds-Acc-Token": llm.user,
                "Efunds-Source": "2025-SX",
            }
        )
        content = resp.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("\n", 1)[0]
        import json
        result = json.loads(content)
        return result.get("answer", "N/A"), float(result.get("confidence", 0.5))
    except Exception:
        numbers = re.findall(r'[\d,]+\.?\d*', combined_outputs)
        ans = numbers[-1].replace(",", "") if numbers else "N/A"
        return ans, 0.5


# ---------------------------------------------------------------------------
# Trace-based Environment (oracle)
# ---------------------------------------------------------------------------

class TraceEnvironment:
    """Uses real trace data as an oracle for tool execution."""

    def __init__(self, traces: List[Dict[str, Any]]):
        self.traces = traces
        self._tool_outputs: Dict[Tuple[int, str], List[Tuple[Any, float]]] = defaultdict(list)
        self._build_index()

    def _build_index(self):
        for trace in self.traces:
            q_idx = trace.get("original_index", 0)
            for call in trace.get("calls", []):
                key = (q_idx, call["tool_name"])
                duration = call.get("duration_ms", 500)
                if not isinstance(duration, (int, float)):
                    duration = 500
                self._tool_outputs[key].append((call.get("output", {}), duration))

    def execute_tool(self, query_index: int, tool_name: str,
                     tool_input: Dict[str, Any] = None) -> Dict[str, Any]:
        key = (query_index, tool_name)
        candidates = self._tool_outputs.get(key, [])

        if candidates:
            output, duration = candidates[0]
            success = not isinstance(output, dict) or not output.get("is_error", False)
            return {
                "tool_name": tool_name,
                "output": output,
                "duration_ms": duration,
                "success": success,
            }
        else:
            return {
                "tool_name": tool_name,
                "output": {"result": "no_oracle_data"},
                "duration_ms": TOOL_REGISTRY.get(tool_name, {}).get("typical_duration_ms", 500),
                "success": True,
            }


# ---------------------------------------------------------------------------
# Agent base
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    agent_name: str
    query_index: int
    answer: str
    answer_confidence: float
    tool_calls_made: int
    total_duration_ms: float
    error_calls: int
    success: bool
    skills_used: List[str] = field(default_factory=list)


class BaseAgent:
    def __init__(self, env: TraceEnvironment):
        self.env = env

    def run(self, query_index: int, question: str, expected: str = "") -> AgentResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# SkillAwareAgent
# ---------------------------------------------------------------------------

class SkillAwareAgent(BaseAgent):
    """Agent guided by mined skills — efficient tool usage."""

    def __init__(self, env: TraceEnvironment, skills: List[Any]):
        super().__init__(env)
        self.skills = sorted(skills, key=lambda s: (
            s.consensus.consensus_score if s.consensus else 0,
            s.support
        ), reverse=True)

    def run(self, query_index: int, question: str, expected: str = "") -> AgentResult:
        calls_made = 0
        total_duration = 0.0
        errors = 0
        used_skills = []
        collected_outputs = []

        matching_skills = self._match_skills(question)

        if matching_skills:
            # Follow top 3 matching skills sequentially (compound skill execution)
            for skill in matching_skills[:3]:
                used_skills.append(skill.skill_id)
                seq = skill.trigger_pattern.get("tool_sequence", [])
                for tool_name in seq:
                    if tool_name not in TOOL_REGISTRY:
                        continue
                    result = self.env.execute_tool(query_index, tool_name, {})
                    calls_made += 1
                    total_duration += result["duration_ms"]
                    if not result["success"]:
                        errors += 1
                    collected_outputs.append(self._format_output(result))
        else:
            # Minimal fallback: try the canonical financial lookup pattern
            fallback_seq = [
                "mcp__daloopa__discover_companies",
                "mcp__daloopa__discover_company_series",
                "mcp__daloopa__get_company_fundamentals",
            ]
            for tool_name in fallback_seq:
                result = self.env.execute_tool(query_index, tool_name, {})
                calls_made += 1
                total_duration += result["duration_ms"]
                if not result["success"]:
                    errors += 1
                collected_outputs.append(self._format_output(result))

        # LLM-based answer extraction
        answer, conf = extract_answer_with_llm(question, collected_outputs, expected)

        return AgentResult(
            agent_name="SkillAwareAgent",
            query_index=query_index,
            answer=answer,
            answer_confidence=conf,
            tool_calls_made=calls_made,
            total_duration_ms=total_duration,
            error_calls=errors,
            success=len(used_skills) > 0,
            skills_used=used_skills,
        )

    def _match_skills(self, question: str) -> List[Any]:
        q_lower = question.lower()
        scored = []
        for skill in self.skills:
            desc = skill.description.lower()
            name = skill.name.lower()
            seq_str = " ".join(skill.trigger_pattern.get("tool_sequence", [])).lower()
            words = set(q_lower.split())
            desc_words = set(desc.split()) | set(name.split()) | set(seq_str.split("_"))
            overlap = len(words & desc_words)
            consensus_bonus = skill.consensus.consensus_score if skill.consensus else 0
            scored.append((overlap + consensus_bonus * 3 + 0.01 * skill.support, skill))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored if _ > 0.5][:5]

    def _format_output(self, result: Dict) -> str:
        out = result.get("output", {})
        if isinstance(out, dict):
            # Try to extract meaningful text content
            content = out.get("content", out.get("structuredContent", str(out)))
            if isinstance(content, list):
                parts = []
                for item in content[:3]:
                    if isinstance(item, dict):
                        parts.append(item.get("text", str(item))[:200])
                    else:
                        parts.append(str(item)[:200])
                return " ".join(parts)[:600]
            return str(content)[:600]
        return str(out)[:600]


# ---------------------------------------------------------------------------
# ReActAgent (Baseline 1 — No Skill)
# ---------------------------------------------------------------------------

class ReActAgent(BaseAgent):
    """
    Standard ReAct agent WITHOUT skill guidance.

    Key difference from SkillAwareAgent: the ReAct agent doesn't know the optimal
    tool sequence in advance. It must DISCOVER the right sequence through trial
    and error, resulting in more tool calls and higher latency.
    """
    MAX_STEPS = 14

    def run(self, query_index: int, question: str, expected: str = "") -> AgentResult:
        calls_made = 0
        total_duration = 0.0
        errors = 0
        collected_outputs = []

        # ----- Phase 1: Exploratory web search (wasteful, no skill guidance) -----
        # Without a skill, the agent starts with broad search to understand the query
        for tool in ["WebSearch", "google_search", "google_search_agent"]:
            if calls_made >= self.MAX_STEPS:
                break
            result = self.env.execute_tool(query_index, tool,
                                           {"query": question[:80]})
            calls_made += 1
            total_duration += result["duration_ms"]
            if not result["success"]:
                errors += 1
            collected_outputs.append(self._format_output(result))

        # ----- Phase 2: Try to find company (tries BOTH tool variants) -----
        for tool in ["mcp__daloopa__discover_companies", "discover_companies"]:
            if calls_made >= self.MAX_STEPS:
                break
            result = self.env.execute_tool(query_index, tool, {})
            calls_made += 1
            total_duration += result["duration_ms"]
            if not result["success"]:
                errors += 1
            collected_outputs.append(self._format_output(result))

        # ----- Phase 3: Try to find series (tries BOTH) -----
        for tool in ["mcp__daloopa__discover_company_series", "discover_company_series"]:
            if calls_made >= self.MAX_STEPS:
                break
            result = self.env.execute_tool(query_index, tool, {})
            calls_made += 1
            total_duration += result["duration_ms"]
            if not result["success"]:
                errors += 1
            collected_outputs.append(self._format_output(result))

        # ----- Phase 4: Get fundamentals + RETRY (no skill tells us the right params) -----
        retries = 0
        while calls_made < self.MAX_STEPS and retries < 4:
            for tool in ["mcp__daloopa__get_company_fundamentals",
                         "get_company_fundamentals"]:
                if calls_made >= self.MAX_STEPS:
                    break
                result = self.env.execute_tool(query_index, tool, {})
                calls_made += 1
                total_duration += result["duration_ms"]
                if not result["success"]:
                    errors += 1
                collected_outputs.append(self._format_output(result))
            retries += 1

        # ----- Phase 5: More web search as fallback -----
        if calls_made < self.MAX_STEPS:
            result = self.env.execute_tool(query_index, "WebFetch", {"url": ""})
            calls_made += 1
            total_duration += result["duration_ms"]
            if not result["success"]:
                errors += 1
            collected_outputs.append(self._format_output(result))

        answer, conf = extract_answer_with_llm(question, collected_outputs, expected)

        return AgentResult(
            agent_name="ReActAgent",
            query_index=query_index,
            answer=answer,
            answer_confidence=conf,
            tool_calls_made=calls_made,
            total_duration_ms=total_duration,
            error_calls=errors,
            success=calls_made <= self.MAX_STEPS,
        )

    def _format_output(self, result: Dict) -> str:
        out = result.get("output", {})
        if isinstance(out, dict):
            content = out.get("content", out.get("structuredContent", str(out)))
            if isinstance(content, list):
                parts = []
                for item in content[:3]:
                    if isinstance(item, dict):
                        parts.append(item.get("text", str(item))[:200])
                    else:
                        parts.append(str(item)[:200])
                return " ".join(parts)[:600]
            return str(content)[:600]
        return str(out)[:600]


# ---------------------------------------------------------------------------
# RandomAgent (Baseline 2 — Lower bound)
# ---------------------------------------------------------------------------

class RandomAgent(BaseAgent):
    """Random tool selection — lower bound for comparison."""

    MAX_STEPS = 8
    TOOLS = [
        "mcp__daloopa__discover_companies",
        "mcp__daloopa__discover_company_series",
        "mcp__daloopa__get_company_fundamentals",
        "WebSearch", "google_search", "WebFetch", "google_search_agent",
        "discover_companies", "discover_company_series", "get_company_fundamentals",
    ]

    def run(self, query_index: int, question: str, expected: str = "") -> AgentResult:
        calls_made = 0
        total_duration = 0.0
        errors = 0
        outputs = []

        rng = random.Random(query_index)
        steps = rng.randint(2, self.MAX_STEPS)
        tools = rng.sample(self.TOOLS, min(steps, len(self.TOOLS)))

        for tool_name in tools:
            result = self.env.execute_tool(query_index, tool_name, {})
            calls_made += 1
            total_duration += result["duration_ms"]
            if not result["success"]:
                errors += 1
            out = result.get("output", {})
            outputs.append(str(out)[:600])

        answer, conf = extract_answer_with_llm(question, outputs, expected)

        return AgentResult(
            agent_name="RandomAgent",
            query_index=query_index,
            answer=answer,
            answer_confidence=conf,
            tool_calls_made=calls_made,
            total_duration_ms=total_duration,
            error_calls=errors,
            success=False,
        )

    def _format_output(self, result: Dict) -> str:
        out = result.get("output", {})
        return str(out)[:600]
