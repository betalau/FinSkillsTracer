"""
Multi-Model Consensus Analyzer — Innovation #1 of the task.

Leverages the fact that FinRetrieval runs the same 500 queries across 14 model
configurations. A skill that appears in traces from many different models is more
robust and model-agnostic than one that only appears in a single model's traces.

Key insight: 6 of 14 configs are "webonly" (no Daloopa API access), so consensus
scoring must account for tool-category availability.
"""
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict
from models import Skill, ConsensusInfo

TOTAL_MODEL_CONFIGS = 14

# Model groups by tool access
WEB_ONLY_CONFIGS = {
    "gemini3pro_webonly", "gemini3pro_webonly_reasoning",
    "gpt5.2_webonly", "gpt5.2_webonly_reasoning",
    "opus4.5_webonly", "opus4.5_webonly_reasoning",
}
FULL_TOOL_CONFIGS = {
    "gemini3pro", "gemini3pro_reasoning",
    "gpt5.2", "gpt5.2_reasoning",
    "opus4.5", "opus4.5_reasoning",
    "sonnet4.5", "sonnet4.5_reasoning",
}
assert len(WEB_ONLY_CONFIGS) + len(FULL_TOOL_CONFIGS) == TOTAL_MODEL_CONFIGS

# Tools exclusive to full-tool models
FULL_TOOL_ONLY_PREFIXES = (
    "discover_companies", "discover_company_series", "get_company_fundamentals",
    "mcp__daloopa__",
)

# Consensus thresholds — calibrated for FinRetrieval's 8 full-tool + 6 webonly config split.
# Full-tool skills can reach at most 8 models; web skills can theoretically reach 14 but
# in practice webonly models are fewer in golden paths (lower accuracy rates).
HIGH_CONSENSUS_THRESHOLD = 5           # raw count: >= 5 of 14 configs
HIGH_CONSENSUS_MIN_FULL = 4            # full-tool skills: >= 4 of 8 models
HIGH_CONSENSUS_MIN_WEB = 6             # web skills: >= 6 of 14 models
HIGH_CONSENSUS_RATIO_FULL = HIGH_CONSENSUS_MIN_FULL / 8   # 0.5
HIGH_CONSENSUS_RATIO_WEB = HIGH_CONSENSUS_MIN_WEB / 14    # ~0.43


class ConsensusAnalyzer:
    """Computes cross-model consensus scores for mined skills."""

    def __init__(self, traces: List[Dict[str, Any]],
                 high_threshold: int = HIGH_CONSENSUS_THRESHOLD):
        self.traces = traces
        self.high_threshold = high_threshold

        # Build lookup: (tuple of tool_names) -> set of configurations
        self._seq_to_configs: Dict[Tuple[str, ...], Set[str]] = defaultdict(set)
        self._seq_to_queries: Dict[Tuple[str, ...], Set[int]] = defaultdict(set)
        self._build_index()

    def _build_index(self):
        for trace in self.traces:
            config = trace.get("configuration", "unknown")
            query_idx = trace.get("original_index", -1)
            seen_seqs = set()

            calls = trace.get("calls", [])
            tool_names = [c["tool_name"] for c in calls]
            n = len(tool_names)

            for i in range(n):
                for j in range(i + 1, min(i + 6, n + 1)):
                    sub_seq = tuple(tool_names[i:j])
                    if sub_seq not in seen_seqs:
                        seen_seqs.add(sub_seq)
                        self._seq_to_configs[sub_seq].add(config)
                        self._seq_to_queries[sub_seq].add(query_idx)

    def _classify_tools(self, seq: Tuple[str, ...]) -> str:
        """Classify a tool sequence as 'full', 'web', or 'mixed'."""
        has_full = any(t.startswith(FULL_TOOL_ONLY_PREFIXES) for t in seq)
        has_web = any(not t.startswith(FULL_TOOL_ONLY_PREFIXES) for t in seq)
        if has_full and has_web:
            return "mixed"
        elif has_full:
            return "full"
        else:
            return "web"

    def _get_max_configs(self, tool_category: str) -> int:
        """Maximum number of model configs that can use these tools."""
        if tool_category == "full":
            return len(FULL_TOOL_CONFIGS)  # 8
        elif tool_category == "web":
            return TOTAL_MODEL_CONFIGS     # 14 — all models can use web tools
        else:
            return len(FULL_TOOL_CONFIGS)  # mixed — limited by full-tool access (8)

    def analyze_skills(self, skills: List[Skill]) -> List[Skill]:
        """Attach consensus metadata to each skill with tool-aware scoring."""
        for skill in skills:
            seq = tuple(skill.trigger_pattern.get("tool_sequence", []))
            if not seq:
                skill.consensus = ConsensusInfo()
                continue

            configs = self._seq_to_configs.get(seq, set())
            queries = self._seq_to_queries.get(seq, set())

            tool_cat = self._classify_tools(seq)
            max_configs = self._get_max_configs(tool_cat)

            # Raw score: count / 14
            raw_score = len(configs) / TOTAL_MODEL_CONFIGS

            # Adjusted score: count / max_possible for this tool category
            adjusted_score = len(configs) / max_configs if max_configs > 0 else 0

            # High consensus: adjusted score meets threshold
            threshold = HIGH_CONSENSUS_RATIO_FULL if tool_cat in ("full", "mixed") else HIGH_CONSENSUS_RATIO_WEB
            is_high = adjusted_score >= threshold

            skill.consensus = ConsensusInfo(
                configuration_count=len(configs),
                configurations=configs,
                query_indices=queries,
                consensus_score=round(adjusted_score, 4),  # use adjusted as primary
                is_high_consensus=is_high
            )

        return skills

    def filter_high_consensus(self, skills: List[Skill]) -> List[Skill]:
        """Return only skills with high cross-model consensus."""
        return [s for s in skills if s.consensus and s.consensus.is_high_consensus]

    def get_consensus_stats(self, skills: List[Skill]) -> Dict[str, Any]:
        """Summary statistics for consensus analysis."""
        if not skills:
            return {}

        scores = [s.consensus.consensus_score for s in skills if s.consensus]
        high_count = sum(1 for s in skills if s.consensus and s.consensus.is_high_consensus)
        config_counts = [s.consensus.configuration_count for s in skills if s.consensus]

        # By tool category
        cat_stats = defaultdict(lambda: {"count": 0, "high": 0, "mean_score": 0.0})
        for s in skills:
            seq = tuple(s.trigger_pattern.get("tool_sequence", []))
            cat = self._classify_tools(seq)
            cat_stats[cat]["count"] += 1
            if s.consensus and s.consensus.is_high_consensus:
                cat_stats[cat]["high"] += 1

        for cat in cat_stats:
            cat_skills = [s for s in skills
                          if self._classify_tools(tuple(s.trigger_pattern.get("tool_sequence", []))) == cat
                          and s.consensus]
            if cat_skills:
                cat_stats[cat]["mean_score"] = round(
                    sum(s.consensus.consensus_score for s in cat_skills) / len(cat_skills), 4
                )

        return {
            "total_skills": len(skills),
            "high_consensus_skills": high_count,
            "consensus_rate": round(high_count / len(skills), 4) if skills else 0,
            "mean_consensus_score": round(sum(scores) / len(scores), 4) if scores else 0,
            "max_consensus_score": round(max(scores), 4) if scores else 0,
            "min_consensus_score": round(min(scores), 4) if scores else 0,
            "mean_config_count": round(sum(config_counts) / len(config_counts), 2) if config_counts else 0,
            "threshold": self.high_threshold,
            "total_configs": TOTAL_MODEL_CONFIGS,
            "by_tool_category": {
                cat: {
                    "skill_count": info["count"],
                    "high_consensus_count": info["high"],
                    "high_consensus_rate": round(info["high"] / info["count"], 4) if info["count"] else 0,
                    "mean_score": info["mean_score"],
                    "max_configs": self._get_max_configs(cat),
                }
                for cat, info in sorted(cat_stats.items())
            }
        }
