import os
import sys
import random
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
import numpy as np

from openai import OpenAI

# Import canonical tool name for coverage matching
try:
    from models import canonical_tool_name
except ImportError:
    TOOL_ALIASES = {
        "mcp__daloopa__discover_companies": "discover_companies",
        "mcp__daloopa__discover_company_series": "discover_company_series",
        "mcp__daloopa__get_company_fundamentals": "get_company_fundamentals",
    }
    def canonical_tool_name(name):
        return TOOL_ALIASES.get(name, name)


# ---------------------------------------------------------------------------
# LLM Judge (EFundGPT)
# ---------------------------------------------------------------------------

class LLMEvaluator:
    def __init__(self):
        self.client = OpenAI(
            base_url=os.getenv("EFUNDS_BASE_URL", "https://aigc.efunds.com.cn/v1"),
            api_key=os.getenv("EFUNDS_API_KEY", "dummy_key")
        )
        self.user = os.getenv("EFUNDS_USER", "default_user")

    def eval_answer(self, prompt: str, answer: str, gt: str) -> float:
        if os.getenv("EFUNDS_API_KEY") is None:
            return 1.0 if str(answer) == str(gt) else 0.0

        try:
            resp = self.client.chat.completions.create(
                model="EFundGPT-air",
                messages=[
                    {"role": "system", "content": "You are a financial evaluation judge."},
                    {"role": "user", "content": f"Question: {prompt}\nAnswer: {answer}\nGround Truth: {gt}\n\nScore correctness from 0 to 1 and return ONLY the numeric score."}
                ],
                temperature=0,
                extra_headers={
                    "Efunds-User-Name": self.user,
                    "Efunds-Acc-Token": self.user,
                    "Efunds-Source": "2025-SX",
                }
            )
            score_text = resp.choices[0].message.content.strip()
            return float(score_text)
        except Exception as e:
            print(f"Eval error: {e}")
            return 0.0

    def generate_skill_description(self, tool_sequence: List[str],
                                   sample_inputs: List[Dict],
                                   trace_examples: List[str]) -> Dict[str, str]:
        """Generate a financial-domain skill name and description using EFundGPT."""
        seq_str = " -> ".join(tool_sequence)
        input_snippets = "\n".join(
            [str(inp)[:200] for inp in sample_inputs[:3]]
        ) if sample_inputs else "(no input samples)"

        prompt = f"""You are a financial AI agent expert. Analyze this tool call sequence discovered from financial data retrieval traces and generate a concise, meaningful skill name and description.

Tool Sequence: {seq_str}

Sample inputs from real traces:
{input_snippets}

Example trace IDs: {', '.join(trace_examples[:5])}

Generate a JSON response with exactly two fields:
- "name": A short, descriptive skill name (max 8 words, e.g. "Company Fundamentals Lookup" or "Multi-Round Web Research")
- "description": One sentence explaining what this skill accomplishes in financial data retrieval context (max 60 words)

Return ONLY valid JSON, no other text."""

        if os.getenv("EFUNDS_API_KEY") is None:
            return {
                "name": f"Skill: {' -> '.join(tool_sequence[:3])}",
                "description": f"Executes tool sequence: {seq_str}"
            }

        try:
            resp = self.client.chat.completions.create(
                model="EFundGPT-air",
                messages=[
                    {"role": "system", "content": "You are a financial AI agent expert. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                extra_headers={
                    "Efunds-User-Name": self.user,
                    "Efunds-Acc-Token": self.user,
                    "Efunds-Source": "2025-SX",
                }
            )
            import json
            content = resp.choices[0].message.content.strip()
            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content[:-3]
            return json.loads(content)
        except Exception as e:
            print(f"LLM description generation error: {e}")
            return {
                "name": f"Skill: {' -> '.join(tool_sequence[:3])}",
                "description": f"Executes tool sequence: {seq_str}"
            }


# ---------------------------------------------------------------------------
# Train / Test Split
# ---------------------------------------------------------------------------

def split_by_query(traces: List[Dict[str, Any]],
                   test_ratio: float = 0.2,
                   seed: int = 42) -> Tuple[List[Dict], List[Dict], Set[int], Set[int]]:
    """
    Split traces by unique query index so that all 14 model traces for the same
    query stay together — prevents data leakage across model configurations.
    """
    query_to_traces = defaultdict(list)
    for t in traces:
        q_idx = t.get("original_index", t.get("trace_id", ""))
        query_to_traces[q_idx].append(t)

    unique_queries = sorted(query_to_traces.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_queries)

    split_point = int(len(unique_queries) * (1 - test_ratio))
    train_queries = set(unique_queries[:split_point])
    test_queries = set(unique_queries[split_point:])

    train_traces = []
    test_traces = []
    for q in train_queries:
        train_traces.extend(query_to_traces[q])
    for q in test_queries:
        test_traces.extend(query_to_traces[q])

    return train_traces, test_traces, train_queries, test_queries


# ---------------------------------------------------------------------------
# Sequence Similarity (for Uniqueness)
# ---------------------------------------------------------------------------

def sequence_similarity(seq_a: List[str], seq_b: List[str]) -> float:
    """Compute normalized longest common subsequence similarity between two tool sequences."""
    if not seq_a or not seq_b:
        return 0.0

    m, n = len(seq_a), len(seq_b)
    # LCS via DP
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    # Normalize by max length (analogous to edit-distance-based similarity)
    return lcs_len / max(m, n)


# ---------------------------------------------------------------------------
# V2 Skill Evaluator (with train/test split, uniqueness, consensus)
# ---------------------------------------------------------------------------

class SkillEvaluatorV2:
    def __init__(self,
                 train_traces: List[Dict[str, Any]],
                 test_traces: List[Dict[str, Any]],
                 skills: List[Any],
                 train_queries: Optional[Set[int]] = None,
                 test_queries: Optional[Set[int]] = None):
        self.train_traces = train_traces
        self.test_traces = test_traces
        self.skills = skills
        self.train_queries = train_queries or set()
        self.test_queries = test_queries or set()

    def calculate_metrics(self) -> Dict[str, Any]:
        total_train = len(self.train_traces)
        total_test = len(self.test_traces)

        if total_train == 0 and total_test == 0:
            return {}

        # ---- Coverage (on TEST set only) ----
        # Use canonical tool names so that mcp__daloopa__X matches X
        covered_test = 0
        for trace in self.test_traces:
            trace_seq = [canonical_tool_name(c.get('tool_name', '')) for c in trace['calls']]
            for skill in self.skills:
                skill_seq = [canonical_tool_name(t) for t in skill.trigger_pattern.get("tool_sequence", [])]
                if self._is_subsequence(skill_seq, trace_seq):
                    covered_test += 1
                    break

        coverage_test = covered_test / total_test if total_test > 0 else 0

        # ---- Coverage on train set (for reference) ----
        covered_train = 0
        for trace in self.train_traces:
            trace_seq = [canonical_tool_name(c.get('tool_name', '')) for c in trace['calls']]
            for skill in self.skills:
                skill_seq = [canonical_tool_name(t) for t in skill.trigger_pattern.get("tool_sequence", [])]
                if self._is_subsequence(skill_seq, trace_seq):
                    covered_train += 1
                    break

        coverage_train = covered_train / total_train if total_train > 0 else 0

        # ---- Confidence ----
        correct_train = sum(1 for t in self.train_traces if t.get('is_correct', False))
        confidence = correct_train / total_train if total_train > 0 else 0

        # ---- Uniqueness ----
        uniqueness = self._compute_uniqueness()

        # ---- Per-skill confidence (error-aware) ----
        per_skill_confidence = self._compute_per_skill_confidence()

        # ---- Consensus stats ----
        consensus_stats = self._compute_consensus_stats()

        # ---- Model coverage: how many model configs are covered ----
        model_diversity = self._compute_model_diversity()

        return {
            # Core quality metrics
            "skill_count": len(self.skills),
            "train_traces": total_train,
            "test_traces": total_test,
            "train_queries": len(self.train_queries),
            "test_queries": len(self.test_queries),

            # Coverage (test-set based — the real metric)
            "coverage_train": round(coverage_train, 4),
            "coverage_test": round(coverage_test, 4),

            # Confidence
            "confidence": round(confidence, 4),

            # Uniqueness
            "uniqueness": round(uniqueness, 4),

            # Consensus
            "consensus": consensus_stats,

            # Model diversity
            "model_diversity": model_diversity,

            # Support sum
            "support_sum": sum(s.support for s in self.skills),

            # Per-skill info
            "per_skill_confidence": per_skill_confidence,

            # High-consensus skill list
            "high_consensus_skill_ids": [
                s.skill_id for s in self.skills
                if s.consensus and s.consensus.is_high_consensus
            ]
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_subsequence(self, sub, main):
        if not sub:
            return False
        for i in range(len(main) - len(sub) + 1):
            if main[i:i + len(sub)] == sub:
                return True
        return False

    def _compute_uniqueness(self) -> float:
        """% of skill pairs whose sequence similarity < 0.8 (i.e. are distinct)."""
        seqs = [[canonical_tool_name(t) for t in s.trigger_pattern.get("tool_sequence", [])] for s in self.skills]
        n = len(seqs)
        if n < 2:
            return 1.0

        distinct_pairs = 0
        total_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                sim = sequence_similarity(seqs[i], seqs[j])
                if sim < 0.8:
                    distinct_pairs += 1
                total_pairs += 1

        return distinct_pairs / total_pairs if total_pairs > 0 else 0

    def _compute_per_skill_confidence(self) -> Dict[str, float]:
        """Compute confidence per skill: 1 - (error_call_count / total_call_count in skill instances)."""
        result = {}
        for skill in self.skills:
            skill_seq = [canonical_tool_name(t) for t in skill.trigger_pattern.get("tool_sequence", [])]
            if not skill_seq:
                result[skill.skill_id] = 1.0
                continue

            total_instances = 0
            error_instances = 0
            for trace in self.train_traces:
                calls = trace.get("calls", [])
                tool_names = [canonical_tool_name(c.get("tool_name", "")) for c in calls]
                for i in range(len(tool_names) - len(skill_seq) + 1):
                    if tool_names[i:i + len(skill_seq)] == skill_seq:
                        total_instances += 1
                        # Check if any call in this window has an error
                        window = calls[i:i + len(skill_seq)]
                        if any(c.get("status") == "error" for c in window):
                            error_instances += 1
                        break  # count each trace once per skill

            conf = 1 - (error_instances / total_instances) if total_instances > 0 else 1.0
            result[skill.skill_id] = round(conf, 4)

        return result

    def _compute_consensus_stats(self) -> Dict[str, Any]:
        """Aggregate consensus statistics across all skills."""
        if not self.skills:
            return {}

        scores = []
        high_count = 0
        config_distribution = defaultdict(int)

        for s in self.skills:
            if s.consensus:
                scores.append(s.consensus.consensus_score)
                config_distribution[s.consensus.configuration_count] += 1
                if s.consensus.is_high_consensus:
                    high_count += 1

        return {
            "mean_consensus_score": round(sum(scores) / len(scores), 4) if scores else 0,
            "max_consensus_score": round(max(scores), 4) if scores else 0,
            "min_consensus_score": round(min(scores), 4) if scores else 0,
            "high_consensus_count": high_count,
            "high_consensus_rate": round(high_count / len(self.skills), 4) if self.skills else 0,
            "config_distribution": dict(sorted(config_distribution.items())),
        }

    def _compute_model_diversity(self) -> Dict[str, Any]:
        """How many unique model configs are covered by the skill set on test data."""
        all_covered_configs: Set[str] = set()
        per_skill_diversity = {}

        for skill in self.skills:
            skill_seq = [canonical_tool_name(t) for t in skill.trigger_pattern.get("tool_sequence", [])]
            covered_configs: Set[str] = set()
            for trace in self.test_traces:
                trace_seq = [canonical_tool_name(c.get('tool_name', '')) for c in trace['calls']]
                if self._is_subsequence(skill_seq, trace_seq):
                    covered_configs.add(trace.get("configuration", "unknown"))
            per_skill_diversity[skill.skill_id] = len(covered_configs)
            all_covered_configs.update(covered_configs)

        return {
            "total_configs_covered": len(all_covered_configs),
            "configs_covered": sorted(list(all_covered_configs)),
            "max_possible": 14,
        }
