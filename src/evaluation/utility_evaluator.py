"""
Utility Evaluator (P2.2/P2.3).

Runs Agent-vs-Agent benchmarks and computes real utility metrics:
  - Task Success Rate  (EFundGPT-judged)
  - Latency Reduction   (% vs baseline)
  - Tool Call Reduction (% vs baseline)
  - Error Rate Reduction (% vs baseline)
  - Usefulness / Completeness / Generality (LLM-judged)
"""
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import json
import time

from agents.agent_simulator import (
    TraceEnvironment, SkillAwareAgent, ReActAgent, RandomAgent, AgentResult
)
from evaluation.llm_evaluator import LLMEvaluator


class UtilityEvaluator:
    """Runs agent benchmarks and computes comparative utility metrics."""

    def __init__(self,
                 env: TraceEnvironment,
                 skills: List[Any],
                 test_queries: List[Dict[str, Any]],
                 llm: Optional[LLMEvaluator] = None):
        self.env = env
        self.skills = skills
        self.test_queries = test_queries
        self.llm = llm or LLMEvaluator()

    def run_benchmark(self, num_queries: int = 50) -> Dict[str, Any]:
        """Run all three agents on test queries and compute metrics."""
        queries = self.test_queries[:num_queries]
        print(f"\nRunning benchmark on {len(queries)} test queries...")

        # Initialize agents
        skill_agent = SkillAwareAgent(self.env, self.skills)
        react_agent = ReActAgent(self.env)
        random_agent = RandomAgent(self.env)

        results: Dict[str, List[AgentResult]] = {
            "SkillAwareAgent": [],
            "ReActAgent": [],
            "RandomAgent": [],
        }

        # Run each agent on each query
        for i, q in enumerate(queries):
            q_idx = q.get("original_index", i)
            question = q.get("question", f"Query {q_idx}")
            expected = str(q.get("expected_value", ""))

            if i % 10 == 0:
                print(f"  Processing query {i + 1}/{len(queries)}...")

            # SkillAwareAgent
            r1 = skill_agent.run(q_idx, question)
            results["SkillAwareAgent"].append(r1)

            # ReActAgent (Baseline 1)
            r2 = react_agent.run(q_idx, question)
            results["ReActAgent"].append(r2)

            # RandomAgent (Baseline 2)
            r3 = random_agent.run(q_idx, question)
            results["RandomAgent"].append(r3)

        # Compute per-agent stats
        agent_stats = {}
        for name, agent_results in results.items():
            agent_stats[name] = self._compute_agent_stats(agent_results, queries)

        # Compute comparative metrics (vs ReAct baseline)
        comparative = self._compute_comparative(agent_stats)

        return {
            "num_queries": len(queries),
            "agent_stats": agent_stats,
            "comparative_metrics": comparative,
        }

    def _compute_agent_stats(self,
                             results: List[AgentResult],
                             queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(results)
        if n == 0:
            return {}

        total_calls = sum(r.tool_calls_made for r in results)
        total_duration = sum(r.total_duration_ms for r in results)
        total_errors = sum(r.error_calls for r in results)

        # EFundGPT evaluation of correctness
        correct_count = 0
        for r, q in zip(results, queries):
            expected = str(q.get("expected_value", ""))
            score = self.llm.eval_answer(
                prompt=q.get("question", ""),
                answer=r.answer,
                gt=expected,
            )
            if score >= 0.5:
                correct_count += 1

        return {
            "num_queries": n,
            "task_success_rate": round(correct_count / n, 4),
            "avg_tool_calls": round(total_calls / n, 2),
            "total_tool_calls": total_calls,
            "avg_duration_ms": round(total_duration / n, 0),
            "total_duration_ms": round(total_duration, 0),
            "avg_error_calls": round(total_errors / n, 2),
            "total_error_calls": total_errors,
            "error_rate": round(total_errors / max(1, total_calls), 4),
            "skills_used_count": sum(len(r.skills_used) for r in results),
        }

    def _compute_comparative(self, agent_stats: Dict[str, Dict]) -> Dict[str, Any]:
        """Compute relative improvements vs ReAct baseline."""
        baseline = agent_stats.get("ReActAgent", {})
        skill = agent_stats.get("SkillAwareAgent", {})
        random_agent = agent_stats.get("RandomAgent", {})

        if not baseline:
            return {}

        def pct_change(new_val, base_val):
            if base_val == 0:
                return 0
            return round((new_val - base_val) / base_val, 4)

        metrics = {
            "baseline": "ReActAgent",
            "task_success_rate_improvement": round(
                skill.get("task_success_rate", 0) - baseline.get("task_success_rate", 0), 4
            ),
            "latency_reduction": round(
                1 - skill.get("avg_duration_ms", 0) / max(1, baseline.get("avg_duration_ms", 1)), 4
            ),
            "tool_call_reduction": round(
                1 - skill.get("avg_tool_calls", 0) / max(1, baseline.get("avg_tool_calls", 1)), 4
            ),
            "error_rate_reduction": round(
                1 - skill.get("error_rate", 0) / max(0.001, baseline.get("error_rate", 0.001)), 4
            ),
            # Absolute metrics for reference
            "skill_agent_success_rate": skill.get("task_success_rate", 0),
            "react_agent_success_rate": baseline.get("task_success_rate", 0),
            "random_agent_success_rate": random_agent.get("task_success_rate", 0),
            "skill_agent_tool_calls": skill.get("avg_tool_calls", 0),
            "react_agent_tool_calls": baseline.get("avg_tool_calls", 0),
            "random_agent_tool_calls": random_agent.get("avg_tool_calls", 0),
            "skill_agent_error_rate": skill.get("error_rate", 0),
            "react_agent_error_rate": baseline.get("error_rate", 0),
            "random_agent_error_rate": random_agent.get("error_rate", 0),
        }

        return metrics

    def evaluate_skill_quality_llm(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Use EFundGPT to judge Usefulness, Completeness, Generality of skills."""
        if not self.skills:
            return []

        top_skills = sorted(self.skills, key=lambda s: (
            s.consensus.consensus_score if s.consensus else 0,
            s.support
        ), reverse=True)[:top_n]

        assessments = []
        for skill in top_skills:
            try:
                seq = " -> ".join(skill.trigger_pattern.get("tool_sequence", []))
                prompt = f"""Evaluate this mined Agent Skill on three dimensions (score each 1-10):

Skill: {skill.name}
Description: {skill.description}
Tool Sequence: {seq}
Support (occurrences): {skill.support}
Consensus Score: {skill.consensus.consensus_score if skill.consensus else 'N/A'}

Score:
- Usefulness (1-10): Would this skill actually help an agent answer financial questions faster?
- Completeness (1-10): Does the sequence cover all necessary steps, or are key steps missing?
- Generality (1-10): Is this skill general enough to work across different companies/queries, or is it overfit?

Return ONLY a JSON object: {{"usefulness": X, "completeness": X, "generality": X}}"""

                resp = self.llm.client.chat.completions.create(
                    model="EFundGPT-air",
                    messages=[
                        {"role": "system", "content": "You are a financial AI evaluation expert. Output only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    extra_headers={
                        "Efunds-User-Name": self.llm.user,
                        "Efunds-Acc-Token": self.llm.user,
                        "Efunds-Source": "2025-SX",
                    }
                )
                content = resp.choices[0].message.content.strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("\n", 1)[0]
                scores = json.loads(content)
                scores["skill_id"] = skill.skill_id
                scores["skill_name"] = skill.name
                assessments.append(scores)
            except Exception as e:
                print(f"  LLM quality eval failed for {skill.skill_id}: {e}")
                assessments.append({
                    "skill_id": skill.skill_id,
                    "skill_name": skill.name,
                    "usefulness": 5, "completeness": 5, "generality": 5,
                })

        return assessments
