"""
Error Pattern Mining (P2.1).

Distinguishes two error types:
  - Tool-level: a tool call returned is_error=True (API failure, timeout, etc.)
  - Answer-level: all tools succeeded but the final answer was wrong

Mines failure modes from the 979 error-containing traces and 2,028 wrong-answer traces.
"""
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict
import json


class ErrorPatternMiner:
    """Mines error patterns from both successful and failed traces."""

    def __init__(self, all_traces: List[Dict[str, Any]]):
        self.all_traces = all_traces

    def compute_confidence(self, skill_sequence: List[str]) -> Dict[str, Any]:
        """
        Compute real confidence for a skill across ALL traces (not just golden paths).

        Returns:
          - tool_confidence: 1 - (instances_with_tool_error / total_instances)
          - answer_confidence: 1 - (instances_with_wrong_answer / total_instances)
          - overall_confidence: weighted combination
        """
        total_instances = 0
        tool_error_instances = 0
        answer_error_instances = 0
        both_error = 0

        for trace in self.all_traces:
            calls = trace.get("calls", [])
            tool_names = [c["tool_name"] for c in calls]
            n = len(tool_names)

            for i in range(n - len(skill_sequence) + 1):
                if tool_names[i:i + len(skill_sequence)] == skill_sequence:
                    total_instances += 1
                    window = calls[i:i + len(skill_sequence)]

                    has_tool_error = any(
                        c.get("status") == "error" for c in window
                    )
                    has_answer_error = not trace.get("is_correct", True)

                    if has_tool_error:
                        tool_error_instances += 1
                    if has_answer_error:
                        answer_error_instances += 1
                    if has_tool_error and has_answer_error:
                        both_error += 1
                    break

        if total_instances == 0:
            return {
                "tool_confidence": 1.0,
                "answer_confidence": 1.0,
                "overall_confidence": 1.0,
                "total_instances": 0,
                "tool_error_count": 0,
                "answer_error_count": 0,
            }

        tool_conf = 1 - (tool_error_instances / total_instances)
        answer_conf = 1 - (answer_error_instances / total_instances)
        # Overall: penalize both types, but count overlap only once
        unique_error = tool_error_instances + answer_error_instances - both_error
        overall = 1 - (unique_error / total_instances)

        return {
            "tool_confidence": round(tool_conf, 4),
            "answer_confidence": round(answer_conf, 4),
            "overall_confidence": round(overall, 4),
            "total_instances": total_instances,
            "tool_error_count": tool_error_instances,
            "answer_error_count": answer_error_instances,
        }

    def mine_failure_modes(self, skill_sequence: List[str]) -> List[Dict[str, Any]]:
        """Extract common failure patterns for a skill sequence."""
        failures = []

        for trace in self.all_traces:
            calls = trace.get("calls", [])
            tool_names = [c["tool_name"] for c in calls]
            n = len(tool_names)

            for i in range(n - len(skill_sequence) + 1):
                if tool_names[i:i + len(skill_sequence)] == skill_sequence:
                    window = calls[i:i + len(skill_sequence)]

                    # Tool-level errors
                    for j, call in enumerate(window):
                        if call.get("status") == "error":
                            failures.append({
                                "type": "tool_error",
                                "trace_id": trace.get("trace_id", ""),
                                "configuration": trace.get("configuration", ""),
                                "step_in_skill": j,
                                "tool_name": call["tool_name"],
                                "error_detail": str(call.get("output", ""))[:200],
                            })

                    # Answer-level errors
                    if not trace.get("is_correct", True) and not any(
                        c.get("status") == "error" for c in window
                    ):
                        failures.append({
                            "type": "answer_error",
                            "trace_id": trace.get("trace_id", ""),
                            "configuration": trace.get("configuration", ""),
                            "expected": trace.get("expected_value", ""),
                        })
                    break

        return failures

    def get_error_statistics(self) -> Dict[str, Any]:
        """Global error statistics across all traces."""
        total = len(self.all_traces)
        tool_error_count = 0
        answer_error_count = 0
        tool_error_traces = set()
        error_tool_distribution: Dict[str, int] = defaultdict(int)
        error_config_distribution: Dict[str, int] = defaultdict(int)

        for trace in self.all_traces:
            tid = trace.get("trace_id", "")
            has_tool_err = False
            has_answer_err = not trace.get("is_correct", True)

            for call in trace.get("calls", []):
                if call.get("status") == "error":
                    has_tool_err = True
                    tool_error_count += 1
                    error_tool_distribution[call["tool_name"]] += 1
                    error_config_distribution[trace.get("configuration", "")] += 1

            if has_tool_err:
                tool_error_traces.add(tid)
            if has_answer_err:
                answer_error_count += 1

        return {
            "total_traces": total,
            "traces_with_tool_errors": len(tool_error_traces),
            "traces_with_answer_errors": answer_error_count,
            "total_tool_error_calls": tool_error_count,
            "tool_error_rate": round(tool_error_count / max(1, sum(
                len(t.get("calls", [])) for t in self.all_traces
            )), 4),
            "most_error_prone_tools": dict(
                sorted(error_tool_distribution.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            "error_by_config": dict(
                sorted(error_config_distribution.items(), key=lambda x: x[1], reverse=True)
            ),
        }
