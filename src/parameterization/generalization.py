"""
P3.2 — Upgraded Parameterization Engine.

Key improvements over V1:
  1. Cross-instance voting — aggregate ALL instances, vote on variable vs constant
  2. Precise key-level matching — JSON-path extraction, not substring containment
  3. Parameter type inference — variable / constant / derived / optional
  4. Canonical template generation — single template with type annotations per param

Input: skill_sequence (list of tool names) + traces
Output: ParameterizedTemplate with typed steps
"""
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
import json
import re

from models import ParameterizedTemplate, canonical_tool_name


class ParameterizationEngine:
    """Upgraded parameterization with cross-instance voting and key-level data flow."""

    def __init__(self, traces: List[Dict[str, Any]]):
        self.traces = traces
        self._instance_cache: Dict[Tuple[str, ...], List[List[Dict]]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_variable_mappings(self, skill_sequence: List[str]) -> List[Dict[str, Any]]:
        """Backward-compatible with V1: return template list."""
        template = self.generate_template(skill_sequence)
        if template is None:
            return []
        return [template.to_dict()]

    def generate_template(self, skill_sequence: List[str]) -> Optional[ParameterizedTemplate]:
        """Generate a canonical parameterized template for a skill sequence.

        Collects ALL instances, votes on parameter types, and produces a
        single canonical template with variable/constant/derived annotations.
        """
        instances = self._find_all_instances(skill_sequence)
        if not instances:
            return None

        # Build per-step parameter analysis
        steps = []
        total_vars = 0
        total_consts = 0

        for step_idx, tool_name in enumerate(skill_sequence):
            step_params = self._analyze_step(step_idx, tool_name, instances)
            steps.append(step_params)
            for p in step_params.get("params", {}).values():
                if isinstance(p, dict):
                    if p.get("type") in ("variable", "derived"):
                        total_vars += 1
                    elif p.get("type") == "constant":
                        total_consts += 1

        # Determine skill ID
        skill_id = self._make_skill_id(skill_sequence)

        return ParameterizedTemplate(
            skill_id=skill_id,
            sequence=list(skill_sequence),
            steps=steps,
            variable_count=total_vars,
            constant_count=total_consts,
            instance_count=len(instances),
        )

    def generate_all_templates(self, skills: List[Any],
                               top_n: int = 30) -> List[ParameterizedTemplate]:
        """Generate templates for the top N skills (by support)."""
        sorted_skills = sorted(skills, key=lambda s: s.support, reverse=True)[:top_n]
        templates = []
        for skill in sorted_skills:
            seq = skill.trigger_pattern.get("tool_sequence", [])
            if len(seq) >= 2:
                tmpl = self.generate_template(seq)
                if tmpl:
                    tmpl.skill_id = skill.skill_id  # Use existing skill ID
                    templates.append(tmpl)
        return templates

    # ------------------------------------------------------------------
    # Instance Collection
    # ------------------------------------------------------------------

    def _find_all_instances(self, skill_sequence: List[str]) -> List[List[Dict]]:
        """Find ALL trace instances matching the skill sequence (by tool names).

        Uses canonical tool name matching so that mcp__daloopa__X matches X.
        """
        cache_key = tuple(skill_sequence)
        if cache_key in self._instance_cache:
            return self._instance_cache[cache_key]

        instances = []
        seq_len = len(skill_sequence)

        for trace in self.traces:
            calls = trace.get("calls", [])
            for i in range(len(calls) - seq_len + 1):
                window = calls[i:i + seq_len]
                window_names = [canonical_tool_name(c.get("tool_name", "")) for c in window]
                if window_names == skill_sequence:
                    instances.append(window)

        self._instance_cache[cache_key] = instances
        return instances

    # ------------------------------------------------------------------
    # Output Parsing
    # ------------------------------------------------------------------

    def _parse_output_keys(self, output: Any, max_depth: int = 3) -> Dict[str, Any]:
        """Extract structured keys from output (handles dict, JSON string, nested text)."""
        if output is None or max_depth <= 0:
            return {}
        if isinstance(output, dict):
            result = {}
            for k, v in output.items():
                if k in ("isError", "type", "annotations", "meta"):
                    continue
                if k in ("structuredContent", "content", "text"):
                    # These keys may contain nested structured data
                    inner = self._parse_output_keys(v, max_depth - 1)
                    result.update(inner)
                elif isinstance(v, str):
                    # Try to parse string values as JSON
                    try:
                        parsed = json.loads(v)
                        inner = self._parse_output_keys(parsed, max_depth - 1)
                        result.update(inner)
                    except Exception:
                        result[k] = v
                elif isinstance(v, (list, dict)):
                    inner = self._parse_output_keys(v, max_depth - 1)
                    if inner:
                        result.update(inner)
                    else:
                        result[k] = v
                else:
                    result[k] = v
            return result
        if isinstance(output, str):
            try:
                parsed = json.loads(output)
                return self._parse_output_keys(parsed, max_depth - 1)
            except Exception:
                return {}
        if isinstance(output, list):
            result = {}
            for item in output[:10]:
                if isinstance(item, dict):
                    for k, v in item.items():
                        if k not in result:
                            result[k] = []
                        result[k].append(v)
            return result
        return {}

    # ------------------------------------------------------------------
    # Per-Step Parameter Analysis
    # ------------------------------------------------------------------

    def _analyze_step(self, step_idx: int, tool_name: str,
                      instances: List[List[Dict]]) -> Dict[str, Any]:
        """Analyze parameters for a single step across all instances."""
        # Collect all input key-values across instances
        param_values: Dict[str, List[Any]] = defaultdict(list)
        output_keys: Dict[str, List[Any]] = defaultdict(list)

        for window in instances:
            call = window[step_idx]

            # Input params
            inp = call.get("input", {})
            if isinstance(inp, dict):
                for k, v in inp.items():
                    param_values[k].append(v)
            elif isinstance(inp, str):
                try:
                    import json as _json
                    inp = _json.loads(inp)
                    if isinstance(inp, dict):
                        for k, v in inp.items():
                            param_values[k].append(v)
                except Exception:
                    pass

            # Output keys — parse structured content regardless of format
            out = call.get("output", {})
            parsed_out = self._parse_output_keys(out)
            for k, v in parsed_out.items():
                if k not in ("isError",):
                    output_keys[k].append(v)

        # Classify each input parameter
        params = {}
        for key, values in param_values.items():
            params[key] = self._classify_parameter(
                key, values, step_idx, instances, output_keys
            )

        # Determine outputs of this step (what keys become available)
        output_summary = {}
        for key, values in output_keys.items():
            unique_count = len(set(str(v)[:100] for v in values))
            output_summary[key] = {
                "present_in": len(values),
                "unique_values": unique_count,
                "sample": str(values[0])[:100] if values else None,
            }

        return {
            "tool": tool_name,
            "step_index": step_idx,
            "params": params,
            "outputs": output_summary,
        }

    def _classify_parameter(self, key: str, values: List[Any],
                            step_idx: int, instances: List[List[Dict]],
                            output_keys: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Classify a parameter as variable, constant, derived, or optional."""
        non_none = [v for v in values if v is not None]
        if not non_none:
            return {"type": "optional", "present_count": 0, "total_count": len(values)}

        unique = set(str(v)[:200] for v in non_none)

        # Check if optional (not present in all instances)
        presence = len(non_none) / len(values) if values else 0

        # Check if constant (same value in all instances)
        is_constant = len(unique) <= 1 or (
            len(unique) <= 2 and len(non_none) >= 5
        )

        # Check if derived from previous step output
        derived_from = None
        if step_idx > 0:
            derived_from = self._check_derived(key, non_none, step_idx, instances)

        # Determine type
        if presence < 0.5:
            ptype = "optional"
        elif derived_from:
            ptype = "derived"
        elif is_constant and len(non_none) >= 3:
            ptype = "constant"
        else:
            ptype = "variable"

        result = {
            "type": ptype,
            "present_count": len(non_none),
            "total_count": len(values),
            "presence_rate": round(presence, 3),
            "unique_value_count": len(unique),
        }

        if ptype == "constant":
            result["canonical_value"] = non_none[0]
        elif ptype == "derived":
            result["source"] = derived_from
        elif ptype == "variable":
            result["sample_values"] = [str(v)[:80] for v in list(unique)[:5]]

        return result

    def _check_derived(self, key: str, values: List[Any],
                       step_idx: int, instances: List[List[Dict]]) -> Optional[Dict]:
        """Check if parameter value comes from a previous step's output."""
        # For each instance, check if this key appears in previous step outputs
        match_counts = defaultdict(int)
        total = len(instances)

        for window in instances:
            for prev_idx in range(step_idx):
                prev_call = window[prev_idx]
                prev_out = prev_call.get("output", {})
                parsed = self._parse_output_keys(prev_out)

                # Check if the parameter KEY appears in parsed output
                if key in parsed:
                    match_counts[(prev_idx, key)] += 1

        if not match_counts:
            return None

        best = max(match_counts, key=match_counts.get)
        best_count = match_counts[best]

        if best_count >= total * 0.5:  # At least 50% of instances match
            return {
                "source_step": best[0],
                "source_key": best[1],
                "match_rate": round(best_count / total, 3),
            }
        return None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _make_skill_id(self, seq: List[str]) -> str:
        import hashlib
        h = hashlib.md5("|".join(seq).encode()).hexdigest()[:8]
        return f"param_{h}"

    def get_data_flow_summary(self, skill_sequence: List[str]) -> Dict[str, Any]:
        """High-level summary of data flow in a skill sequence."""
        template = self.generate_template(skill_sequence)
        if template is None:
            return {"error": "No instances found"}

        flow_edges = []
        for step in template.steps:
            for pname, pinfo in step.get("params", {}).items():
                if isinstance(pinfo, dict) and pinfo.get("type") == "derived":
                    src = pinfo.get("source", {})
                    flow_edges.append({
                        "from_step": src.get("source_step"),
                        "from_tool": template.sequence[src.get("source_step")] if src.get("source_step") is not None else "?",
                        "key": src.get("source_key"),
                        "to_step": step["step_index"],
                        "to_tool": step["tool"],
                        "to_param": pname,
                        "match_rate": src.get("match_rate"),
                    })

        return {
            "sequence": template.sequence,
            "total_instances": template.instance_count,
            "variable_params": template.variable_count,
            "constant_params": template.constant_count,
            "data_flow_edges": flow_edges,
            "steps_detail": [
                {
                    "tool": s["tool"],
                    "params_summary": {
                        k: v.get("type", "?")
                        for k, v in s.get("params", {}).items()
                    },
                }
                for s in template.steps
            ],
        }


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------

if __name__ == "__main__":
    # Test with mock data
    mock_window = [
        {
            "tool_name": "discover_companies",
            "input": {"keywords": ["Apple"]},
            "output": {"company_id": 123, "company_name": "Apple Inc."}
        },
        {
            "tool_name": "get_fundamentals",
            "input": {"company_id": 123, "period": "FY2023"},
            "output": {"revenue": "383B", "net_income": "97B"}
        },
    ]

    engine = ParameterizationEngine([])
    # Test internal method
    result = engine._analyze_step(0, "discover_companies", [mock_window])
    print("Step 0 analysis:")
    print(json.dumps(result, indent=2, default=str))

    result1 = engine._analyze_step(1, "get_fundamentals", [mock_window])
    print("\nStep 1 analysis:")
    print(json.dumps(result1, indent=2, default=str))
