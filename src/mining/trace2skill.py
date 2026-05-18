from typing import List, Dict, Any
from models import Skill
import collections

class Trace2SkillMiner:
    def __init__(self, traces: List[Dict[str, Any]]):
        self.traces = traces

    def mine(self) -> List[Skill]:
        # Each successful trace is a lesson. We extract policy-like rules.
        # Simplified: Extract rules based on Tool + Input -> Output patterns
        rules = collections.defaultdict(int)
        examples = collections.defaultdict(list)

        for trace in self.traces:
            if not trace.get('is_correct', False):
                continue
            
            for call in trace['calls']:
                # Rule: IF tool_name used with certain input keys, THEN it produced output
                input_keys = tuple(sorted(call.get('input', {}).keys()))
                rule_key = (call['tool_name'], input_keys)
                rules[rule_key] += 1
                if len(examples[rule_key]) < 3:
                    examples[rule_key].append(trace['trace_id'])

        skills = []
        for (tool, keys), count in rules.items():
            if count >= 10:
                skill_id = f"t2s_{hash((tool, keys)) % 10000}"
                skills.append(Skill(
                    skill_id=skill_id,
                    name=f"Policy: {tool} with {keys}",
                    description=f"Operational rule learned from successful trajectories using {tool}.",
                    trigger_pattern={"tool_sequence": [tool], "semantic_constraint": f"Input keys: {keys}"},
                    preconditions=[f"Input contains {keys}"],
                    postconditions=["Accurate output generated"],
                    failure_modes=["Parameter mismatch"],
                    support=count,
                    confidence=1.0,
                    examples=examples[(tool, keys)]
                ))
        return sorted(skills, key=lambda x: x.support, reverse=True)
