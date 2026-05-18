from typing import List, Dict, Any
from models import Skill
import collections

class HierarchicalMiner:
    def __init__(self, traces: List[Dict[str, Any]]):
        self.traces = traces

    def mine(self) -> List[Skill]:
        # Heuristic: Detect sub-goal boundaries (e.g., when a tool returns a large structured object or status changes)
        skills_map = collections.defaultdict(int)
        examples_map = collections.defaultdict(list)

        for trace in self.traces:
            current_seq = []
            for call in trace['calls']:
                current_seq.append(call['tool_name'])
                # Boundary signals: status change or specific tools
                if call['tool_name'] in ['Task', 'WebFetch', 'get_company_fundamentals']:
                    if len(current_seq) >= 2:
                        seq_tuple = tuple(current_seq)
                        skills_map[seq_tuple] += 1
                        if len(examples_map[seq_tuple]) < 3:
                            examples_map[seq_tuple].append(trace['trace_id'])
                    current_seq = []
            if len(current_seq) >= 2:
                seq_tuple = tuple(current_seq)
                skills_map[seq_tuple] += 1
                if len(examples_map[seq_tuple]) < 3:
                    examples_map[seq_tuple].append(trace['trace_id'])

        skills = []
        for seq_tuple, count in skills_map.items():
            if count >= 3:
                seq = list(seq_tuple)
                skill_id = f"hier_{hash(seq_tuple) % 10000}"
                skills.append(Skill(
                    skill_id=skill_id,
                    name=f"Hierarchical Sub-skill: {' -> '.join(seq)}",
                    description="Sub-skill identified via task boundary detection.",
                    trigger_pattern={"tool_sequence": seq, "semantic_constraint": "Task boundary"},
                    preconditions=["Previous sub-goal complete"],
                    postconditions=["Intermediate result produced"],
                    failure_modes=["Boundary misdetection"],
                    support=count,
                    confidence=1.0,
                    examples=examples_map[seq_tuple]
                ))
        return sorted(skills, key=lambda x: x.support, reverse=True)
