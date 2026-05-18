import collections
from typing import List, Dict, Any
from models import Skill

class FSPMiner:
    def __init__(self, traces: List[Dict[str, Any]]):
        self.traces = traces

    def mine(self, min_support: int = 5) -> List[Skill]:
        sequences = [[call['tool_name'] for call in trace['calls']] for trace in self.traces]
        pattern_counts = collections.defaultdict(int)
        pattern_examples = collections.defaultdict(list)

        for idx, seq in enumerate(sequences):
            n = len(seq)
            for i in range(n):
                for j in range(i + 2, min(i + 6, n + 1)): # length 2 to 5
                    sub_seq = tuple(seq[i:j])
                    pattern_counts[sub_seq] += 1
                    if len(pattern_examples[sub_seq]) < 3:
                        pattern_examples[sub_seq].append(self.traces[idx]['trace_id'])

        skills = []
        for seq_tuple, count in pattern_counts.items():
            if count >= min_support:
                seq = list(seq_tuple)
                skill_id = f"fsp_{hash(seq_tuple) % 10000}"
                skills.append(Skill(
                    skill_id=skill_id,
                    name=f"Sequential Skill: {' -> '.join(seq)}",
                    description=f"Syntactic regularity found in tool usage: {' -> '.join(seq)}",
                    trigger_pattern={"tool_sequence": seq, "semantic_constraint": None},
                    preconditions=["Tools available in sequence"],
                    postconditions=["Sequence executed successfully"],
                    failure_modes=["Tool in sequence fails"],
                    support=count,
                    confidence=1.0, # To be refined by evaluator
                    examples=pattern_examples[seq_tuple]
                ))
        return sorted(skills, key=lambda x: x.support, reverse=True)
