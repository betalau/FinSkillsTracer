from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class Skill:
    skill_id: str
    name: str
    description: str
    trigger_pattern: Dict[str, Any] # {"tool_sequence": List[str], "semantic_constraint": Optional[str]}
    preconditions: List[str]
    postconditions: List[str]
    failure_modes: List[str]
    support: int
    confidence: float
    examples: List[str] # trace_segment_ids

    def to_dict(self):
        return asdict(self)
