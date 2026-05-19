from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict, field

# Tool category classification for hierarchical mining
TOOL_CATEGORIES = {
    "mcp__daloopa__discover_companies": "financial_mcp",
    "mcp__daloopa__discover_company_series": "financial_mcp",
    "mcp__daloopa__get_company_fundamentals": "financial_mcp",
    "discover_companies": "financial_direct",
    "discover_company_series": "financial_direct",
    "get_company_fundamentals": "financial_direct",
    "WebSearch": "web_search",
    "google_search": "web_search",
    "google_search_agent": "web_search",
    "WebFetch": "web_fetch",
    "Read": "file_ops",
    "Grep": "file_ops",
    "Glob": "file_ops",
    "Bash": "file_ops",
    "Task": "meta",
    "ListMcpResourcesTool": "meta",
}

# Canonical tool names (normalize mcp__daloopa__X -> X for comparison)
TOOL_ALIASES = {
    "mcp__daloopa__discover_companies": "discover_companies",
    "mcp__daloopa__discover_company_series": "discover_company_series",
    "mcp__daloopa__get_company_fundamentals": "get_company_fundamentals",
}


def get_tool_category(tool_name: str) -> str:
    return TOOL_CATEGORIES.get(tool_name, "unknown")


def canonical_tool_name(tool_name: str) -> str:
    return TOOL_ALIASES.get(tool_name, tool_name)


@dataclass
class ConsensusInfo:
    """Cross-model consensus tracking for a mined skill."""
    configuration_count: int = 0
    configurations: Set[str] = field(default_factory=set)
    query_indices: Set[int] = field(default_factory=set)
    consensus_score: float = 0.0  # configurations / 14 (max model count)
    is_high_consensus: bool = False  # >= 10/14 models

    def to_dict(self):
        return {
            "configuration_count": self.configuration_count,
            "configurations": sorted(list(self.configurations)),
            "query_indices": sorted(list(self.query_indices)),
            "consensus_score": round(self.consensus_score, 4),
            "is_high_consensus": self.is_high_consensus
        }


@dataclass
class Skill:
    skill_id: str
    name: str
    description: str
    trigger_pattern: Dict[str, Any]  # {"tool_sequence": List[str], "semantic_constraint": Optional[str]}
    preconditions: List[str]
    postconditions: List[str]
    failure_modes: List[str]
    support: int
    confidence: float
    examples: List[str]  # trace_segment_ids
    consensus: Optional[ConsensusInfo] = None

    def to_dict(self):
        d = asdict(self)
        if self.consensus:
            d["consensus"] = self.consensus.to_dict()
        return d


@dataclass
class HierarchicalSkill(Skill):
    """Skill with tree structure support for P3.1."""
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    depth: int = 0
    category: str = "mixed"
    is_leaf: bool = False
    boundary_signals: List[str] = field(default_factory=list)

    def to_dict(self):
        d = super().to_dict()
        d.update({
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "depth": self.depth,
            "category": self.category,
            "is_leaf": self.is_leaf,
            "boundary_signals": self.boundary_signals,
        })
        return d


@dataclass
class ParameterizedTemplate:
    """Canonical parameter template for a skill (P3.2)."""
    skill_id: str
    sequence: List[str]
    steps: List[Dict[str, Any]]  # each step: {tool, params: {name: {type, source, value}}}
    variable_count: int = 0
    constant_count: int = 0
    instance_count: int = 0

    def to_dict(self):
        return {
            "skill_id": self.skill_id,
            "sequence": self.sequence,
            "steps": self.steps,
            "variable_count": self.variable_count,
            "constant_count": self.constant_count,
            "instance_count": self.instance_count,
        }
