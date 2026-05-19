"""
P3.1 — Hierarchical Mining with Tree Structure.

Recursive boundary detection using 3 signal types:
  1. Category transitions — tool category changes mark sub-goal boundaries
  2. Duration spikes — tool taking >3x neighbor median signals sub-goal completion
  3. I/O dependency chains — no data flow between tools indicates boundary

Output: HierarchicalSkill tree with parent_id/children_ids/depth/category.
"""
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import statistics
import hashlib

from models import (
    Skill, HierarchicalSkill, ConsensusInfo,
    get_tool_category, canonical_tool_name
)


class HierarchicalMiner:
    """Mines skills with recursive boundary detection → tree structure."""

    # Categories that indicate phase transitions
    CATEGORY_ORDER = [
        "web_search",
        "web_fetch",
        "financial_mcp",
        "financial_direct",
        "file_ops",
        "meta",
    ]

    def __init__(self, traces: List[Dict[str, Any]]):
        self.traces = traces
        self._category_cache: Dict[str, str] = {}

    def mine(self, min_support: int = 3) -> List[Skill]:
        """Main entry: mine skills (backward-compatible, returns flat List[Skill]).

        The returned Skill objects now include parent-child relationships
        via trigger_pattern['parent_id'] and trigger_pattern['children_ids'].
        """
        # Phase 1: Segment all traces into sub-goal sequences
        all_segments: List[Tuple[str, ...]] = []
        segment_trace_map: Dict[Tuple[str, ...], List[Dict]] = defaultdict(list)

        for trace in self.traces:
            calls = trace.get("calls", [])
            if len(calls) < 2:
                continue

            boundaries = self._detect_boundaries(calls)
            segments = self._split_at_boundaries(calls, boundaries)

            for seg in segments:
                if len(seg) >= 2:
                    seq = tuple(c["tool_name"] for c in seg)
                    all_segments.append(seq)
                    segment_trace_map[seq].append(trace)

        # Phase 2: Build flat skills from segments
        flat_skills: List[HierarchicalSkill] = []
        seq_counter: Dict[Tuple[str, ...], int] = defaultdict(int)
        for seq in all_segments:
            seq_counter[seq] += 1

        for seq, count in seq_counter.items():
            if count < min_support:
                continue
            skill = self._make_skill(seq, count, segment_trace_map[seq])
            flat_skills.append(skill)

        # Phase 3: Link parent-child relationships
        self._link_tree(flat_skills)

        # Phase 4: Build parent/child links in trigger_pattern for backward compat
        result = []
        for node in flat_skills:
            skill = self._to_skill(node)
            result.append(skill)
        return sorted(result, key=lambda x: x.support, reverse=True)

    def mine_tree(self, min_support: int = 3) -> List[HierarchicalSkill]:
        """Return tree-structured skills with parent_id/children_ids set."""
        all_segments: List[Tuple[str, ...]] = []
        segment_trace_map: Dict[Tuple[str, ...], List[Dict]] = defaultdict(list)

        for trace in self.traces:
            calls = trace.get("calls", [])
            if len(calls) < 2:
                continue
            boundaries = self._detect_boundaries(calls)
            segments = self._split_at_boundaries(calls, boundaries)
            for seg in segments:
                if len(seg) >= 2:
                    seq = tuple(c["tool_name"] for c in seg)
                    all_segments.append(seq)
                    segment_trace_map[seq].append(trace)

        flat_skills: List[HierarchicalSkill] = []
        seq_counter: Dict[Tuple[str, ...], int] = defaultdict(int)
        for seq in all_segments:
            seq_counter[seq] += 1
        for seq, count in seq_counter.items():
            if count < min_support:
                continue
            skill = self._make_skill(seq, count, segment_trace_map[seq])
            flat_skills.append(skill)

        # Build parent-child relationships in-place on flat_skills
        self._link_tree(flat_skills)

        # Create category root nodes
        cat_roots = self._make_category_roots(flat_skills)

        return cat_roots + flat_skills

    def _link_tree(self, skills: List[HierarchicalSkill]):
        """Set parent_id and children_ids on skills in-place."""
        # Sort by sequence length (longer = more general = parent)
        sorted_skills = sorted(skills, key=lambda s: (
            -len(s.trigger_pattern.get("tool_sequence", [])),
            -(s.consensus.consensus_score if s.consensus else 0),
            -s.support
        ))

        for i, skill in enumerate(sorted_skills):
            skill_seq = skill.trigger_pattern.get("tool_sequence", [])
            for j in range(i):
                parent = sorted_skills[j]
                parent_seq = parent.trigger_pattern.get("tool_sequence", [])
                if self._is_subsequence_of(skill_seq, parent_seq):
                    skill.parent_id = parent.skill_id
                    if skill.skill_id not in parent.children_ids:
                        parent.children_ids.append(skill.skill_id)
                    skill.depth = parent.depth + 1
                    break

    def _make_category_roots(self, skills: List[HierarchicalSkill]) -> List[HierarchicalSkill]:
        """Create category-level root nodes that group skills by category."""
        by_cat: Dict[str, List[HierarchicalSkill]] = defaultdict(list)
        for s in skills:
            if s.parent_id is None:
                s.parent_id = f"root_{s.category}"
            by_cat[s.category].append(s)

        roots = []
        for cat, cat_skills in by_cat.items():
            cat_root = HierarchicalSkill(
                skill_id=f"root_{cat}",
                name=f"Category: {cat}",
                description=f"Root for {len(cat_skills)} {cat} sub-skills",
                trigger_pattern={"tool_sequence": []},
                preconditions=[],
                postconditions=[],
                failure_modes=[],
                support=sum(s.support for s in cat_skills),
                confidence=1.0,
                examples=[],
                parent_id="virtual_root",
                children_ids=[s.skill_id for s in cat_skills],
                depth=0,
                category=cat,
                is_leaf=False,
            )
            roots.append(cat_root)
        return roots

    def _to_skill(self, node: HierarchicalSkill) -> Skill:
        """Convert HierarchicalSkill to backward-compatible Skill."""
        tp = dict(node.trigger_pattern)
        tp["parent_id"] = node.parent_id
        tp["children_ids"] = node.children_ids
        tp["depth"] = node.depth
        tp["category"] = node.category

        return Skill(
            skill_id=node.skill_id,
            name=node.name,
            description=node.description,
            trigger_pattern=tp,
            preconditions=node.preconditions,
            postconditions=node.postconditions,
            failure_modes=node.failure_modes,
            support=node.support,
            confidence=node.confidence,
            examples=node.examples,
        )

    # ------------------------------------------------------------------
    # Boundary Detection
    # ------------------------------------------------------------------

    def _detect_boundaries(self, calls: List[Dict]) -> List[int]:
        """Return indices AFTER which a boundary exists (i.e., split points)."""
        if len(calls) < 2:
            return []

        boundaries = []
        durations = [c.get("duration_ms", 500) or 500 for c in calls]
        med_dur = statistics.median(durations) if durations else 500

        for i in range(len(calls) - 1):
            signals = []

            # Signal 1: Category transition
            cat_curr = get_tool_category(calls[i].get("tool_name", ""))
            cat_next = get_tool_category(calls[i + 1].get("tool_name", ""))
            if cat_curr != cat_next:
                signals.append(f"category:{cat_curr}->{cat_next}")

            # Signal 2: Duration spike (>3x median of neighbors)
            curr_dur = calls[i].get("duration_ms", 500) or 500
            neighbor_durs = [d for j, d in enumerate(durations) if j != i and abs(j - i) <= 2]
            local_med = statistics.median(neighbor_durs) if neighbor_durs else med_dur
            if local_med > 0 and curr_dur > 3 * local_med:
                signals.append(f"duration_spike:{curr_dur}ms_vs_{local_med:.0f}ms")

            # Signal 3: I/O dependency break (no data flow to next tool)
            has_flow = self._has_data_flow(calls[i], calls[i + 1])
            if not has_flow:
                signals.append("no_data_flow")

            # Need at least 2 signals to declare a boundary
            if len(signals) >= 2:
                boundaries.append(i + 1)  # split AFTER position i

        return sorted(set(boundaries))

    def _has_data_flow(self, call_a: Dict, call_b: Dict) -> bool:
        """Check if call_a's output keys appear in call_b's input."""
        out_a = call_a.get("output", {})
        in_b = call_b.get("input", {})

        if not isinstance(out_a, dict) or not isinstance(in_b, dict):
            return False

        # Extract meaningful keys from output
        out_keys = set()
        for key in out_a:
            if key in ("isError", "structuredContent", "content", "result"):
                continue
            out_keys.add(key)
        # Also extract keys from structuredContent if present
        sc = out_a.get("structuredContent", {})
        if isinstance(sc, dict):
            for key in sc:
                if key not in ("isError",):
                    out_keys.add(key)

        in_keys = set(in_b.keys())

        # Check for any key overlap
        common = out_keys & in_keys
        return len(common) > 0

    def _split_at_boundaries(self, calls: List[Dict], boundaries: List[int]) -> List[List[Dict]]:
        """Split call list at detected boundary indices."""
        segments = []
        start = 0
        for b in sorted(boundaries):
            if b > start and (b - start) >= 2:
                segments.append(calls[start:b])
            start = b
        if len(calls) - start >= 2:
            segments.append(calls[start:])
        return segments

    # ------------------------------------------------------------------
    # Skill Construction
    # ------------------------------------------------------------------

    def _make_skill(self, seq: Tuple[str, ...], support: int,
                    traces: List[Dict]) -> HierarchicalSkill:
        """Create a HierarchicalSkill from a segment sequence."""
        seq_list = list(seq)
        categories = [get_tool_category(t) for t in seq_list]
        dominant_cat = max(set(categories), key=categories.count)

        skill_id = self._make_id(seq_list)
        name = self._make_name(seq_list, dominant_cat)
        depth = self._estimate_depth(seq_list, categories)

        return HierarchicalSkill(
            skill_id=skill_id,
            name=name,
            description=f"{dominant_cat} sub-goal: {' -> '.join(seq_list[:4])}"
                       f"{'...' if len(seq_list) > 4 else ''}",
            trigger_pattern={"tool_sequence": seq_list},
            preconditions=["Previous sub-goal complete"],
            postconditions=[f"Sub-goal achieved ({dominant_cat})"],
            failure_modes=["Boundary misdetection"],
            support=support,
            confidence=1.0,
            examples=[t.get("trace_id", "") for t in traces[:5]],
            parent_id=None,
            children_ids=[],
            depth=depth,
            category=dominant_cat,
            is_leaf=len(seq_list) <= 2,
        )

    def _make_id(self, seq: List[str]) -> str:
        h = hashlib.md5("|".join(seq).encode()).hexdigest()[:6]
        return f"hier3_{h}"

    def _make_name(self, seq: List[str], category: str) -> str:
        """Generate descriptive name based on category and tools."""
        canonical = [canonical_tool_name(t) for t in seq]
        unique_tools = list(dict.fromkeys(canonical))  # deduplicated, order-preserved

        if category == "web_search":
            return f"Web Research ({len(seq)} searches)"
        elif category in ("financial_mcp", "financial_direct"):
            if len(unique_tools) >= 3:
                return "Daloopa Pipeline"
            elif len(unique_tools) == 2:
                return f"Financial Lookup: {' → '.join(unique_tools[:2])}"
            else:
                return f"Financial: {unique_tools[0]}"
        elif category == "web_fetch":
            return f"Web Extraction: {unique_tools[0]}"
        elif category == "file_ops":
            return f"File Operations: {len(seq)} steps"
        else:
            return f"Mixed: {len(seq)} steps"

    def _estimate_depth(self, seq: List[str], categories: List[str]) -> int:
        """Estimate tree depth based on sequence complexity."""
        unique_cats = len(set(categories))
        if unique_cats <= 1 and len(seq) <= 2:
            return 3  # leaf
        elif unique_cats <= 1:
            return 2  # mid-level
        else:
            return 1  # top-level (cross-category)

    # ------------------------------------------------------------------
    # Tree Construction
    # ------------------------------------------------------------------

    def _is_subsequence_of(self, short: List[str], long: List[str]) -> bool:
        """Check if short sequence is a sub-sequence of long sequence."""
        if len(short) >= len(long):
            return False
        short_str = "|".join(short)
        long_str = "|".join(long)
        return short_str in long_str

    def get_tree_summary(self, roots: List[HierarchicalSkill]) -> Dict[str, Any]:
        """Return tree statistics for reporting."""
        all_skills: List[HierarchicalSkill] = []

        def collect(node):
            all_skills.append(node)
            for child_id in node.children_ids:
                # Collect stats even without pointer chasing
                pass

        for root in roots:
            collect(root)

        # Count children recursively from roots
        def count_descendants(node, nodes_map):
            total = 1
            for cid in node.children_ids:
                if cid in nodes_map:
                    total += count_descendants(nodes_map[cid], nodes_map)
            return total

        # Build node map from all skills passed to tree builder
        depths = [s.depth for s in all_skills]
        categories = set(s.category for s in all_skills)
        non_empty_roots = [r for r in roots if r.children_ids]

        return {
            "total_nodes": len(all_skills),
            "root_nodes": len(non_empty_roots),
            "max_depth": max(depths) if depths else 0,
            "avg_depth": sum(depths) / len(depths) if depths else 0,
            "categories": sorted(categories),
            "by_category": {
                cat: len([s for s in all_skills if s.category == cat])
                for cat in categories
            },
            "tree_structure": self._render_tree_text(roots),
        }

    def _render_tree_text(self, roots: List[HierarchicalSkill], indent: int = 0) -> str:
        """Render tree as indented text."""
        lines = []
        for i, node in enumerate(roots[:20]):  # cap at 20 roots
            prefix = "  " * indent + ("├─ " if i < len(roots) - 1 else "└─ ")
            lines.append(
                f"{prefix}{node.name} [{node.skill_id}] "
                f"support={node.support} depth={node.depth} cat={node.category}"
            )
        return "\n".join(lines)


