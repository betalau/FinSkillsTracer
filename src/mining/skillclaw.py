"""
P3.3 — SkillClaw Real Evolution.

Evolutionary algorithm with:
  - Tool alias normalization (mcp__daloopa__X -> X)
  - 5 mutation operators: merge_namespace, collapse_loop, add_tool, delete_tool, replace_tool
  - Crossover: combine sub-sequences from two parent skills
  - Fitness function: coverage + consensus + support - uniqueness_penalty
  - Multi-round iteration with tournament selection (k=3) + elitism (top 20)
"""
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
import random
import copy
import math

from models import Skill, ConsensusInfo, canonical_tool_name, get_tool_category


class EvolutionarySkillClaw:
    """P3.3 — Real evolutionary optimizer for mined skills.

    Usage:
        evolver = EvolutionarySkillClaw(skills, traces=train_traces)
        evolved = evolver.evolve(rounds=5)
    """

    # Tool name aliases to normalize (merge variants)
    NAMESPACE_ALIASES = {
        "mcp__daloopa__discover_companies": "discover_companies",
        "mcp__daloopa__discover_company_series": "discover_company_series",
        "mcp__daloopa__get_company_fundamentals": "get_company_fundamentals",
    }

    # Acceptable tool replacements (semantically equivalent)
    TOOL_REPLACEMENTS = {
        "google_search": ["WebSearch", "google_search_agent"],
        "google_search_agent": ["WebSearch", "google_search"],
        "WebSearch": ["google_search", "google_search_agent"],
        "discover_companies": ["mcp__daloopa__discover_companies"],
        "discover_company_series": ["mcp__daloopa__discover_company_series"],
        "get_company_fundamentals": ["mcp__daloopa__get_company_fundamentals"],
    }

    def __init__(self, skills: List[Skill], traces: Optional[List[Dict]] = None):
        self.original_skills = list(skills)
        self.skills = list(skills)
        self.traces = traces or []
        self.population_size = 200
        self.elite_count = 20
        self.tournament_k = 3

        # Pre-compute sequence frequencies from traces for coverage calculation
        self._seq_freq: Dict[Tuple[str, ...], int] = {}
        self._total_traces = len(self.traces)
        self._query_indices: Set[int] = set()
        if self.traces:
            for t in self.traces:
                self._query_indices.add(t.get("original_index", 0))
                calls = t.get("calls", [])
                for i in range(len(calls)):
                    for j in range(i + 1, min(i + 10, len(calls) + 1)):
                        subseq = tuple(c.get("tool_name", "") for c in calls[i:j])
                        self._seq_freq[subseq] = self._seq_freq.get(subseq, 0) + 1

    # ------------------------------------------------------------------
    # Main Evolution Loop
    # ------------------------------------------------------------------

    def evolve(self, rounds: int = 5) -> List[Skill]:
        """Run multi-round evolution and return optimized skills."""
        if not self.skills:
            return []

        print(f"  SkillClaw: evolving {len(self.skills)} skills over {rounds} rounds...")

        population = list(self.skills)
        best_fitness_history = []
        best_population = list(population)

        # Round 0: normalize tool aliases only (collapse_loop is too aggressive for init)
        population = self._apply_operator_silent(population, "merge_namespace")
        print(f"    Round 0 (normalize): {len(self.skills)} -> {len(population)} skills "
              f"(merge_namespace)")

        for rnd in range(rounds):
            # 1. Compute fitness for all
            scored = [(s, self._fitness(s)) for s in population]
            scored.sort(key=lambda x: x[1], reverse=True)

            avg_fitness = sum(f for _, f in scored) / len(scored) if scored else 0
            best_fitness_history.append(avg_fitness)
            print(f"    Round {rnd + 1}: pop={len(population)}, "
                  f"avg_fitness={avg_fitness:.4f}, "
                  f"best_fitness={scored[0][1]:.4f}")

            # Track best population
            if rnd == 0 or avg_fitness >= max(best_fitness_history):
                best_population = list(population)

            # 2. Elitism: preserve top N (capped at population size)
            elite_n = min(self.elite_count, len(scored))
            elites = [copy.deepcopy(s) for s, _ in scored[:elite_n]]

            # 3. Generate offspring
            offspring = list(elites)
            attempts = 0

            while len(offspring) < self.population_size and attempts < self.population_size * 3:
                attempts += 1
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)

                if parent1 is None:
                    continue

                # Crossover (30% probability)
                if random.random() < 0.3 and parent2:
                    child = self._crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(parent1)

                # Mutation (40% probability — less aggressive)
                if random.random() < 0.4 and child:
                    op = random.choice(["add_tool", "delete_tool", "replace_tool"])
                    mutated = self._apply_operator_silent([child], op)
                    if mutated:
                        child = mutated[0]

                if child and child.support > 0:
                    offspring.append(child)

            # 4. Deduplicate + merge original high-support skills back
            population = self._deduplicate(offspring)

            # Keep original high-support skills in the mix (preserve coverage)
            for orig in self.original_skills:
                if len(population) >= self.population_size:
                    break
                if orig.support >= 20:
                    already_in = False
                    oseq = tuple(canonical_tool_name(t) for t in orig.trigger_pattern.get("tool_sequence", []))
                    for p in population:
                        pseq = tuple(canonical_tool_name(t) for t in p.trigger_pattern.get("tool_sequence", []))
                        if oseq == pseq:
                            already_in = True
                            break
                    if not already_in:
                        population.append(copy.deepcopy(orig))

            # 5. Convergence check
            if rnd >= 2:
                recent = best_fitness_history[-3:]
                if max(recent) - min(recent) < 0.005:
                    print(f"    Converged at round {rnd + 1}")
                    break

        # Merge best evolved + original high-support skills for coverage
        # Keep best_population as base but add back original skills
        final = list(best_population)
        for orig in self.original_skills:
            oseq = tuple(canonical_tool_name(t) for t in orig.trigger_pattern.get("tool_sequence", []))
            if not oseq:
                continue
            already_in = any(
                tuple(canonical_tool_name(t) for t in p.trigger_pattern.get("tool_sequence", [])) == oseq
                for p in final
            )
            if not already_in and orig.support >= 10:
                final.append(copy.deepcopy(orig))

        # Filter and sort
        result = []
        seen_seqs: Set[Tuple[str, ...]] = set()
        for skill in sorted(final, key=lambda s: self._fitness(s), reverse=True):
            seq = tuple(canonical_tool_name(t) for t in skill.trigger_pattern.get("tool_sequence", []))
            if seq in seen_seqs:
                continue
            seen_seqs.add(seq)
            if skill.support >= 3:
                result.append(skill)

        print(f"    Final: {len(result)} skills (from {len(self.original_skills)} original)")
        return sorted(result, key=lambda x: (
            x.consensus.consensus_score if x.consensus else 0,
            x.support
        ), reverse=True)

    # ------------------------------------------------------------------
    # Fitness Function
    # ------------------------------------------------------------------

    def _fitness(self, skill: Skill) -> float:
        """Compute fitness: coverage + consensus + support_norm - uniqueness_penalty."""
        seq = skill.trigger_pattern.get("tool_sequence", [])
        if not seq:
            return 0.0

        seq_tuple = tuple(seq)

        # Coverage: how many traces match this sequence (partial match counts)
        coverage = self._compute_coverage(seq_tuple)

        # Consensus: normalized 0-1
        if skill.consensus and skill.consensus.consensus_score:
            consensus = skill.consensus.consensus_score
        else:
            consensus = 0.0

        # Support: normalized by total skills
        max_support = max((s.support for s in self.skills), default=1)
        support_norm = min(skill.support / max_support, 1.0)

        # Uniqueness penalty: overly-specific sequences (long, rare) get penalized
        uniqueness_penalty = self._compute_uniqueness_penalty(seq)

        return (
            0.30 * coverage +
            0.30 * consensus +
            0.20 * support_norm +
            0.20 * (1.0 - uniqueness_penalty)
        )

    def _compute_coverage(self, seq: Tuple[str, ...]) -> float:
        """Coverage: fraction of traces that contain this sequence (as subsequence)."""
        if not self.traces:
            # Fallback: use support relative to total traces in original
            if self._total_traces > 0:
                freq = self._seq_freq.get(seq, 0)
                return min(freq / self._total_traces, 1.0)
            return 0.0

        # Direct counting from pre-computed frequencies
        freq = self._seq_freq.get(seq, 0)
        return min(freq / max(self._total_traces, 1), 1.0)

    def _compute_uniqueness_penalty(self, seq: List[str]) -> float:
        """Penalize overly long or rare sequences (encourages generalization)."""
        if not seq:
            return 0.0

        # Length penalty: sequences longer than 6 steps are too specific
        length_penalty = max(0, (len(seq) - 4) / 20)

        # Rarity penalty: very rare sequences
        canonical = [canonical_tool_name(t) for t in seq]
        unique_ratio = len(set(canonical)) / len(canonical) if canonical else 0

        # Too many unique tools = overfitted
        uniqueness = length_penalty * 0.5 + (1 - unique_ratio) * 0.5
        return min(uniqueness, 0.8)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _tournament_select(self, population: List[Skill]) -> Optional[Skill]:
        """Tournament selection: pick k random, return highest fitness."""
        if not population:
            return None
        k = min(self.tournament_k, len(population))
        candidates = random.sample(population, k)
        best = max(candidates, key=lambda s: self._fitness(s))
        return copy.deepcopy(best)

    # ------------------------------------------------------------------
    # Mutation Operators
    # ------------------------------------------------------------------

    def _apply_operator(self, population: List[Skill], operator: str) -> List[Skill]:
        """Apply a mutation/merge operator to the population (with logging)."""
        result = self._apply_operator_silent(population, operator)
        if result != population:
            print(f"      {operator}: {len(population)} -> {len(result)} skills")
        return result

    def _apply_operator_silent(self, population: List[Skill], operator: str) -> List[Skill]:
        """Apply a mutation/merge operator without logging."""
        if operator == "merge_namespace":
            return self._merge_namespace(population)
        elif operator == "collapse_loop":
            return self._collapse_loop(population)
        elif operator == "add_tool":
            return self._mutate_add_tool(population)
        elif operator == "delete_tool":
            return self._mutate_delete_tool(population)
        elif operator == "replace_tool":
            return self._mutate_replace_tool(population)
        return population

    def _merge_namespace(self, population: List[Skill]) -> List[Skill]:
        """Merge skills that differ only by mcp__daloopa__ prefix."""
        merged: Dict[Tuple[str, ...], Skill] = {}

        for skill in population:
            seq = skill.trigger_pattern.get("tool_sequence", [])
            canonical_seq = tuple(canonical_tool_name(t) for t in seq)

            if canonical_seq in merged:
                existing = merged[canonical_seq]
                existing.support += skill.support
                existing.examples = list(set(existing.examples + skill.examples))[:5]
                if skill.consensus and existing.consensus:
                    existing.consensus.configurations.update(skill.consensus.configurations)
                    existing.consensus.query_indices.update(skill.consensus.query_indices)
                    existing.consensus.configuration_count = len(existing.consensus.configurations)
                    existing.consensus.consensus_score = existing.consensus.configuration_count / 14
                    existing.consensus.is_high_consensus = existing.consensus.configuration_count >= 10
                # Use the most common sequence as canonical
                if skill.support > existing.support:
                    existing.trigger_pattern["tool_sequence"] = seq
                    existing.skill_id = self._new_id(existing.skill_id + "_merged")
            else:
                merged[canonical_seq] = skill
                # Update sequence to use canonical names if different
                if list(canonical_seq) != seq:
                    skill.trigger_pattern["tool_sequence"] = list(canonical_seq)

        result = list(merged.values())
        print(f"      merge_namespace: {len(population)} -> {len(result)} skills")
        return result

    def _collapse_loop(self, population: List[Skill]) -> List[Skill]:
        """Collapse repeated tool calls (WebSearch^n) into single parameterized skill."""
        loop_tools = {"WebSearch", "google_search", "google_search_agent"}
        seen_loops: Dict[str, Skill] = {}  # keyed by canonical tool name
        non_loops = []

        for skill in population:
            seq = skill.trigger_pattern.get("tool_sequence", [])
            canonical = [canonical_tool_name(t) for t in seq]
            unique_tools = set(canonical)

            # Check if this is a pure loop (all same canonical tool)
            if len(unique_tools) == 1 and canonical[0] in loop_tools:
                tool = canonical[0]
                if tool in seen_loops:
                    existing = seen_loops[tool]
                    existing.support += skill.support
                    existing.examples = list(set(existing.examples + skill.examples))[:5]
                    # Keep the shorter sequence
                    if len(seq) < len(existing.trigger_pattern.get("tool_sequence", [])):
                        existing.trigger_pattern["tool_sequence"] = seq
                else:
                    skill.name = f"Research Loop ({tool}, {len(seq)} iterations)"
                    skill.description = f"Parameterized {tool} loop, {len(seq)} iterations typical"
                    skill.trigger_pattern["loop_tool"] = tool
                    skill.trigger_pattern["loop_count"] = len(seq)
                    seen_loops[tool] = skill
            else:
                non_loops.append(skill)

        result = non_loops + list(seen_loops.values())
        print(f"      collapse_loop: {len(population)} -> {len(result)} skills")
        return result

    def _mutate_add_tool(self, population: List[Skill]) -> List[Skill]:
        """Insert a commonly co-occurring tool into a sequence."""
        result = []
        for skill in population:
            seq = list(skill.trigger_pattern.get("tool_sequence", []))
            if len(seq) < 2 or len(seq) >= 8:
                result.append(skill)
                continue

            # Find a position where a tool frequently co-occurs
            best_insert = None
            best_freq = 0

            for i in range(len(seq) + 1):
                for tool_name, freq in self._get_cooccurring_tools(seq, i).items():
                    if freq > best_freq:
                        best_freq = freq
                        best_insert = (i, tool_name)

            if best_insert and best_freq >= 3:
                new_seq = list(seq)
                new_seq.insert(best_insert[0], best_insert[1])
                new_skill = copy.deepcopy(skill)
                new_skill.skill_id = self._new_id(skill.skill_id + "_add")
                new_skill.trigger_pattern["tool_sequence"] = new_seq
                new_skill.name = skill.name + " (extended)"
                new_skill.support = max(1, skill.support // 2)  # reduced confidence
                result.append(new_skill)

            result.append(skill)
        return result

    def _mutate_delete_tool(self, population: List[Skill]) -> List[Skill]:
        """Remove a redundant tool from a sequence."""
        result = []
        for skill in population:
            seq = list(skill.trigger_pattern.get("tool_sequence", []))
            if len(seq) <= 2:
                result.append(skill)
                continue

            # Try removing each tool position, keep best fitness
            best_seq = seq
            best_fit = self._fitness(skill)

            for i in range(len(seq)):
                candidate = seq[:i] + seq[i + 1:]
                # Check fitness via sequence frequency
                cand_freq = self._seq_freq.get(tuple(candidate), 0)
                if cand_freq > self._seq_freq.get(tuple(best_seq), 0):
                    best_seq = candidate

            if best_seq != seq:
                new_skill = copy.deepcopy(skill)
                new_skill.skill_id = self._new_id(skill.skill_id + "_del")
                new_skill.trigger_pattern["tool_sequence"] = best_seq
                new_skill.name = skill.name + " (simplified)"
                new_skill.support = max(skill.support, self._seq_freq.get(tuple(best_seq), 0))
                result.append(new_skill)

            result.append(skill)
        return result

    def _mutate_replace_tool(self, population: List[Skill]) -> List[Skill]:
        """Replace a tool with its semantic equivalent."""
        result = []
        for skill in population:
            seq = list(skill.trigger_pattern.get("tool_sequence", []))
            replacements = self.TOOL_REPLACEMENTS
            mutated = False

            for i, tool in enumerate(seq):
                if tool in replacements:
                    for alt in replacements[tool]:
                        new_seq = list(seq)
                        new_seq[i] = alt
                        alt_freq = self._seq_freq.get(tuple(new_seq), 0)
                        orig_freq = self._seq_freq.get(tuple(seq), 0)
                        if alt_freq > orig_freq:
                            seq = new_seq
                            mutated = True
                            break

            if mutated:
                new_skill = copy.deepcopy(skill)
                new_skill.skill_id = self._new_id(skill.skill_id + "_rep")
                new_skill.trigger_pattern["tool_sequence"] = seq
                new_skill.support = self._seq_freq.get(tuple(seq), skill.support)
                result.append(new_skill)

            result.append(skill)
        return result

    # ------------------------------------------------------------------
    # Crossover
    # ------------------------------------------------------------------

    def _crossover(self, parent1: Skill, parent2: Skill) -> Optional[Skill]:
        """Combine sub-sequences from two parent skills.

        If parents share a prefix or suffix, exchange middle segments.
        """
        seq1 = list(parent1.trigger_pattern.get("tool_sequence", []))
        seq2 = list(parent2.trigger_pattern.get("tool_sequence", []))

        if len(seq1) < 2 or len(seq2) < 2:
            return copy.deepcopy(parent1)

        # Find longest common prefix
        prefix_len = 0
        for a, b in zip(seq1, seq2):
            if canonical_tool_name(a) == canonical_tool_name(b):
                prefix_len += 1
            else:
                break

        # Find longest common suffix
        suffix_len = 0
        for a, b in zip(reversed(seq1), reversed(seq2)):
            if canonical_tool_name(a) == canonical_tool_name(b):
                suffix_len += 1
            else:
                break

        if prefix_len > 0 and suffix_len > 0:
            # Crossover: prefix from parent1, middle, suffix from parent2
            middle1 = seq1[prefix_len:len(seq1) - suffix_len] if suffix_len > 0 else seq1[prefix_len:]
            middle2 = seq2[prefix_len:len(seq2) - suffix_len] if suffix_len > 0 else seq2[prefix_len:]
            suffix = seq1[len(seq1) - suffix_len:] if suffix_len > 0 else []

            # Combine: keep whichever middle is shorter (favored)
            middle = middle1 if len(middle1) <= len(middle2) else middle2
            new_seq = seq1[:prefix_len] + middle + suffix

            if len(new_seq) >= 2 and new_seq != seq1:
                child = copy.deepcopy(parent1)
                child.skill_id = self._new_id(parent1.skill_id + "_xover")
                child.trigger_pattern["tool_sequence"] = new_seq
                child.name = f"Crossover: {' -> '.join(new_seq[:3])}{'...' if len(new_seq) > 3 else ''}"
                child.support = max(1, (parent1.support + parent2.support) // 4)
                return child

        return copy.deepcopy(parent1)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _get_cooccurring_tools(self, seq: List[str], position: int) -> Dict[str, int]:
        """Find tools that frequently appear at a given position in similar sequences."""
        cooccur = defaultdict(int)
        if not self.traces:
            return cooccur

        prefix = seq[:position]
        suffix = seq[position:]

        for trace in self.traces:
            calls = trace.get("calls", [])
            tool_names = [c.get("tool_name", "") for c in calls]
            for i in range(len(tool_names) - len(prefix) - len(suffix)):
                if (tool_names[i:i + len(prefix)] == prefix and
                        tool_names[i + len(prefix) + 1:i + len(prefix) + 1 + len(suffix)] == suffix):
                    middle_tool = tool_names[i + len(prefix)]
                    cooccur[middle_tool] += 1

        return dict(cooccur)

    def _deduplicate(self, population: List[Skill]) -> List[Skill]:
        """Remove duplicate skills (same canonical sequence)."""
        seen: Dict[Tuple[str, ...], Skill] = {}
        for skill in population:
            seq = tuple(canonical_tool_name(t)
                       for t in skill.trigger_pattern.get("tool_sequence", []))
            if seq in seen:
                existing = seen[seq]
                existing.support += skill.support
                if skill.consensus and existing.consensus:
                    existing.consensus.configurations.update(skill.consensus.configurations)
                    existing.consensus.query_indices.update(skill.consensus.query_indices)
                    existing.consensus.configuration_count = len(existing.consensus.configurations)
            else:
                seen[seq] = skill
        return list(seen.values())

    def _new_id(self, base: str) -> str:
        import hashlib
        h = hashlib.md5(base.encode()).hexdigest()[:6]
        return f"evo_{h}"


# ------------------------------------------------------------------
# Backward-compatible wrapper used by Phase 1/2 pipelines
# ------------------------------------------------------------------

class SkillClawEvolver:
    """Original Phase 1/2 evolver: merge identical sequences + filter by support ≥ 5.

    This is the backward-compatible class used by main_v2.py and run_phase2.py.
    For Phase 3 real evolution, use EvolutionarySkillClaw.
    """

    def __init__(self, skills: List[Skill]):
        self.skills = skills

    def evolve(self) -> List[Skill]:
        if not self.skills:
            return []

        merged_skills: Dict[Tuple[str, ...], Skill] = {}
        for skill in self.skills:
            seq_tuple = tuple(skill.trigger_pattern.get("tool_sequence", []))
            if seq_tuple in merged_skills:
                existing = merged_skills[seq_tuple]
                existing.support += skill.support
                existing.examples = list(set(existing.examples + skill.examples))[:5]
                if existing.consensus and skill.consensus:
                    existing.consensus.configurations.update(skill.consensus.configurations)
                    existing.consensus.query_indices.update(skill.consensus.query_indices)
                    existing.consensus.configuration_count = len(existing.consensus.configurations)
                    existing.consensus.consensus_score = existing.consensus.configuration_count / 14
                    existing.consensus.is_high_consensus = existing.consensus.configuration_count >= 10
                elif skill.consensus:
                    existing.consensus = skill.consensus
            else:
                merged_skills[seq_tuple] = skill

        evolved = [s for s in merged_skills.values() if s.support >= 5]
        return sorted(evolved, key=lambda x: (
            x.consensus.consensus_score if x.consensus else 0,
            x.support
        ), reverse=True)
