from typing import List
from models import Skill
import collections

class SkillClawEvolver:
    def __init__(self, skills: List[Skill]):
        self.skills = skills

    def evolve(self) -> List[Skill]:
        """
        Merge similar skills, mutate descriptions, and drop low-fitness skills.
        """
        if not self.skills:
            return []

        # 1. Merge by sequence similarity
        merged_skills = {}
        for skill in self.skills:
            seq_tuple = tuple(skill.trigger_pattern.get("tool_sequence", []))
            if seq_tuple in merged_skills:
                # Update existing skill
                existing = merged_skills[seq_tuple]
                existing.support += skill.support
                existing.examples = list(set(existing.examples + skill.examples))[:5]
            else:
                merged_skills[seq_tuple] = skill

        # 2. Filter by fitness (support threshold)
        evolved = [s for s in merged_skills.values() if s.support >= 5]
        
        # 3. Sort by support
        return sorted(evolved, key=lambda x: x.support, reverse=True)
