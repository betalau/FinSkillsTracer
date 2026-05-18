from openai import OpenAI
import os
from typing import List, Dict, Any
import numpy as np

class LLMEvaluator:
    def __init__(self):
        self.client = OpenAI(
            base_url=os.getenv("EFUNDS_BASE_URL", "https://aigc.efunds.com.cn/v1"),
            api_key=os.getenv("EFUNDS_API_KEY", "dummy_key")
        )
        self.user = os.getenv("EFUNDS_USER", "default_user")

    def eval_answer(self, prompt: str, answer: str, gt: str) -> float:
        """
        Score correctness from 0 to 1 using EFundGPT.
        """
        if os.getenv("EFUNDS_API_KEY") is None:
            # Fallback for testing if no key provided
            return 1.0 if str(answer) == str(gt) else 0.0

        try:
            resp = self.client.chat.completions.create(
                model="EFundGPT-air",
                messages=[
                    {"role": "system", "content": "You are a financial evaluation judge."},
                    {"role": "user", "content": f"Question: {prompt}\nAnswer: {answer}\nGround Truth: {gt}\n\nScore correctness from 0 to 1 and return ONLY the numeric score."}
                ],
                temperature=0,
                extra_headers={
                    "Efunds-User-Name": self.user,
                    "Efunds-Acc-Token": self.user,
                    "Efunds-Source": "2025-SX",
                }
            )
            score_text = resp.choices[0].message.content.strip()
            return float(score_text)
        except Exception as e:
            print(f"Eval error: {e}")
            return 0.0

class SkillEvaluatorV2:
    def __init__(self, traces: List[Dict[str, Any]], skills: List[Any]):
        self.traces = traces
        self.skills = skills

    def calculate_metrics(self) -> Dict[str, Any]:
        total_traces = len(self.traces)
        if total_traces == 0:
            return {}

        # 1. Coverage
        covered_traces = 0
        for trace in self.traces:
            trace_seq = [c['tool_name'] for c in trace['calls']]
            for skill in self.skills:
                skill_seq = skill.trigger_pattern.get("tool_sequence", [])
                if self._is_subsequence(skill_seq, trace_seq):
                    covered_traces += 1
                    break
        
        coverage = covered_traces / total_traces

        # 2. Confidence (Average correctness of traces containing the skills)
        # Simplified: 1 - error rate of the dataset
        correct_traces = sum(1 for t in self.traces if t.get('is_correct', False))
        confidence = correct_traces / total_traces

        return {
            "skill_count": len(self.skills),
            "coverage": coverage,
            "confidence": confidence,
            "support_sum": sum(s.support for s in self.skills)
        }

    def _is_subsequence(self, sub, main):
        if not sub: return False
        for i in range(len(main) - len(sub) + 1):
            if main[i:i+len(sub)] == sub:
                return True
        return False
