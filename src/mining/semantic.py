import collections
from typing import List, Dict, Any
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from models import Skill

# Lazy import to avoid circular dependency and allow offline usage
_llm_evaluator = None


def _get_llm():
    global _llm_evaluator
    if _llm_evaluator is None:
        from evaluation.llm_evaluator import LLMEvaluator
        _llm_evaluator = LLMEvaluator()
    return _llm_evaluator


class SemanticMiner:
    def __init__(self, traces: List[Dict[str, Any]]):
        self.traces = traces

    def mine(self, n_clusters: int = 10, use_llm_descriptions: bool = True) -> List[Skill]:
        trace_texts = []
        for trace in self.traces:
            text = " ".join([call['tool_name'] for call in trace['calls']])
            trace_texts.append(text)

        if not trace_texts:
            return []

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(trace_texts)

        kmeans = KMeans(n_clusters=min(n_clusters, len(trace_texts)),
                        random_state=42, n_init=10)
        kmeans.fit(X)

        clusters = collections.defaultdict(list)
        for idx, label in enumerate(kmeans.labels_):
            clusters[label].append(self.traces[idx])

        skills = []
        for label, cluster_traces in clusters.items():
            # Find representative sequence
            seq_counts = collections.defaultdict(int)
            for t in cluster_traces:
                seq = tuple([call['tool_name'] for call in t['calls']])
                seq_counts[seq] += 1

            top_seq = list(max(seq_counts.items(), key=lambda x: x[1])[0])

            # Collect sample inputs for LLM description
            sample_inputs = []
            for t in cluster_traces[:3]:
                for call in t.get('calls', [])[:3]:
                    inp = call.get('input', {})
                    if inp and isinstance(inp, dict):
                        sample_inputs.append(inp)

            # Generate LLM description or fall back to template
            name = f"Semantic Skill Cluster {label}"
            description = "Skill discovered via semantic clustering of tool sequences."

            if use_llm_descriptions:
                try:
                    llm = _get_llm()
                    desc = llm.generate_skill_description(
                        tool_sequence=top_seq,
                        sample_inputs=sample_inputs,
                        trace_examples=[t.get('trace_id', '') for t in cluster_traces[:5]]
                    )
                    if desc and isinstance(desc, dict):
                        name = desc.get("name", name)
                        description = desc.get("description", description)
                except Exception as e:
                    print(f"  [SemanticMiner] LLM desc generation failed for cluster {label}: {e}")

            skill_id = f"semantic_{label}"
            skills.append(Skill(
                skill_id=skill_id,
                name=name,
                description=description,
                trigger_pattern={"tool_sequence": top_seq,
                                 "semantic_constraint": "Semantic similarity to cluster"},
                preconditions=["Context matches cluster intent"],
                postconditions=["Sub-goal achieved"],
                failure_modes=["Intent mismatch"],
                support=len(cluster_traces),
                confidence=1.0,
                examples=[t['trace_id'] for t in cluster_traces[:3]]
            ))

        return skills
