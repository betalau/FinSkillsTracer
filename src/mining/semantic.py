import collections
from typing import List, Dict, Any
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from models import Skill
import numpy as np

class SemanticMiner:
    def __init__(self, traces: List[Dict[str, Any]]):
        self.traces = traces

    def mine(self, n_clusters: int = 10) -> List[Skill]:
        trace_texts = []
        for trace in self.traces:
            # Encode tool calls and some metadata
            text = " ".join([call['tool_name'] for call in trace['calls']])
            trace_texts.append(text)
        
        if not trace_texts:
            return []

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(trace_texts)
        
        kmeans = KMeans(n_clusters=min(n_clusters, len(trace_texts)), random_state=42, n_init=10)
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
            
            skill_id = f"semantic_{label}"
            skills.append(Skill(
                skill_id=skill_id,
                name=f"Semantic Skill Cluster {label}",
                description=f"Skill discovered via semantic clustering of tool sequences.",
                trigger_pattern={"tool_sequence": top_seq, "semantic_constraint": "Semantic similarity to cluster"},
                preconditions=["Context matches cluster intent"],
                postconditions=["Sub-goal achieved"],
                failure_modes=["Intent mismatch"],
                support=len(cluster_traces),
                confidence=1.0,
                examples=[t['trace_id'] for t in cluster_traces[:3]]
            ))
            
        return skills
