import collections
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import json

class SkillMiningEngine:
    def __init__(self, normalized_traces: List[Dict[str, Any]]):
        self.traces = normalized_traces

    # --- Strategy A: Frequent Sequence Mining (Simplified PrefixSpan approach) ---
    def mine_frequent_sequences(self, min_support: int = 5, max_len: int = 5) -> List[Dict[str, Any]]:
        """
        Find frequent sub-sequences of tool names.
        """
        sequences = [[call['tool_name'] for call in trace['calls']] for trace in self.traces]
        
        pattern_counts = collections.defaultdict(int)
        for seq in sequences:
            # Generate all sub-sequences up to max_len
            n = len(seq)
            for i in range(n):
                for j in range(i + 1, min(i + max_len + 1, n + 1)):
                    sub_seq = tuple(seq[i:j])
                    pattern_counts[sub_seq] += 1
        
        # Filter by support
        frequent = [
            {"sequence": list(k), "support": v, "strategy": "FrequentSequence"}
            for k, v in pattern_counts.items() if v >= min_support and len(k) > 1
        ]
        return sorted(frequent, key=lambda x: x['support'], reverse=True)

    # --- Strategy B: Semantic Clustering ---
    def mine_semantic_clusters(self, n_clusters: int = 10) -> List[Dict[str, Any]]:
        """
        Cluster traces based on the semantic content of tool calls.
        """
        # Represent each trace as a string of tool names and some input keywords
        trace_texts = []
        for trace in self.traces:
            text = " ".join([call['tool_name'] for call in trace['calls']])
            trace_texts.append(text)
        
        if not trace_texts:
            return []

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(trace_texts)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X)
        
        clusters = collections.defaultdict(list)
        for idx, label in enumerate(kmeans.labels_):
            clusters[label].append(self.traces[idx])
            
        results = []
        for label, cluster_traces in clusters.items():
            # Find the most common sequence in this cluster
            seq_counts = collections.defaultdict(int)
            for t in cluster_traces:
                seq = tuple([call['tool_name'] for call in t['calls']])
                seq_counts[seq] += 1
            
            top_seq = max(seq_counts.items(), key=lambda x: x[1])[0]
            results.append({
                "cluster_id": int(label),
                "representative_sequence": list(top_seq),
                "support": len(cluster_traces),
                "strategy": "SemanticClustering"
            })
            
        return results

    # --- Strategy C: Hierarchical Mining (Sub-goal boundaries) ---
    def mine_hierarchical_skills(self) -> List[Dict[str, Any]]:
        """
        Identify skills by looking for natural boundaries (e.g., changes in target entity).
        Simplified version: split by 'Task' or 'WebSearch' transitions.
        """
        skills = collections.defaultdict(int)
        for trace in self.traces:
            current_skill = []
            for call in trace['calls']:
                current_skill.append(call['tool_name'])
                # Heuristic: certain tools act as "finishers" or "transitioners"
                if call['tool_name'] in ['Task', 'WebFetch']:
                    if len(current_skill) > 1:
                        skills[tuple(current_skill)] += 1
                    current_skill = []
            if len(current_skill) > 1:
                skills[tuple(current_skill)] += 1
                
        results = [
            {"sequence": list(k), "support": v, "strategy": "Hierarchical"}
            for k, v in skills.items() if v > 2
        ]
        return sorted(results, key=lambda x: x['support'], reverse=True)

if __name__ == "__main__":
    # Mock data for testing
    mock_traces = [
        {"calls": [{"tool_name": "A"}, {"tool_name": "B"}, {"tool_name": "C"}]},
        {"calls": [{"tool_name": "A"}, {"tool_name": "B"}, {"tool_name": "C"}]},
        {"calls": [{"tool_name": "A"}, {"tool_name": "B"}]},
        {"calls": [{"tool_name": "X"}, {"tool_name": "Y"}]},
    ]
    engine = SkillMiningEngine(mock_traces)
    print("Frequent Sequences:", engine.mine_frequent_sequences(min_support=2))
    print("Hierarchical Skills:", engine.mine_hierarchical_skills())
