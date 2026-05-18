import sys
import os
import json
import argparse
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ingestion.normalization import DataIngestionEngine
from mining.fsp import FSPMiner
from mining.semantic import SemanticMiner
from mining.hierarchical import HierarchicalMiner
from mining.trace2skill import Trace2SkillMiner
from mining.skillclaw import SkillClawEvolver
from evaluation.llm_evaluator import SkillEvaluatorV2

def run_ablation(methods, train_sizes):
    all_results = []
    
    ingest = DataIngestionEngine('../data/tool_traces.parquet', '../data/scores.parquet')
    
    for size in train_sizes:
        print(f"\n--- Testing Train Size: {size} ---")
        golden_df = ingest.filter_golden_paths()
        # Sample by query index
        subset_df = golden_df[golden_df['index'] < size]
        normalized = ingest.normalize_traces(subset_df)
        
        for method in methods:
            print(f"Running Method: {method}")
            skills = []
            if method == 'A':
                miner = FSPMiner(normalized)
                skills = miner.mine(min_support=max(2, size // 50))
            elif method == 'B':
                miner = SemanticMiner(normalized)
                skills = miner.mine(n_clusters=min(10, len(normalized)))
            elif method == 'C':
                miner = HierarchicalMiner(normalized)
                skills = miner.mine()
            elif method == 'D':
                miner = Trace2SkillMiner(normalized)
                skills = miner.mine()
            elif method == 'E':
                # Combined + Evolution
                all_raw = []
                all_raw += FSPMiner(normalized).mine(min_support=max(2, size // 50))
                all_raw += SemanticMiner(normalized).mine(n_clusters=min(5, len(normalized)))
                evolver = SkillClawEvolver(all_raw)
                skills = evolver.evolve()
            
            evaluator = SkillEvaluatorV2(normalized, skills)
            metrics = evaluator.calculate_metrics()
            metrics['method'] = method
            metrics['train_size'] = size
            all_results.append(metrics)
            
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='+', default=['A', 'B', 'C', 'D', 'E'])
    parser.add_argument('--train_sizes', nargs='+', type=int, default=[100, 250, 500])
    args = parser.parse_args()
    
    results = run_ablation(args.methods, args.train_sizes)
    
    with open('ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    df = pd.DataFrame(results)
    print("\nAblation Results Summary:")
    print(df.pivot(index='train_size', columns='method', values='coverage'))
    
    # Save CSV for plotting
    df.to_csv('ablation_results.csv', index=False)
