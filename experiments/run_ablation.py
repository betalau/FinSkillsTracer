"""
Ablation experiment with proper train/test split (Phase 1.2).

Evaluates each mining strategy (A/B/C/D/E) independently on a held-out test set
to produce honest coverage and consensus metrics.
"""
import sys
import os
import json
import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ingestion.normalization import DataIngestionEngine
from mining.fsp import FSPMiner
from mining.semantic import SemanticMiner
from mining.hierarchical import HierarchicalMiner
from mining.trace2skill import Trace2SkillMiner
from mining.skillclaw import SkillClawEvolver
from mining.consensus import ConsensusAnalyzer
from evaluation.llm_evaluator import SkillEvaluatorV2, split_by_query


def run_ablation(methods, train_sizes):
    all_results = []

    ingest = DataIngestionEngine('../data/tool_traces.parquet', '../data/scores.parquet')

    for size in train_sizes:
        print(f"\n{'='*60}")
        print(f"Train Size: {size} queries")
        print(f"{'='*60}")

        golden_df = ingest.filter_golden_paths()
        subset_df = golden_df[golden_df['index'] < size]
        normalized = ingest.normalize_traces(subset_df)

        # Split by query index
        train_traces, test_traces, train_q, test_q = split_by_query(
            normalized, test_ratio=0.2, seed=42
        )
        print(f"  Split: {len(train_traces)} train, {len(test_traces)} test "
              f"({len(train_q)}/{len(test_q)} queries)")

        for method in methods:
            print(f"  Method {method}...")
            skills = []

            if method == 'A':
                miner = FSPMiner(train_traces)
                skills = miner.mine(min_support=max(2, size // 50))
            elif method == 'B':
                miner = SemanticMiner(train_traces)
                skills = miner.mine(n_clusters=min(10, len(train_traces)),
                                    use_llm_descriptions=False)  # skip LLM in ablation
            elif method == 'C':
                miner = HierarchicalMiner(train_traces)
                skills = miner.mine()
            elif method == 'D':
                miner = Trace2SkillMiner(train_traces)
                skills = miner.mine()
            elif method == 'E':
                all_raw = []
                all_raw += FSPMiner(train_traces).mine(min_support=max(2, size // 50))
                all_raw += SemanticMiner(train_traces).mine(n_clusters=min(5, len(train_traces)),
                                                            use_llm_descriptions=False)
                evolver = SkillClawEvolver(all_raw)
                skills = evolver.evolve()

            # Consensus analysis
            consensus = ConsensusAnalyzer(train_traces)
            skills = consensus.analyze_skills(skills)

            # Evaluate on test set
            evaluator = SkillEvaluatorV2(
                train_traces=train_traces,
                test_traces=test_traces,
                skills=skills,
                train_queries=train_q,
                test_queries=test_q,
            )
            metrics = evaluator.calculate_metrics()

            metrics['method'] = method
            metrics['train_size'] = size
            all_results.append(metrics)

            print(f"    Skills: {metrics['skill_count']}, "
                  f"Coverage(test): {metrics['coverage_test']:.2%}, "
                  f"Uniqueness: {metrics['uniqueness']:.2%}, "
                  f"Consensus rate: {metrics['consensus']['high_consensus_rate']:.1%}")

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
    print("\n" + "=" * 70)
    print("ABLATION RESULTS — Coverage (test set)")
    print("=" * 70)
    pivot_cov = df.pivot(index='train_size', columns='method', values='coverage_test')
    print(pivot_cov.to_string())

    print("\nABLATION RESULTS — Uniqueness")
    print("=" * 70)
    pivot_uniq = df.pivot(index='train_size', columns='method', values='uniqueness')
    print(pivot_uniq.to_string())

    print("\nABLATION RESULTS — Consensus Rate")
    print("=" * 70)
    try:
        pivot_cons = df.pivot(index='train_size', columns='method',
                              values='consensus')
        # Extract high_consensus_rate from nested dict
        for _, row in df.iterrows():
            c = row['consensus']
            if isinstance(c, dict):
                row['high_consensus_rate'] = c.get('high_consensus_rate', 0)
        pivot_cons = df.pivot(index='train_size', columns='method',
                              values='high_consensus_rate')
        print(pivot_cons.to_string())
    except Exception:
        print("(consensus details in JSON)")

    df.to_csv('ablation_results.csv', index=False)
    print(f"\nFull results saved to ablation_results.json / ablation_results.csv")
