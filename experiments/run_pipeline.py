"""
Scaling Law experiment — Phase 1 enhanced with train/test split and consensus.
"""
import sys
import os
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ingestion.normalization import DataIngestionEngine
from mining.mining_engine import SkillMiningEngine
from parameterization.generalization import ParameterizationEngine
from evaluation.llm_evaluator import split_by_query, SkillEvaluatorV2


def run_experiment(subset_size=None):
    print(f"\n--- Running Experiment (Subset: {subset_size if subset_size else 'Full'}) ---")

    ingest = DataIngestionEngine('../data/tool_traces.parquet', '../data/scores.parquet')
    golden_df = ingest.filter_golden_paths()

    if subset_size:
        golden_df = golden_df[golden_df['index'] < subset_size]

    normalized = ingest.normalize_traces(golden_df)

    # Phase 1.2: proper train/test split
    train_traces, test_traces, train_q, test_q = split_by_query(
        normalized, test_ratio=0.2, seed=42
    )
    print(f"Normalized {len(normalized)} traces "
          f"(train={len(train_traces)}, test={len(test_traces)}).")

    # Mining on train only
    miner = SkillMiningEngine(train_traces)

    print("Mining Frequent Sequences...")
    freq_skills = miner.mine_frequent_sequences(min_support=max(5, len(train_traces) // 100))

    print("Mining Semantic Clusters...")
    cluster_skills = miner.mine_semantic_clusters(n_clusters=10)

    print("Mining Hierarchical Skills...")
    hier_skills = miner.mine_hierarchical_skills()

    all_discovered = freq_skills + cluster_skills + hier_skills

    # Convert V1 skill dicts to Skill objects for V2 evaluator
    from models import Skill
    skill_objects = []
    for s in all_discovered:
        seq = s.get('sequence') or s.get('representative_sequence', [])
        skill_objects.append(Skill(
            skill_id=f"v1_{len(skill_objects)}",
            name=f"V1 Skill: {' -> '.join(seq[:3])}",
            description=f"Mined via {s.get('strategy', 'unknown')}",
            trigger_pattern={"tool_sequence": seq, "semantic_constraint": None},
            preconditions=[],
            postconditions=[],
            failure_modes=[],
            support=s.get('support', 0),
            confidence=1.0,
            examples=[]
        ))

    # Evaluate on test set
    evaluator = SkillEvaluatorV2(train_traces, test_traces, skill_objects, train_q, test_q)
    metrics = evaluator.calculate_metrics()

    print(f"  Skills: {metrics['skill_count']}, "
          f"Coverage(test): {metrics['coverage_test']:.2%}, "
          f"Uniqueness: {metrics['uniqueness']:.2%}")

    # Parameterization on top skills
    param_engine = ParameterizationEngine(train_traces)
    skills_with_templates = []
    for skill in freq_skills[:5]:
        seq = skill.get('sequence') or skill.get('representative_sequence', [])
        template = param_engine.extract_variable_mappings(seq)
        skill['template'] = template
        skills_with_templates.append(skill)

    return {
        "subset_size": subset_size or "Full",
        "num_traces_total": len(normalized),
        "num_traces_train": len(train_traces),
        "num_traces_test": len(test_traces),
        "num_skills": metrics['skill_count'],
        "coverage_train": metrics['coverage_train'],
        "coverage_test": metrics['coverage_test'],
        "uniqueness": metrics['uniqueness'],
        "confidence": metrics['confidence'],
    }


if __name__ == "__main__":
    results = []
    for size in [100, 250, 500]:
        res = run_experiment(size)
        results.append(res)

    with open("experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nExperiment complete. Results saved to experiment_results.json")

    df_results = pd.DataFrame(results)
    print("\nScaling Law Summary (with test-set metrics):")
    print(df_results.to_string())
