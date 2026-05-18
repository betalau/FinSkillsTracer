import sys
import os
import json
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ingestion.normalization import DataIngestionEngine
from mining.mining_engine import SkillMiningEngine
from parameterization.generalization import ParameterizationEngine
from evaluation.evaluator import SkillEvaluator

def run_experiment(subset_size=None):
    print(f"\n--- Running Experiment (Subset: {subset_size if subset_size else 'Full'}) ---")
    
    # 1. Ingestion
    ingest = DataIngestionEngine('../data/tool_traces.parquet', '../data/scores.parquet')
    golden_df = ingest.filter_golden_paths()
    
    if subset_size:
        # Filter by query index (0 to subset_size-1)
        golden_df = golden_df[golden_df['index'] < subset_size]
        
    normalized = ingest.normalize_traces(golden_df)
    print(f"Normalized {len(normalized)} traces.")

    # 2. Mining
    miner = SkillMiningEngine(normalized)
    
    print("Mining Frequent Sequences...")
    freq_skills = miner.mine_frequent_sequences(min_support=max(5, len(normalized)//100))
    
    print("Mining Semantic Clusters...")
    cluster_skills = miner.mine_semantic_clusters(n_clusters=10)
    
    print("Mining Hierarchical Skills...")
    hier_skills = miner.mine_hierarchical_skills()

    all_discovered = freq_skills + cluster_skills + hier_skills
    print(f"Total skills discovered: {len(all_discovered)}")

    # 3. Parameterization (on top 5 frequent skills)
    param_engine = ParameterizationEngine(normalized)
    skills_with_templates = []
    for skill in freq_skills[:5]:
        template = param_engine.extract_variable_mappings(skill['sequence'])
        skill['template'] = template
        skills_with_templates.append(skill)

    # 4. Evaluation
    evaluator = SkillEvaluator(normalized, all_discovered)
    quality_metrics = evaluator.evaluate_quality()
    
    print(f"Overall Coverage: {quality_metrics['overall_coverage']:.2%}")
    
    return {
        "subset_size": subset_size or "Full",
        "num_traces": len(normalized),
        "num_skills": len(all_discovered),
        "coverage": quality_metrics['overall_coverage']
    }

if __name__ == "__main__":
    results = []
    # Scaling Law Experiment
    for size in [100, 250, 500]:
        res = run_experiment(size)
        results.append(res)
    
    # Full run
    # res_full = run_experiment(None)
    # results.append(res_full)
    
    # Save results
    with open("experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nExperiment complete. Results saved to experiment_results.json")
    
    # Output as a nice table
    df_results = pd.DataFrame(results)
    print("\nScaling Law Summary:")
    print(df_results)
