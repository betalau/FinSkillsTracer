"""
FinSkillsTracer V2 — Phase 1 Enhanced Pipeline.

Key improvements over the original demo:
  P1.1 — Multi-Model Consensus Mining: skills scored by cross-model agreement
  P1.2 — Train/Test Split + Uniqueness: real coverage on held-out queries
  P1.3 — LLM-Generated Skill Descriptions (via EFundGPT for semantic clusters)
"""
import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

from ingestion.normalization import DataIngestionEngine
from mining.fsp import FSPMiner
from mining.semantic import SemanticMiner
from mining.hierarchical import HierarchicalMiner
from mining.trace2skill import Trace2SkillMiner
from mining.skillclaw import SkillClawEvolver
from mining.consensus import ConsensusAnalyzer
from evaluation.llm_evaluator import SkillEvaluatorV2, split_by_query


def main():
    # ------------------------------------------------------------------
    # 1. Load & normalize
    # ------------------------------------------------------------------
    ingest = DataIngestionEngine(
        '../data/tool_traces.parquet',
        '../data/scores.parquet',
        '../data/questions.parquet'
    )
    golden_df = ingest.filter_golden_paths()
    normalized = ingest.normalize_traces(golden_df)
    print(f"[1/6] Loaded {len(normalized)} golden traces.")

    # ------------------------------------------------------------------
    # 2. Train / Test split (by query index to prevent leakage)
    # ------------------------------------------------------------------
    train_traces, test_traces, train_queries, test_queries = split_by_query(
        normalized, test_ratio=0.2, seed=42
    )
    print(f"[2/6] Split: {len(train_traces)} train traces ({len(train_queries)} queries), "
          f"{len(test_traces)} test traces ({len(test_queries)} queries).")

    # ------------------------------------------------------------------
    # 3. Mine skills on TRAIN data only
    # ------------------------------------------------------------------
    print("[3/6] Mining skills with all strategies (on train set)...")
    skills_a = FSPMiner(train_traces).mine(min_support=10)
    skills_b = SemanticMiner(train_traces).mine(n_clusters=10)
    skills_c = HierarchicalMiner(train_traces).mine()
    skills_d = Trace2SkillMiner(train_traces).mine()

    all_raw = skills_a + skills_b + skills_c + skills_d
    print(f"  Raw skills discovered: {len(all_raw)} "
          f"(A:{len(skills_a)} B:{len(skills_b)} C:{len(skills_c)} D:{len(skills_d)})")

    # ------------------------------------------------------------------
    # 4. Consensus analysis (P1.1)
    # ------------------------------------------------------------------
    print("[4/6] Computing cross-model consensus scores...")
    consensus = ConsensusAnalyzer(train_traces)
    all_raw = consensus.analyze_skills(all_raw)
    stats = consensus.get_consensus_stats(all_raw)
    print(f"  Consensus: mean={stats['mean_consensus_score']}, "
          f"high_consensus={stats['high_consensus_skills']}/{stats['total_skills']} "
          f"({stats['consensus_rate']:.1%})")

    # ------------------------------------------------------------------
    # 5. Evolve with SkillClaw (consensus-aware)
    # ------------------------------------------------------------------
    print("[5/6] Evolving skills with SkillClaw...")
    evolver = SkillClawEvolver(all_raw)
    final_skills = evolver.evolve()
    print(f"  Final skills after evolution: {len(final_skills)}")

    # ------------------------------------------------------------------
    # 6. Evaluate on TEST set (P1.2)
    # ------------------------------------------------------------------
    print("[6/6] Evaluating on test set...")
    evaluator = SkillEvaluatorV2(
        train_traces=train_traces,
        test_traces=test_traces,
        skills=final_skills,
        train_queries=train_queries,
        test_queries=test_queries,
    )
    metrics = evaluator.calculate_metrics()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 1 EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Skills discovered:     {metrics['skill_count']}")
    print(f"  Train / Test traces:   {metrics['train_traces']} / {metrics['test_traces']}")
    print(f"  Train / Test queries:  {metrics['train_queries']} / {metrics['test_queries']}")
    print(f"  Coverage (train):      {metrics['coverage_train']:.2%}")
    print(f"  Coverage (test):       {metrics['coverage_test']:.2%}  <-- REAL metric")
    print(f"  Confidence:            {metrics['confidence']:.2%}")
    print(f"  Uniqueness:            {metrics['uniqueness']:.2%}  (>0.7 is good)")
    print(f"  High-consensus skills: {metrics['consensus']['high_consensus_count']}/{metrics['skill_count']}")
    print(f"  Mean consensus score:  {metrics['consensus']['mean_consensus_score']:.4f}")
    print(f"  Models covered:        {metrics['model_diversity']['total_configs_covered']}/14")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    output_skills = "../skills_bank_v2.json"
    with open(output_skills, "w") as f:
        json.dump([s.to_dict() for s in final_skills], f, indent=2, ensure_ascii=False)
    print(f"\nSkills saved to {output_skills}")

    output_metrics = "../evaluation_metrics.json"
    with open(output_metrics, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to {output_metrics}")


if __name__ == "__main__":
    main()
