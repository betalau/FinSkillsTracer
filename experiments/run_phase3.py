"""
Phase 3 Optimization Experiment Runner.

Runs P3.1 (tree hierarchy), P3.2 (parameterization), P3.3 (real evolution)
and produces comparison report vs Phase 2 baseline.
"""
import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ingestion.normalization import DataIngestionEngine
from mining.fsp import FSPMiner
from mining.semantic import SemanticMiner
from mining.hierarchical import HierarchicalMiner
from mining.trace2skill import Trace2SkillMiner
from mining.skillclaw import SkillClawEvolver, EvolutionarySkillClaw
from mining.consensus import ConsensusAnalyzer
from mining.error_miner import ErrorPatternMiner
from evaluation.llm_evaluator import split_by_query, SkillEvaluatorV2
from evaluation.utility_evaluator import UtilityEvaluator
from agents.agent_simulator import TraceEnvironment
from parameterization.generalization import ParameterizationEngine


def main():
    print("=" * 70)
    print("PHASE 3: DEEP OPTIMIZATION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1/9] Loading data...")
    ingest = DataIngestionEngine(
        '../data/tool_traces.parquet',
        '../data/scores.parquet',
        '../data/questions.parquet'
    )

    all_df = ingest.load_all_traces()
    all_normalized = ingest.normalize_traces(all_df)
    print(f"  Loaded {len(all_normalized)} total traces.")

    golden_df = ingest.filter_golden_paths()
    golden_normalized = ingest.normalize_traces(golden_df)

    train_traces, test_traces, train_q, test_q = split_by_query(
        golden_normalized, test_ratio=0.2, seed=42
    )
    print(f"  Split: {len(train_traces)} train / {len(test_traces)} test traces.")

    # ------------------------------------------------------------------
    # 2. Mine skills (same 4 strategies as Phase 1/2)
    # ------------------------------------------------------------------
    print("\n[2/9] Mining skills...")
    skills_a = FSPMiner(train_traces).mine(min_support=10)
    skills_b = SemanticMiner(train_traces).mine(n_clusters=10)
    skills_c = HierarchicalMiner(train_traces).mine(min_support=3)
    skills_d = Trace2SkillMiner(train_traces).mine()
    all_raw = skills_a + skills_b + skills_c + skills_d
    print(f"  Raw skills: A={len(skills_a)} B={len(skills_b)} C={len(skills_c)} D={len(skills_d)} = {len(all_raw)} total")

    # Consensus analysis
    consensus = ConsensusAnalyzer(train_traces)
    all_raw = consensus.analyze_skills(all_raw)
    stats = consensus.get_consensus_stats(all_raw)
    print(f"  Consensus: mean={stats['mean_consensus_score']:.4f}, high={stats['high_consensus_skills']}/{stats['total_skills']}")

    # ------------------------------------------------------------------
    # 3. P3.3 — Evolutionary SkillClaw (real evolution)
    # ------------------------------------------------------------------
    print("\n[3/9] P3.3 — Running Evolutionary SkillClaw...")
    evolver = EvolutionarySkillClaw(all_raw, traces=train_traces)
    evolved_skills = evolver.evolve(rounds=5)
    print(f"  Evolved skills: {len(evolved_skills)} (from {len(all_raw)} raw)")

    # Also run V1 for comparison
    v1_evolver = SkillClawEvolver(all_raw)
    v1_skills = v1_evolver.evolve()
    print(f"  V1 baseline skills: {len(v1_skills)}")

    # ------------------------------------------------------------------
    # 4. P3.1 — Hierarchical Tree Structure
    # ------------------------------------------------------------------
    print("\n[4/9] P3.1 — Building hierarchical tree...")
    hier_miner = HierarchicalMiner(train_traces)
    tree_nodes = hier_miner.mine_tree(min_support=3)
    tree_summary = hier_miner.get_tree_summary(tree_nodes)

    total_tree_nodes = len(tree_nodes)
    cat_roots = [n for n in tree_nodes if n.depth == 0]
    flat_nodes = [n for n in tree_nodes if n.depth > 0]
    print(f"  Tree nodes: {total_tree_nodes} ({len(cat_roots)} categories + {len(flat_nodes)} skills)")
    print(f"  Categories: {tree_summary['categories']}")
    print(f"  By category: {tree_summary['by_category']}")
    print(f"  Max depth: {tree_summary['max_depth']}")
    print(f"  Structure:\n{tree_summary['tree_structure']}")

    # ------------------------------------------------------------------
    # 5. P3.2 — Parameterization Engine
    # ------------------------------------------------------------------
    print("\n[5/9] P3.2 — Generating parameterized templates...")
    param_engine = ParameterizationEngine(all_normalized)
    templates = param_engine.generate_all_templates(evolved_skills, top_n=15)

    # Data flow summaries for top templates
    data_flow_summaries = []
    for tmpl in templates[:10]:
        seq = tmpl.sequence
        if len(seq) >= 2:
            summary = param_engine.get_data_flow_summary(seq)
            data_flow_summaries.append(summary)

    print(f"  Templates generated: {len(templates)}")
    total_flows = sum(len(s.get("data_flow_edges", [])) for s in data_flow_summaries)
    print(f"  Total data flow edges detected: {total_flows}")

    # ------------------------------------------------------------------
    # 6. Quality evaluation (Phase 1 metrics)
    # ------------------------------------------------------------------
    print("\n[6/9] Computing quality metrics...")
    evaluator = SkillEvaluatorV2(train_traces, test_traces, evolved_skills, train_q, test_q)
    quality_v3 = evaluator.calculate_metrics()

    # V1 baseline for comparison
    evaluator_v1 = SkillEvaluatorV2(train_traces, test_traces, v1_skills, train_q, test_q)
    quality_v1 = evaluator_v1.calculate_metrics()

    print(f"  P3.3  Coverage (test): {quality_v3['coverage_test']:.2%}")
    print(f"  V1    Coverage (test): {quality_v1['coverage_test']:.2%}")
    print(f"  P3.3  Uniqueness:      {quality_v3['uniqueness']:.2%}")
    print(f"  V1    Uniqueness:      {quality_v1['uniqueness']:.2%}")
    print(f"  P3.3  High-consensus:  {quality_v3['consensus']['high_consensus_count']}/{quality_v3['skill_count']}")

    # ------------------------------------------------------------------
    # 7. Error mining & real confidence
    # ------------------------------------------------------------------
    print("\n[7/9] Computing real confidence (error-miner)...")
    error_miner = ErrorPatternMiner(all_normalized)
    error_stats = error_miner.get_error_statistics()

    for skill in evolved_skills:
        seq = skill.trigger_pattern.get("tool_sequence", [])
        if seq:
            conf_data = error_miner.compute_confidence(seq)
            skill.confidence = conf_data["overall_confidence"]

    real_confidences = [s.confidence for s in evolved_skills]
    mean_confidence = sum(real_confidences) / len(real_confidences) if real_confidences else 0
    quality_v3["confidence"] = round(mean_confidence, 4)
    quality_v3["confidence_source"] = "error_miner_P3.3"
    print(f"  Mean real confidence: {mean_confidence:.4f}")
    print(f"  Tool error rate: {error_stats['tool_error_rate']:.4f}")

    # ------------------------------------------------------------------
    # 8. Agent-vs-Agent benchmark
    # ------------------------------------------------------------------
    print("\n[8/9] Running Agent-vs-Agent benchmark (P3.3 skills)...")
    env = TraceEnvironment(all_normalized)

    test_query_data = []
    for t in test_traces:
        test_query_data.append({
            "original_index": t.get("original_index", 0),
            "question": t.get("question", f"Query {t.get('original_index', 0)}"),
            "expected_value": t.get("expected_value", ""),
            "configuration": t.get("configuration", ""),
        })
    seen_idx = set()
    unique_test_queries = []
    for q in test_query_data:
        if q["original_index"] not in seen_idx:
            seen_idx.add(q["original_index"])
            unique_test_queries.append(q)

    utility_eval = UtilityEvaluator(env, evolved_skills, unique_test_queries)
    benchmark_v3 = utility_eval.run_benchmark(num_queries=min(50, len(unique_test_queries)))

    # ------------------------------------------------------------------
    # 9. Compile report
    # ------------------------------------------------------------------
    print("\n[9/9] Compiling Phase 3 report...")

    # P3.1 summary
    p31_summary = {
        "total_nodes": total_tree_nodes,
        "category_nodes": len(cat_roots),
        "skill_nodes": len(flat_nodes),
        "max_depth": tree_summary["max_depth"],
        "avg_depth": tree_summary["avg_depth"],
        "categories": tree_summary["categories"],
        "by_category": tree_summary["by_category"],
    }

    # P3.2 summary
    p32_summary = {
        "templates_generated": len(templates),
        "total_data_flow_edges": total_flows,
        "top_templates": [
            {
                "skill_id": t.skill_id,
                "sequence": t.sequence,
                "variable_count": t.variable_count,
                "constant_count": t.constant_count,
                "instance_count": t.instance_count,
            }
            for t in templates[:10]
        ],
        "data_flow_summaries": data_flow_summaries[:5],
    }

    # P3.3 summary
    skill_reduction_pct = (1 - len(evolved_skills) / max(1, len(v1_skills))) * 100
    p33_summary = {
        "raw_skills_count": len(all_raw),
        "v1_skills_count": len(v1_skills),
        "evolved_skills_count": len(evolved_skills),
        "skill_reduction_pct": round(skill_reduction_pct, 1),
        "rounds": 5,
        "operators": ["merge_namespace", "collapse_loop", "add_tool", "delete_tool", "replace_tool", "crossover"],
    }

    # Comparative metrics
    comparative = benchmark_v3.get("comparative_metrics", {})

    report = {
        "phase": "Phase 3 — Deep Optimization",
        "p31_hierarchical_tree": p31_summary,
        "p32_parameterization": p32_summary,
        "p33_evolution": p33_summary,
        "quality_metrics": {
            "phase2_baseline": {
                "skill_count": quality_v1["skill_count"],
                "coverage_test": quality_v1["coverage_test"],
                "uniqueness": quality_v1["uniqueness"],
                "consensus_mean": quality_v1["consensus"]["mean_consensus_score"],
                "high_consensus_rate": quality_v1["consensus"]["high_consensus_count"] / max(1, quality_v1["skill_count"]),
            },
            "phase3_optimized": {
                "skill_count": quality_v3["skill_count"],
                "coverage_test": quality_v3["coverage_test"],
                "coverage_train": quality_v3["coverage_train"],
                "uniqueness": quality_v3["uniqueness"],
                "confidence": quality_v3["confidence"],
                "confidence_source": quality_v3.get("confidence_source", ""),
                "consensus_mean": quality_v3["consensus"]["mean_consensus_score"],
                "high_consensus_rate": quality_v3["consensus"]["high_consensus_count"] / max(1, quality_v3["skill_count"]),
                "model_diversity": quality_v3["model_diversity"],
            },
            "improvements": {
                "uniqueness_delta": round(quality_v3["uniqueness"] - quality_v1["uniqueness"], 4),
                "coverage_test_delta": round(quality_v3["coverage_test"] - quality_v1["coverage_test"], 4),
                "skill_reduction_pct": round(skill_reduction_pct, 1),
                "consensus_mean_delta": round(quality_v3["consensus"]["mean_consensus_score"] - quality_v1["consensus"]["mean_consensus_score"], 4),
            },
        },
        "error_analysis": {
            "total_traces_analyzed": error_stats["total_traces"],
            "tool_error_rate": error_stats["tool_error_rate"],
            "traces_with_tool_errors": error_stats["traces_with_tool_errors"],
            "traces_with_answer_errors": error_stats["traces_with_answer_errors"],
            "most_error_prone_tools": error_stats["most_error_prone_tools"],
        },
        "utility_benchmark": {
            "num_test_queries": benchmark_v3["num_queries"],
            "per_agent_stats": benchmark_v3["agent_stats"],
            "comparative_vs_baseline": comparative,
        },
    }

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 3 RESULTS SUMMARY")
    print("=" * 60)

    print(f"\n  P3.1 — Hierarchical Tree:")
    print(f"    Nodes: {p31_summary['total_nodes']} ({p31_summary['category_nodes']} categories + "
          f"{p31_summary['skill_nodes']} skills), Max Depth: {p31_summary['max_depth']}")

    print(f"\n  P3.2 — Parameterization:")
    print(f"    Templates: {p32_summary['templates_generated']}, "
          f"Data flow edges: {p32_summary['total_data_flow_edges']}")

    print(f"\n  P3.3 — Evolution:")
    print(f"    Skills: {p33_summary['raw_skills_count']} raw → "
          f"{p33_summary['v1_skills_count']} V1 → "
          f"{p33_summary['evolved_skills_count']} evolved "
          f"({p33_summary['skill_reduction_pct']:+.1f}%)")

    print(f"\n  Quality Improvements (vs Phase 2 V1 baseline):")
    imp = report["quality_metrics"]["improvements"]
    print(f"    Coverage (test):  {quality_v1['coverage_test']:.2%} → {quality_v3['coverage_test']:.2%} "
          f"({imp['coverage_test_delta']:+.2%})")
    print(f"    Uniqueness:       {quality_v1['uniqueness']:.2%} → {quality_v3['uniqueness']:.2%} "
          f"({imp['uniqueness_delta']:+.2%})")
    print(f"    Consensus mean:   {quality_v1['consensus']['mean_consensus_score']:.4f} → "
          f"{quality_v3['consensus']['mean_consensus_score']:.4f} "
          f"({imp['consensus_mean_delta']:+.4f})")
    print(f"    Skill reduction:  {imp['skill_reduction_pct']:+.1f}%")

    for agent_name in ["SkillAwareAgent", "ReActAgent", "RandomAgent"]:
        stats = benchmark_v3["agent_stats"].get(agent_name, {})
        tag = ""
        if agent_name == "SkillAwareAgent":
            tag = "  ← OUR AGENT (P3.3 evolved)"
        elif agent_name == "ReActAgent":
            tag = "  ← Baseline 1"
        elif agent_name == "RandomAgent":
            tag = "  ← Baseline 2"
        print(f"\n  {agent_name}{tag}")
        print(f"    Task Success Rate:  {stats.get('task_success_rate', 0):.2%}")
        print(f"    Avg Tool Calls:     {stats.get('avg_tool_calls', 0)}")
        print(f"    Avg Duration (ms):  {stats.get('avg_duration_ms', 0):.0f}")

    print(f"\n  Comparative vs ReActAgent baseline:")
    print(f"    Latency Reduction:    {comparative.get('latency_reduction', 0):+.1%}")
    print(f"    Tool Call Reduction:  {comparative.get('tool_call_reduction', 0):+.1%}")
    print(f"    Error Rate Reduction: {comparative.get('error_rate_reduction', 0):+.1%}")

    print("=" * 60)

    # Save report
    output_path = "../phase3_results.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nFull report saved to {output_path}")

    return report


if __name__ == "__main__":
    main()
