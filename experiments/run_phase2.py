"""
Phase 2 Utility Benchmark Experiment Runner.

Runs Agent-vs-Agent comparison and produces the final evaluation report.
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
from mining.skillclaw import SkillClawEvolver
from mining.consensus import ConsensusAnalyzer
from mining.error_miner import ErrorPatternMiner
from evaluation.llm_evaluator import split_by_query, SkillEvaluatorV2
from evaluation.utility_evaluator import UtilityEvaluator
from agents.agent_simulator import TraceEnvironment


def main():
    print("=" * 70)
    print("PHASE 2: UTILITY EVALUATION — Agent-vs-Agent Benchmark")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load ALL traces (including errors) for confidence + agent sim
    # ------------------------------------------------------------------
    print("\n[1/8] Loading data...")
    ingest = DataIngestionEngine(
        '../data/tool_traces.parquet',
        '../data/scores.parquet',
        '../data/questions.parquet'
    )

    # All traces for error mining and agent environment
    all_df = ingest.load_all_traces()
    all_normalized = ingest.normalize_traces(all_df)
    print(f"  Loaded {len(all_normalized)} total traces (including error traces).")

    # Golden-only for skill mining
    golden_df = ingest.filter_golden_paths()
    golden_normalized = ingest.normalize_traces(golden_df)

    # Train/test split
    train_traces, test_traces, train_q, test_q = split_by_query(
        golden_normalized, test_ratio=0.2, seed=42
    )
    print(f"  Split: {len(train_traces)} train / {len(test_traces)} test traces.")

    # ------------------------------------------------------------------
    # 2. Mine skills (same as Phase 1)
    # ------------------------------------------------------------------
    print("\n[2/8] Mining skills...")
    skills_a = FSPMiner(train_traces).mine(min_support=10)
    skills_b = SemanticMiner(train_traces).mine(n_clusters=10)
    skills_c = HierarchicalMiner(train_traces).mine()
    skills_d = Trace2SkillMiner(train_traces).mine()
    all_raw = skills_a + skills_b + skills_c + skills_d

    # Consensus
    consensus = ConsensusAnalyzer(train_traces)
    all_raw = consensus.analyze_skills(all_raw)

    # Evolve
    evolver = SkillClawEvolver(all_raw)
    final_skills = evolver.evolve()
    print(f"  Final skills: {len(final_skills)}")

    # ------------------------------------------------------------------
    # 3. Phase 1 evaluation (quality metrics)
    # ------------------------------------------------------------------
    print("\n[3/8] Phase 1 quality evaluation...")
    evaluator = SkillEvaluatorV2(train_traces, test_traces, final_skills, train_q, test_q)
    quality_metrics = evaluator.calculate_metrics()
    print(f"  Coverage (test): {quality_metrics['coverage_test']:.2%}")
    print(f"  Uniqueness: {quality_metrics['uniqueness']:.2%}")
    print(f"  High-consensus: {quality_metrics['consensus']['high_consensus_count']}/{quality_metrics['skill_count']}")

    # ------------------------------------------------------------------
    # 4. Error mining & real confidence (P2.1)
    # ------------------------------------------------------------------
    print("\n[4/8] Mining error patterns and computing real confidence...")
    error_miner = ErrorPatternMiner(all_normalized)

    # Update skill confidence with real error data
    for skill in final_skills:
        seq = skill.trigger_pattern.get("tool_sequence", [])
        if seq:
            conf_data = error_miner.compute_confidence(seq)
            skill.confidence = conf_data["overall_confidence"]
            if conf_data["tool_error_count"] > 0 or conf_data["answer_error_count"] > 0:
                skill.failure_modes.append(
                    f"Tool errors: {conf_data['tool_error_count']}, "
                    f"Answer errors: {conf_data['answer_error_count']}"
                )

    # Get global error stats
    error_stats = error_miner.get_error_statistics()
    print(f"  Traces with tool errors: {error_stats['traces_with_tool_errors']}")
    print(f"  Traces with answer errors: {error_stats['traces_with_answer_errors']}")
    print(f"  Most error-prone tools: {list(error_stats['most_error_prone_tools'].keys())[:5]}")

    # P2.1 — Per-skill confidence breakdown (tool vs answer error)
    per_skill_confidence_detail = {}
    for skill in final_skills[:30]:  # top 30 for detail
        seq = skill.trigger_pattern.get("tool_sequence", [])
        if seq:
            conf_data = error_miner.compute_confidence(seq)
            per_skill_confidence_detail[skill.skill_id] = {
                "skill_name": skill.name,
                "sequence": seq,
                "tool_confidence": conf_data["tool_confidence"],
                "answer_confidence": conf_data["answer_confidence"],
                "overall_confidence": conf_data["overall_confidence"],
                "total_instances": conf_data["total_instances"],
                "tool_errors": conf_data["tool_error_count"],
                "answer_errors": conf_data["answer_error_count"],
            }

    # P2.1 — Mine failure patterns for top skills
    failure_patterns_by_skill = {}
    for skill in final_skills[:10]:
        seq = skill.trigger_pattern.get("tool_sequence", [])
        if seq:
            failures = error_miner.mine_failure_modes(seq)
            # Summarize by type
            tool_failures = [f for f in failures if f["type"] == "tool_error"]
            answer_failures = [f for f in failures if f["type"] == "answer_error"]
            failure_patterns_by_skill[skill.skill_id] = {
                "skill_name": skill.name,
                "sequence": seq,
                "total_failures": len(failures),
                "tool_level_failures": len(tool_failures),
                "answer_level_failures": len(answer_failures),
                "sample_tool_errors": [
                    {"step": f["step_in_skill"], "tool": f["tool_name"]}
                    for f in tool_failures[:5]
                ],
                "affected_configs": list(set(
                    f["configuration"] for f in tool_failures
                ))[:5],
            }

    # Compute aggregate real confidence across all skills
    all_confidences = [
        s.confidence for s in final_skills
        if s.confidence != 1.0 or hasattr(s, '_error_checked')
    ]
    # Use error-derived confidence values (set in step 4 above)
    real_confidences = [s.confidence for s in final_skills]
    mean_real_confidence = sum(real_confidences) / len(real_confidences) if real_confidences else 1.0

    # Override quality metrics confidence with error-derived value
    quality_metrics_v2 = evaluator.calculate_metrics()
    quality_metrics_v2["confidence"] = round(mean_real_confidence, 4)
    quality_metrics_v2["confidence_source"] = "error_miner_on_all_7000_traces"
    # Replace per-skill confidence with error-derived values
    quality_metrics_v2["per_skill_confidence"] = {
        sid: d["overall_confidence"]
        for sid, d in per_skill_confidence_detail.items()
    }

    # ------------------------------------------------------------------
    # 5. Agent-vs-Agent benchmark (P2.2)
    # ------------------------------------------------------------------
    print("\n[5/8] Running Agent-vs-Agent benchmark...")
    env = TraceEnvironment(all_normalized)

    # Prepare test queries with expected values
    test_query_data = []
    for t in test_traces:
        test_query_data.append({
            "original_index": t.get("original_index", 0),
            "question": t.get("question", f"Query {t.get('original_index', 0)}"),
            "expected_value": t.get("expected_value", ""),
            "configuration": t.get("configuration", ""),
        })
    # Deduplicate by query index
    seen_idx = set()
    unique_test_queries = []
    for q in test_query_data:
        if q["original_index"] not in seen_idx:
            seen_idx.add(q["original_index"])
            unique_test_queries.append(q)

    utility_eval = UtilityEvaluator(env, final_skills, unique_test_queries)
    benchmark_results = utility_eval.run_benchmark(num_queries=min(50, len(unique_test_queries)))

    # ------------------------------------------------------------------
    # 6. LLM skill quality assessment (P2.3)
    # ------------------------------------------------------------------
    print("\n[6/8] Evaluating skill quality via EFundGPT...")
    skill_assessments = utility_eval.evaluate_skill_quality_llm(top_n=10)

    # ------------------------------------------------------------------
    # 7. Compile full report
    # ------------------------------------------------------------------
    print("\n[7/8] Compiling final report...")
    comparative = benchmark_results.get("comparative_metrics", {})

    report = {
        "phase": "Phase 1 + Phase 2",
        "quality_metrics": {
            "skill_count": quality_metrics_v2["skill_count"],
            "coverage_test": quality_metrics_v2["coverage_test"],
            "coverage_train": quality_metrics_v2["coverage_train"],
            "uniqueness": quality_metrics_v2["uniqueness"],
            "confidence": quality_metrics_v2["confidence"],
            "confidence_source": quality_metrics_v2.get("confidence_source", ""),
            "consensus": quality_metrics_v2["consensus"],
            "model_diversity": quality_metrics_v2["model_diversity"],
            "train_test_split": {
                "train_traces": quality_metrics_v2["train_traces"],
                "test_traces": quality_metrics_v2["test_traces"],
                "train_queries": quality_metrics_v2["train_queries"],
                "test_queries": quality_metrics_v2["test_queries"],
            },
        },
        "error_analysis": {
            "total_traces_analyzed": error_stats["total_traces"],
            "tool_error_rate": error_stats["tool_error_rate"],
            "traces_with_tool_errors": error_stats["traces_with_tool_errors"],
            "traces_with_answer_errors": error_stats["traces_with_answer_errors"],
            "total_tool_error_calls": error_stats["total_tool_error_calls"],
            "most_error_prone_tools": error_stats["most_error_prone_tools"],
            "error_by_config": error_stats["error_by_config"],
            # P2.1 gap fix: per-skill confidence breakdown
            "per_skill_confidence_top30": per_skill_confidence_detail,
            # P2.1 gap fix: failure patterns for top skills
            "failure_patterns_top10": failure_patterns_by_skill,
        },
        "utility_benchmark": {
            "num_test_queries": benchmark_results["num_queries"],
            "per_agent_stats": benchmark_results["agent_stats"],
            "comparative_vs_baseline": comparative,
        },
        "skill_quality_llm_assessment": skill_assessments,
    }

    # ------------------------------------------------------------------
    # 8. Print summary & save
    # ------------------------------------------------------------------
    print("\n[8/8] " + "=" * 60)
    print("PHASE 2 BENCHMARK RESULTS")
    print("=" * 60)

    for agent_name in ["SkillAwareAgent", "ReActAgent", "RandomAgent"]:
        stats = benchmark_results["agent_stats"].get(agent_name, {})
        tag = ""
        if agent_name == "SkillAwareAgent":
            tag = "  ← OUR AGENT"
        elif agent_name == "ReActAgent":
            tag = "  ← Baseline 1"
        elif agent_name == "RandomAgent":
            tag = "  ← Baseline 2"
        print(f"\n  {agent_name}{tag}")
        print(f"    Task Success Rate:  {stats.get('task_success_rate', 0):.2%}")
        print(f"    Avg Tool Calls:     {stats.get('avg_tool_calls', 0)}")
        print(f"    Avg Duration (ms):  {stats.get('avg_duration_ms', 0):.0f}")
        print(f"    Error Rate:         {stats.get('error_rate', 0):.2%}")

    print(f"\n  Comparative vs ReActAgent baseline:")
    print(f"    Task Success Rate Δ:  {comparative.get('task_success_rate_improvement', 0):+.2%}")
    print(f"    Latency Reduction:    {comparative.get('latency_reduction', 0):+.1%}")
    print(f"    Tool Call Reduction:  {comparative.get('tool_call_reduction', 0):+.1%}")
    print(f"    Error Rate Reduction: {comparative.get('error_rate_reduction', 0):+.1%}")

    # P2.1 summary — confidence breakdown
    print(f"\n  P2.1 Real Confidence (error-miner on 7,000 traces):")
    print(f"    Mean Confidence:       {mean_real_confidence:.4f}")
    tool_confs = [d["tool_confidence"] for d in per_skill_confidence_detail.values()]
    answer_confs = [d["answer_confidence"] for d in per_skill_confidence_detail.values()]
    if tool_confs:
        print(f"    Mean Tool Confidence:  {sum(tool_confs)/len(tool_confs):.4f}")
        print(f"    Mean Answer Confidence:{sum(answer_confs)/len(answer_confs):.4f}")
    # Show skills with confidence < 1.0 (actual error-prone ones)
    low_conf_skills = [
        (sid, d) for sid, d in per_skill_confidence_detail.items()
        if d["overall_confidence"] < 1.0
    ][:5]
    if low_conf_skills:
        print(f"    Skills with errors (top 5):")
        for sid, d in low_conf_skills:
            print(f"      {sid}: tool_errors={d['tool_errors']}, answer_errors={d['answer_errors']}, "
                  f"overall_confidence={d['overall_confidence']:.4f}")

    if skill_assessments:
        avg_u = sum(a.get("usefulness", 0) for a in skill_assessments) / len(skill_assessments)
        avg_c = sum(a.get("completeness", 0) for a in skill_assessments) / len(skill_assessments)
        avg_g = sum(a.get("generality", 0) for a in skill_assessments) / len(skill_assessments)
        print(f"\n  LLM Skill Quality (top 10 skills):")
        print(f"    Avg Usefulness:   {avg_u:.1f}/10")
        print(f"    Avg Completeness: {avg_c:.1f}/10")
        print(f"    Avg Generality:   {avg_g:.1f}/10")

    print("=" * 60)

    output_path = "../phase2_results.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nFull report saved to {output_path}")

    return report


if __name__ == "__main__":
    main()
