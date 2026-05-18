import sys
import os
import json
from ingestion.normalization import DataIngestionEngine
from mining.fsp import FSPMiner
from mining.semantic import SemanticMiner
from mining.hierarchical import HierarchicalMiner
from mining.trace2skill import Trace2SkillMiner
from mining.skillclaw import SkillClawEvolver
from evaluation.llm_evaluator import SkillEvaluatorV2

def main():
    # 1. Load data
    ingest = DataIngestionEngine('../data/tool_traces.parquet', '../data/scores.parquet')
    golden_df = ingest.filter_golden_paths()
    normalized = ingest.normalize_traces(golden_df)
    
    # 2. Run all miners
    print("Mining skills with all strategies...")
    skills_a = FSPMiner(normalized).mine(min_support=10)
    skills_b = SemanticMiner(normalized).mine(n_clusters=10)
    skills_c = HierarchicalMiner(normalized).mine()
    skills_d = Trace2SkillMiner(normalized).mine()
    
    # 3. Evolve with SkillClaw
    print("Evolving skills with SkillClaw...")
    all_raw = skills_a + skills_b + skills_c + skills_d
    evolver = SkillClawEvolver(all_raw)
    final_skills = evolver.evolve()
    
    # 4. Save to skills_bank_v2.json
    output_path = "../skills_bank_v2.json"
    with open(output_path, "w") as f:
        json.dump([s.to_dict() for s in final_skills], f, indent=2)
        
    print(f"Successfully generated {len(final_skills)} v2 skills in {output_path}")

if __name__ == "__main__":
    main()
