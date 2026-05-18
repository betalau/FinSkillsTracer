import sys
import os
import json
from ingestion.normalization import DataIngestionEngine
from mining.mining_engine import SkillMiningEngine
from parameterization.generalization import ParameterizationEngine
from evaluation.evaluator import SkillEvaluator

def main():
    # 1. Load data
    ingest = DataIngestionEngine('../data/tool_traces.parquet', '../data/scores.parquet')
    golden_df = ingest.filter_golden_paths()
    normalized = ingest.normalize_traces(golden_df)
    
    # 2. Mine skills
    miner = SkillMiningEngine(normalized)
    freq_skills = miner.mine_frequent_sequences(min_support=100)
    
    # 3. Parameterize top skills
    param_engine = ParameterizationEngine(normalized)
    final_skills = []
    for skill in freq_skills[:20]: # Top 20 skills
        template = param_engine.extract_variable_mappings(skill['sequence'])
        skill_entry = {
            "skill_id": f"skill_{len(final_skills)}",
            "sequence": skill['sequence'],
            "support": skill['support'],
            "strategy": skill['strategy'],
            "template": template
        }
        final_skills.append(skill_entry)
        
    # 4. Save to skills_bank.json
    output_path = "../skills_bank.json"
    with open(output_path, "w") as f:
        json.dump(final_skills, f, indent=2)
        
    print(f"Successfully generated {len(final_skills)} skills in {output_path}")

if __name__ == "__main__":
    main()
