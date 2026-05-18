import pandas as pd
import json
from typing import List, Dict, Any

class DataIngestionEngine:
    def __init__(self, traces_path: str, scores_path: str):
        self.traces_df = pd.read_parquet(traces_path)
        self.scores_df = pd.read_parquet(scores_path)
        
    def filter_golden_paths(self) -> pd.DataFrame:
        """
        Filter successful traces based on scores.
        """
        # Join traces with scores to identify successful ones
        merged_df = pd.merge(self.traces_df, self.scores_df[['index', 'configuration', 'is_correct']], 
                             on=['index', 'configuration'])
        golden_paths = merged_df[merged_df['is_correct'] == True]
        return golden_paths

    def normalize_traces(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Flatten nested JSON tool calls into structured sequence data.
        """
        normalized_data = []
        for _, row in df.iterrows():
            try:
                calls = json.loads(row['tool_calls'])
                formatted_calls = []
                for call in calls:
                    # Basic cleaning and normalization
                    formatted_calls.append({
                        "tool_name": call.get("name"),
                        "input": call.get("input"),
                        "output": call.get("output"),
                        "status": "success" if not call.get("isError", False) else "error"
                    })
                
                normalized_data.append({
                    "trace_id": f"{row['configuration']}_{row['index']}",
                    "original_index": row['index'],
                    "configuration": row['configuration'],
                    "calls": formatted_calls
                })
            except Exception as e:
                print(f"Error processing row {row['index']}: {e}")
                continue
                
        return normalized_data

if __name__ == "__main__":
    engine = DataIngestionEngine("../../data/tool_traces.parquet", "../../data/scores.parquet")
    golden_df = engine.filter_golden_paths()
    normalized = engine.normalize_traces(golden_df)
    print(f"Total golden traces: {len(normalized)}")
    print(json.dumps(normalized[0], indent=2, ensure_ascii=False)[:1000])
