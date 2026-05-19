import pandas as pd
import json
from typing import List, Dict, Any, Optional


class DataIngestionEngine:
    def __init__(self, traces_path: str, scores_path: str,
                 questions_path: Optional[str] = None):
        self.traces_df = pd.read_parquet(traces_path)
        self.scores_df = pd.read_parquet(scores_path)
        self.questions_df = None
        if questions_path:
            import os
            if os.path.exists(questions_path):
                self.questions_df = pd.read_parquet(questions_path)

    def filter_golden_paths(self) -> pd.DataFrame:
        """Filter successful traces based on scores."""
        merged_df = pd.merge(
            self.traces_df,
            self.scores_df[['index', 'configuration', 'is_correct', 'expected_value']],
            on=['index', 'configuration']
        )

        # Also merge question text if available
        if self.questions_df is not None:
            merged_df = pd.merge(
                merged_df,
                self.questions_df[['index', 'question', 'answer']],
                on='index', how='left'
            )

        golden_paths = merged_df[merged_df['is_correct'] == True]
        return golden_paths

    def load_all_traces(self) -> pd.DataFrame:
        """Load ALL traces (including errors) with scores and questions."""
        merged_df = pd.merge(
            self.traces_df,
            self.scores_df[['index', 'configuration', 'is_correct', 'expected_value']],
            on=['index', 'configuration']
        )

        if self.questions_df is not None:
            merged_df = pd.merge(
                merged_df,
                self.questions_df[['index', 'question', 'answer']],
                on='index', how='left'
            )

        return merged_df

    def normalize_traces(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Flatten nested JSON tool calls into structured sequence data."""
        normalized_data = []
        for _, row in df.iterrows():
            try:
                calls = json.loads(row['tool_calls'])
                formatted_calls = []
                for call in calls:
                    formatted_calls.append({
                        "tool_name": call.get("name"),
                        "input": call.get("input"),
                        "output": call.get("output"),
                        "duration_ms": call.get("duration_ms", 0),
                        "status": "success" if not call.get("is_error", False) else "error"
                    })

                entry = {
                    "trace_id": f"{row['configuration']}_{row['index']}",
                    "original_index": row['index'],
                    "configuration": row['configuration'],
                    "calls": formatted_calls,
                    "expected_value": row.get('expected_value'),
                    "is_correct": row.get('is_correct', False),
                }

                # Optional question text
                if 'question' in row.index and pd.notna(row.get('question')):
                    entry["question"] = row['question']
                if 'answer' in row.index and pd.notna(row.get('answer')):
                    entry["answer"] = row['answer']

                normalized_data.append(entry)
            except Exception as e:
                print(f"Error processing row {row['index']}: {e}")
                continue

        return normalized_data


if __name__ == "__main__":
    engine = DataIngestionEngine(
        "../../data/tool_traces.parquet",
        "../../data/scores.parquet",
        "../../data/questions.parquet"
    )
    golden_df = engine.filter_golden_paths()
    normalized = engine.normalize_traces(golden_df)
    print(f"Total golden traces: {len(normalized)}")
    print(f"Sample trace keys: {list(normalized[0].keys())}")
    print(f"Question: {normalized[0].get('question', 'N/A')[:100]}")
