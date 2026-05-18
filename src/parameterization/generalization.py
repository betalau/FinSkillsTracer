from typing import List, Dict, Any, Tuple
import json
import re

class ParameterizationEngine:
    def __init__(self, traces: List[Dict[str, Any]]):
        self.traces = traces

    def extract_variable_mappings(self, skill_sequence: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze how data flows between tools in a sequence.
        """
        # We look for instances of the skill sequence in the traces
        templates = []
        
        for trace in self.traces:
            calls = trace['calls']
            for i in range(len(calls) - len(skill_sequence) + 1):
                window = calls[i:i + len(skill_sequence)]
                window_names = [c['tool_name'] for c in window]
                
                if window_names == skill_sequence:
                    # Found an instance, now analyze data flow
                    template = self._generate_template_from_instance(window)
                    templates.append(template)
        
        # Aggregate templates (simplified: take the most common structure)
        if not templates:
            return []
            
        return templates[0] # Return the first one as a representative for now

    def _generate_template_from_instance(self, window: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        template = []
        # Store outputs to see if they are used as inputs later
        outputs_pool = {}
        
        for idx, call in enumerate(window):
            tool_template = {
                "tool_name": call['tool_name'],
                "input_mapping": {},
                "static_inputs": {}
            }
            
            # Analyze inputs
            inputs = call.get('input', {})
            if isinstance(inputs, dict):
                for k, v in inputs.items():
                    # Check if this value was seen in previous outputs
                    found_mapping = False
                    for prev_idx, prev_output in outputs_pool.items():
                        # Simplified check: is the value contained in the previous output?
                        if str(v) in str(prev_output) and len(str(v)) > 2:
                            tool_template["input_mapping"][k] = {
                                "source_step": prev_idx,
                                "source_key": "auto_detected" # In a real system, we'd find the exact key
                            }
                            found_mapping = True
                            break
                    
                    if not found_mapping:
                        tool_template["static_inputs"][k] = v
            
            template.append(tool_template)
            # Add current output to pool
            outputs_pool[idx] = call.get('output', {})
            
        return template

if __name__ == "__main__":
    # Example usage
    mock_window = [
        {
            "tool_name": "discover_companies",
            "input": {"keywords": ["Apple"]},
            "output": {"company_id": 123}
        },
        {
            "tool_name": "get_fundamentals",
            "input": {"company_id": 123, "period": "FY2023"},
            "output": {"revenue": "383B"}
        }
    ]
    engine = ParameterizationEngine([])
    template = engine._generate_template_from_instance(mock_window)
    print(json.dumps(template, indent=2))
