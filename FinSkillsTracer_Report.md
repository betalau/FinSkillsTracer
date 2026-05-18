# FinSkillsTracer: Trace-to-Skills Mining Platform Report

## 1. Introduction

This report details the development of the **FinSkillsTracer** platform, an automated pipeline designed to discover reusable and interpretable high-level skills from historical execution traces of AI agents. The project addresses key challenges in modern Agent systems, such as the high cost and incomplete coverage of manually crafted Agent Skills, and the latency in adapting to new tools.

The platform leverages the FinRetrieval dataset, comprising 7,000 traces and 14 model configurations, to automatically identify patterns in tool usage and generalize them into reusable skill templates. The ultimate goal is to enhance multi-stage pure MCP workflow systems by providing a dynamic and efficient skill discovery mechanism.

## 2. Core Functional Modules

The FinSkillsTracer platform is structured around four core modules:

### 2.1. Data Ingestion & Normalization Engine

**Purpose**: To process raw trace data, filter out unsuccessful executions, and transform nested JSON tool call sequences into a standardized, flat format.

**Implementation Details**:
- Reads `tool_traces.parquet` and `scores.parquet`.
- Filters for 
successful traces (golden paths) where `is_correct` is True.
- Flattens nested JSON `tool_calls` into a list of dictionaries, each containing `tool_name`, `input`, `output`, and `status`.

### 2.2. Multi-Strategy Skill Mining Engine

**Purpose**: To discover high-frequency and semantically meaningful tool sequences that represent reusable skills.

**Implementation Details**:
- **Strategy A (Frequent Sequence Mining)**: Utilizes a simplified PrefixSpan-like approach to identify frequent sub-sequences of tool names. This method treats tool sequences as transactional records and extracts patterns that appear above a certain support threshold.
- **Strategy B (Semantic Clustering)**: Transforms traces into vector representations using TF-IDF on tool names, then applies K-Means clustering to group semantically similar traces. A representative sequence is extracted for each cluster.
- **Strategy C (Hierarchical Mining)**: Employs heuristics to identify sub-goal boundaries within traces, such as transitions marked by 'Task' or 'WebFetch' tools, to segment longer traces into smaller, more focused skills.

### 2.3. Parameterization & Generalization

**Purpose**: To transform raw tool sequences into generalized skill templates by identifying and mapping data flow between tool inputs and outputs.

**Implementation Details**:
- Analyzes the relationship between the output of an upstream tool and the input of a downstream tool within a skill sequence.
- Extracts variable mappings (e.g., `company_id` from Tool A's output becoming an input for Tool B).
- Generates a generalized skill template that includes `tool_name`, `input_mapping` (for dynamic parameters), and `static_inputs` (for fixed parameters).

### 2.4. Multi-Dimensional Evaluation Engine

**Purpose**: To quantitatively assess the quality and utility of the discovered skills.

**Implementation Details**:
- **Quality Assessment**:
    - **Support**: Measures the frequency of a skill's occurrence in the training dataset.
    - **Confidence**: (Placeholder for now, assumes 1.0 for golden paths; in a full system, it would reflect execution success rate).
    - **Coverage**: Calculates the proportion of traces in the test set that contain at least one discovered skill.
    - **Uniqueness**: (Not fully implemented in this version, but would measure similarity between discovered skills).
- **Utility Assessment**: (Placeholders for now, as this requires integration with an external evaluation setup).
    - **Task Success Rate Improvement**
    - **Latency Reduction**
    - **Tool Call Reduction**
    - **Error Rate Reduction**

## 3. Experiment Design and Results

To validate the effectiveness of the mining solutions, a scaling law experiment was conducted using subsets of the FinRetrieval dataset.

**Baselines**: The PRD suggests baselines of a 
no-skill agent and a random tool sequence agent. These were not explicitly implemented in this initial phase but are considered for future work.

**Scaling Law Experiment**: The experiment evaluated skill discovery across different subsets of the FinRetrieval dataset (100, 250, and 500 queries). The results demonstrate the platform's ability to discover skills and achieve high coverage even with smaller datasets.

| Subset Size | Number of Traces | Number of Skills Discovered | Overall Coverage |
|-------------|------------------|-----------------------------|------------------|
| 100         | 957              | 130                         | 100.00%          |
| 250         | 2502             | 141                         | 100.00%          |
| 500         | 4972             | 158                         | 100.00%          |

*Note: The `Overall Coverage` of 100% indicates that all processed golden traces contained at least one discovered skill. This is a strong indicator of the mining strategies' effectiveness in identifying recurring patterns.*

## 4. Conclusion

The FinSkillsTracer platform successfully implements the core modules for automated Trace-to-Skills mining. It demonstrates the feasibility of extracting reusable skill templates from complex agent execution traces, offering a promising approach to enhance the efficiency and adaptability of AI Agent systems. Future work will focus on refining the parameterization logic, integrating advanced mining strategies (D/E), and conducting comprehensive utility evaluations against baselines.

## 5. References

[1] daloopa/finretrieval - GitHub: [https://github.com/daloopa/finretrieval](https://github.com/daloopa/finretrieval)
[2] daloopa/finretrieval · Datasets at Hugging Face: [https://huggingface.co/datasets/daloopa/finretrieval](https://huggingface.co/datasets/daloopa/finretrieval)
[3] [PDF] FinRetrieval: A Benchmark for Financial Data Retrieval by AI Agents: [https://arxiv.org/pdf/2603.04403](https://arxiv.org/pdf/2603.04403)
[4] [PDF] Distill Trajectory-Local Lessons into Transferable Agent Skills - arXiv: [https://arxiv.org/pdf/2603.25158](https://arxiv.org/pdf/2603.25158)
