# FinSkillsTracer: Trace-to-Skills Mining Platform

Automated pipeline for mining reusable Agent Skills from financial retrieval traces (FinRetrieval dataset).

## Features
- **Module 1: Data Ingestion**: Filters golden paths and normalizes parquet traces.
- **Module 2: Multi-Strategy Mining**: Implements Frequent Sequence Mining, Semantic Clustering, and Hierarchical Mining.
- **Module 3: Parameterization**: Automatically detects data flow between tools to create generalized skill templates.
- **Module 4: Evaluation Engine**: Evaluates skills based on Support, Confidence, Coverage, and Utility.

## Project Structure
- `src/`: Core engine implementation.
- `experiments/`: Scripts for running scaling law and comparison experiments.
- `data/`: FinRetrieval dataset files (not tracked in git).
- `skills_bank.json`: The final output of the mining process.

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run the mining pipeline: `cd src && python main.py`
3. View experiment results: `cd experiments && python run_pipeline.py`