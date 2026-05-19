"""
Scaling Law visualization — updated for Phase 1 with train/test coverage.
"""
import json
import matplotlib.pyplot as plt
import pandas as pd
import os


def visualize():
    results_path = 'experiment_results.json'
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found. Run run_pipeline.py first.")
        return

    with open(results_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['subset_size'] = df['subset_size'].astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ---- Left: Skills count & Coverage ----
    ax1 = axes[0]
    color1 = 'tab:blue'
    ax1.set_xlabel('Subset Size (Queries)')
    ax1.set_ylabel('Number of Skills', color=color1)
    ax1.bar(df['subset_size'], df['num_skills'], color=color1, alpha=0.6, label='Num Skills')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax1b = ax1.twinx()
    color2 = 'tab:red'
    ax1b.set_ylabel('Coverage (%)', color=color2)
    ax1b.plot(df['subset_size'], [c * 100 for c in df['coverage_train']],
              color='tab:orange', marker='s', linestyle='--', label='Coverage (train)')
    ax1b.plot(df['subset_size'], [c * 100 for c in df['coverage_test']],
              color=color2, marker='o', label='Coverage (test)')
    ax1b.tick_params(axis='y', labelcolor=color2)
    ax1.set_title('Skill Discovery vs. Data Size')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # ---- Right: Uniqueness & Confidence ----
    ax2 = axes[1]
    x = range(len(df))
    width = 0.35
    ax2.bar([i - width/2 for i in x], df['uniqueness'], width,
            color='tab:green', alpha=0.7, label='Uniqueness')
    ax2.bar([i + width/2 for i in x], df['confidence'], width,
            color='tab:purple', alpha=0.7, label='Confidence')
    ax2.set_xlabel('Subset Size (Queries)')
    ax2.set_ylabel('Score')
    ax2.set_title('Skill Quality Metrics')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['subset_size'])
    ax2.legend()
    ax2.set_ylim(0, 1.1)

    fig.suptitle('Scaling Law: Skill Discovery vs. Data Size (Phase 1)', fontsize=14)
    fig.tight_layout()

    output_path = 'scaling_law_v1.png'
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    visualize()
