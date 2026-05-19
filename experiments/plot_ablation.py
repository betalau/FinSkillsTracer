"""
Ablation visualization — Phase 1: coverage, uniqueness, consensus.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json


def plot():
    df = pd.read_csv('ablation_results.csv')

    # Parse nested consensus dict if stored as string
    if 'high_consensus_rate' not in df.columns:
        with open('ablation_results.json') as f:
            raw = json.load(f)
        hcr_map = {}
        for r in raw:
            c = r.get('consensus', {})
            if isinstance(c, dict):
                hcr = c.get('high_consensus_rate', 0)
            else:
                hcr = 0
            hcr_map[(r['train_size'], r['method'])] = hcr
        df['high_consensus_rate'] = df.apply(
            lambda row: hcr_map.get((row['train_size'], row['method']), 0), axis=1
        )

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # ---- Coverage (test set) ----
    ax = axes[0]
    sns.lineplot(data=df, x='train_size', y='coverage_test', hue='method',
                 marker='o', ax=ax)
    ax.set_title('Coverage on Test Set')
    ax.set_xlabel('Train Size (Queries)')
    ax.set_ylabel('Coverage')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(title='Method', labels=['A: FSP', 'B: Semantic', 'C: Hierarchical',
                                       'D: Trace2Skill', 'E: SkillClaw'])

    # ---- Uniqueness ----
    ax = axes[1]
    sns.lineplot(data=df, x='train_size', y='uniqueness', hue='method',
                 marker='s', ax=ax)
    ax.set_title('Uniqueness (< 0.8 similarity)')
    ax.set_xlabel('Train Size (Queries)')
    ax.set_ylabel('Uniqueness Score')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(title='Method')

    # ---- High Consensus Rate ----
    ax = axes[2]
    if 'high_consensus_rate' in df.columns:
        sns.lineplot(data=df, x='train_size', y='high_consensus_rate', hue='method',
                     marker='D', ax=ax)
        ax.set_title('High Consensus Rate (>=10/14 models)')
        ax.set_xlabel('Train Size (Queries)')
        ax.set_ylabel('High Consensus Rate')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(title='Method')
    else:
        ax.text(0.5, 0.5, 'Consensus data not available', ha='center', va='center')
        ax.set_title('Consensus')

    fig.suptitle('Ablation Study: Phase 1 Metrics by Mining Method and Data Size',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()

    output_path = 'ablation_coverage.png'
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    plot()
