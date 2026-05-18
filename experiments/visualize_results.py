import json
import matplotlib.pyplot as plt
import pandas as pd
import os

def visualize():
    results_path = 'experiment_results.json'
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return

    with open(results_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['subset_size'] = df['subset_size'].astype(str)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot number of skills
    color = 'tab:blue'
    ax1.set_xlabel('Subset Size (Queries)')
    ax1.set_ylabel('Number of Skills', color=color)
    ax1.bar(df['subset_size'], df['num_skills'], color=color, alpha=0.6, label='Num Skills')
    ax1.tick_params(axis='y', labelcolor=color)

    # Plot coverage
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Coverage (%)', color=color)
    ax2.plot(df['subset_size'], df['coverage'] * 100, color=color, marker='o', label='Coverage')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Scaling Law: Skill Discovery vs. Data Size')
    fig.tight_layout()
    
    output_path = 'scaling_law_v1.png'
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    visualize()
