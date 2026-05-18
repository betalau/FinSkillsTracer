import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot():
    df = pd.read_csv('ablation_results.csv')
    
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df, x='train_size', y='coverage', hue='method', marker='o')
    
    plt.title('Ablation Study: Coverage by Mining Method and Data Size')
    plt.xlabel('Subset Size (Queries)')
    plt.ylabel('Coverage')
    plt.grid(True)
    plt.legend(title='Method', labels=['A: FSP', 'B: Semantic', 'C: Hierarchical', 'D: Trace2Skill', 'E: SkillClaw'])
    
    output_path = 'ablation_coverage.png'
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot()
