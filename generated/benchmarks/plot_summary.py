import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Read data from JSON file
json_file = 'benchmark_summary.json'

if not os.path.exists(json_file):
    print(f"Error: {json_file} not found. Please run calc_sum.py first to generate the summary data.")
    exit(1)

with open(json_file, 'r') as f:
    summary_data = json.load(f)

# Extract data from JSON
models = []
total_means = []
total_stds = []
successful_means = []
successful_stds = []

for entry in summary_data:
    models.append(entry['model'])
    total_means.append(entry['total_scores']['mean'])
    total_stds.append(entry['total_scores']['std'])
    successful_means.append(entry['successful_scores']['mean'])
    successful_stds.append(entry['successful_scores']['std'])

# Create the plot
fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(models))
width = 0.35  # Width of bars

# Create bars with error bars
bars1 = ax.bar(x - width/2, total_means, width, yerr=total_stds, 
               label='Total Scores', alpha=0.8, capsize=5, 
               color='#2E86AB', edgecolor='black', linewidth=1.2)

bars2 = ax.bar(x + width/2, successful_means, width, yerr=successful_stds,
               label='Successful Scores', alpha=0.8, capsize=5,
               color='#A23B72', edgecolor='black', linewidth=1.2)

# Customize the plot
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Benchmark Results Comparison', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
ax.set_ylim(0, max(max(total_means) + max(total_stds), max(successful_means) + max(successful_stds)) * 1.15)

# Add value labels on bars
def add_value_labels(bars, means, stds):
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

add_value_labels(bars1, total_means, total_stds)
add_value_labels(bars2, successful_means, successful_stds)

plt.tight_layout()
plt.savefig('benchmark_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plot saved as 'benchmark_comparison.png'")