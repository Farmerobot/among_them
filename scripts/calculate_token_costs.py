#!/usr/bin/env python3
import os
import csv
import sys
from collections import defaultdict
from typing import Dict, List, Any

# Add the project root to the Python path to import the TOKEN_COSTS
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.among_them.game.consts import TOKEN_COSTS

# Project paths
csv_path = os.path.join(project_root, "data", "voting_token_estimates2.csv")

def load_csv_data(file_path: str) -> List[Dict[str, Any]]:
    """Load token estimates from the CSV file."""
    data = []
    try:
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Convert numeric fields to appropriate types
                row['input_tokens'] = int(row['input_tokens'])
                row['estimated_output_tokens'] = int(row['estimated_output_tokens'])
                row['total_tokens'] = int(row['total_tokens'])
                data.append(row)
        return data
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        return []

def calculate_costs(data: List[Dict[str, Any]], token_costs: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
    """Calculate costs based on token counts and TOKEN_COSTS."""
    model_costs = defaultdict(lambda: {
        'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0,
        'input_cost': 0.0, 'output_cost': 0.0, 'total_cost': 0.0,
        'samples': 0
    })
    
    missing_models = set()
    
    for entry in data:
        model = entry['model']
        if model not in token_costs:
            missing_models.add(model)
            continue
            
        cost_info = token_costs[model]
        
        # Add token counts
        model_costs[model]['input_tokens'] += entry['input_tokens']
        model_costs[model]['output_tokens'] += entry['estimated_output_tokens']
        model_costs[model]['total_tokens'] += entry['total_tokens']
        model_costs[model]['samples'] += 1
        
        # Calculate costs in dollars per million tokens (divide by 1,000,000)
        input_cost = entry['input_tokens'] * cost_info['input_tokens'] / 1_000_000
        output_cost = entry['estimated_output_tokens'] * cost_info['output_tokens'] / 1_000_000
        
        # Add costs
        model_costs[model]['input_cost'] += input_cost
        model_costs[model]['output_cost'] += output_cost
        model_costs[model]['total_cost'] += (input_cost + output_cost)
    
    if missing_models:
        print(f"Warning: The following models were not found in TOKEN_COSTS: {', '.join(missing_models)}")
    
    return model_costs

def main():
    print(f"Loading token estimates from {csv_path}")
    data = load_csv_data(csv_path)
    
    if not data:
        print("No data found in CSV file. Exiting.")
        return
    
    print(f"Calculating costs for {len(data)} records using TOKEN_COSTS")
    model_costs = calculate_costs(data, TOKEN_COSTS)
    
    # Generate a detailed report
    print("\nToken Usage and Cost Report:")
    print("-" * 120)
    print(f"{'Model':<35} {'Total Tokens':<12} {'Avg Tokens':<12} {'Total Cost':<12} {'Avg Cost':<12} {'Cost per 1K':<12} {'Samples':<8}")
    print("-" * 120)
    
    for model, stats in sorted(model_costs.items()):
        if stats['samples'] > 0:
            avg_tokens = stats['total_tokens'] / stats['samples']
            avg_cost = stats['total_cost'] / stats['samples']
            cost_per_1k = stats['total_cost'] / stats['total_tokens'] * 1000 if stats['total_tokens'] > 0 else 0
            
            print(f"{model:<35} {stats['total_tokens']:<12} {avg_tokens:<12.1f} "
                  f"${stats['total_cost']:<11.6f} ${avg_cost:<11.6f} ${cost_per_1k:<11.6f} {stats['samples']:<8}")
    
    # Calculate totals across all models
    total_tokens = sum(stats['total_tokens'] for stats in model_costs.values())
    total_cost = sum(stats['total_cost'] for stats in model_costs.values())
    total_samples = sum(stats['samples'] for stats in model_costs.values())
    avg_cost_per_sample = total_cost / total_samples if total_samples > 0 else 0
    
    print("-" * 120)
    print(f"{'TOTAL':<35} {total_tokens:<12} {'-':<12} ${total_cost:<11.6f} ${avg_cost_per_sample:<11.6f} {'-':<12} {total_samples:<8}")
    
    # Save to CSV
    output_path = os.path.join(project_root, "data", "token_cost_analysis.csv")
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['model', 'total_tokens', 'avg_tokens', 'total_cost', 'avg_cost', 'cost_per_1k', 'samples']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for model, stats in sorted(model_costs.items()):
            if stats['samples'] > 0:
                avg_tokens = stats['total_tokens'] / stats['samples']
                avg_cost = stats['total_cost'] / stats['samples']
                cost_per_1k = stats['total_cost'] / stats['total_tokens'] * 1000 if stats['total_tokens'] > 0 else 0
                
                writer.writerow({
                    'model': model,
                    'total_tokens': stats['total_tokens'],
                    'avg_tokens': f"{avg_tokens:.1f}",
                    'total_cost': f"${stats['total_cost']:.6f}",
                    'avg_cost': f"${avg_cost:.6f}",
                    'cost_per_1k': f"${cost_per_1k:.6f}",
                    'samples': stats['samples']
                })
        
    print(f"\nDetailed analysis saved to {output_path}")
    print("\nDone!")

if __name__ == "__main__":
    main()
