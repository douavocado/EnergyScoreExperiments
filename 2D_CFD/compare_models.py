import json
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(results_path: str) -> dict:
    with open(results_path, 'r') as f:
        return json.load(f)


def create_comparison_table(results_dict: dict) -> pd.DataFrame:
    data = []
    
    for model_name, results in results_dict.items():
        metrics = results['metrics']
        row = {'Model': model_name}
        
        for metric, value in metrics.items():
            if not metric.endswith('_std'):
                row[metric] = value
                std_key = f'{metric}_std'
                if std_key in metrics:
                    row[f'{metric}_std'] = metrics[std_key]
        
        data.append(row)
    
    df = pd.DataFrame(data)
    df = df.set_index('Model')
    
    metric_names = ['mae', 'mse', 'variogram_score', 'patched_energy_score', 'pairwise_energy_score']
    ordered_cols = []
    for metric in metric_names:
        if metric in df.columns:
            ordered_cols.append(metric)
        if f'{metric}_std' in df.columns:
            ordered_cols.append(f'{metric}_std')
    
    return df[ordered_cols]


def plot_comparison(df: pd.DataFrame, save_path: str = None):
    metrics = [col for col in df.columns if not col.endswith('_std')]
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        values = df[metric]
        std_col = f'{metric}_std'
        
        bars = ax.bar(range(len(df)), values)
        
        if std_col in df.columns:
            yerr = df[std_col]
            ax.errorbar(range(len(df)), values, yerr=yerr, 
                       fmt='none', color='black', capsize=5)
        
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df.index, rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title())
        
        colors = plt.cm.RdYlGn_r(values / values.max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def calculate_improvement(df: pd.DataFrame, baseline_name: str = 'Persistence') -> pd.DataFrame:
    if baseline_name not in df.index:
        print(f"Warning: Baseline '{baseline_name}' not found in results")
        return None
    
    baseline_values = df.loc[baseline_name]
    improvements = pd.DataFrame(index=df.index, columns=[col for col in df.columns if not col.endswith('_std')])
    
    for model in df.index:
        if model == baseline_name:
            continue
        
        for metric in improvements.columns:
            model_value = df.loc[model, metric]
            baseline_value = baseline_values[metric]
            
            improvement = ((baseline_value - model_value) / baseline_value) * 100
            improvements.loc[model, metric] = improvement
    
    improvements = improvements.drop(baseline_name)
    
    return improvements


def main():
    parser = argparse.ArgumentParser(description="Compare model evaluation results")
    parser.add_argument('--results', nargs='+', required=True,
                       help='Paths to evaluation result JSON files')
    parser.add_argument('--names', nargs='+', 
                       help='Names for each model (default: use filenames)')
    parser.add_argument('--save_table', type=str, default=None,
                       help='Path to save comparison table (CSV format)')
    parser.add_argument('--save_plot', type=str, default=None,
                       help='Path to save comparison plot')
    parser.add_argument('--baseline', type=str, default='Persistence',
                       help='Name of baseline model for improvement calculation')
    
    args = parser.parse_args()
    
    results_dict = {}
    for i, results_path in enumerate(args.results):
        if args.names and i < len(args.names):
            name = args.names[i]
        else:
            name = Path(results_path).stem
            results = load_results(results_path)
            if 'model_type' in results:
                name = results['model_type']
            elif 'evaluation_config' in results and 'model_dir' in results['evaluation_config']:
                name = Path(results['evaluation_config']['model_dir']).name
        
        results_dict[name] = load_results(results_path)
    
    df = create_comparison_table(results_dict)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print("\nMetrics (lower is better for all):")
    print(df.to_string(float_format='%.6f'))
    
    if args.baseline in df.index:
        improvements = calculate_improvement(df, args.baseline)
        if improvements is not None:
            print(f"\n\nPercentage Improvement over {args.baseline} (positive = better):")
            print(improvements.to_string(float_format='%.2f%%'))
    
    if args.save_table:
        df.to_csv(args.save_table)
        print(f"\nComparison table saved to {args.save_table}")
    
    if args.save_plot:
        plot_comparison(df, args.save_plot)
        print(f"Comparison plot saved to {args.save_plot}")


if __name__ == "__main__":
    main()
