"""
Generate results table and convergence plots from experiment results.
This script reads existing result files and creates summary tables and plots.
"""

import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_summary_json(filepath: Path) -> dict:
    """Load summary JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def create_results_table(results_dir: Path = Path("results")) -> pd.DataFrame:
    """
    Create extensible results table from summary.json files.
    
    Args:
        results_dir: Root directory containing method result folders
        
    Returns:
        DataFrame with experiment results (one row per method)
    """
    rows = []
    
    # Look for summary.json files in method subdirectories
    for method_dir in results_dir.iterdir():
        if method_dir.is_dir():
            summary_path = method_dir / "summary.json"
            if summary_path.exists():
                summary = load_summary_json(summary_path)
                
                # Extract key metrics and parameters
                row = {
                    'method': summary.get('method', ''),
                    'dataset': summary.get('dataset', ''),
                    'train_split': summary.get('train_split', ''),
                    'test_split': summary.get('test_split', ''),
                    'n_users': summary.get('n_users', 0),
                    'n_items': summary.get('n_items', 0),
                    'latent_dim': summary.get('latent_dim', 0),
                    'learning_rate': summary.get('learning_rate', None),
                    'regularization': summary.get('regularization', None),
                    'n_iterations': summary.get('n_iterations', 0),
                    'random_seed': summary.get('random_seed', 0),
                    'rmse': summary.get('rmse', None),
                    'mae': summary.get('mae', None)
                }
                rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Sort by method name for consistent ordering
    if not df.empty:
        df = df.sort_values('method').reset_index(drop=True)
    
    return df


def plot_convergence_curve(loss_curve_path: Path, output_path: Path, 
                           method_name: str = "MF-SGD", dataset_name: str = "MovieLens 100K"):
    """
    Plot training convergence curve from loss curve CSV.
    
    Args:
        loss_curve_path: Path to loss_curve.csv
        output_path: Path to save the plot
        method_name: Name of the method for title
        dataset_name: Name of the dataset for title
    """
    # Read loss curve
    df_loss = pd.read_csv(loss_curve_path)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(df_loss['iteration'], df_loss['loss'], linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'{method_name} Training Convergence ({dataset_name})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Convergence plot saved to: {output_path}")


def main():
    """Generate results table and plots."""
    results_dir = Path("results")
    
    # Create results table
    print("Creating results table...")
    df_results = create_results_table(results_dir)
    
    if df_results.empty:
        print("Warning: No summary.json files found. Results table will be empty.")
    else:
        print(f"Found {len(df_results)} method(s) in results table.")
    
    # Save results table
    output_table_path = results_dir / "results_table.csv"
    df_results.to_csv(output_table_path, index=False)
    print(f"Results table saved to: {output_table_path}")
    
    # Generate convergence plots for each method
    for method_dir in results_dir.iterdir():
        if method_dir.is_dir():
            loss_curve_path = method_dir / "loss_curve.csv"
            summary_path = method_dir / "summary.json"
            
            if loss_curve_path.exists() and summary_path.exists():
                # Get method name and dataset from summary
                summary = load_summary_json(summary_path)
                method_name = summary.get('method', method_dir.name)
                
                # Use "MovieLens 100K" as dataset name (standard format)
                dataset_name = "MovieLens 100K"
                
                # Create plot
                plot_output_path = method_dir / f"{method_dir.name}_convergence.png"
                plot_convergence_curve(loss_curve_path, plot_output_path, method_name, dataset_name)


if __name__ == "__main__":
    main()

