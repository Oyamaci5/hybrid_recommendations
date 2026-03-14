"""
Compare COA-SGD-MF and HHO-SGD-MF on both MovieLens 100K and 1M datasets.
"""

import sys
from pathlib import Path
import pandas as pd
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import Config
from experiments.run_coa_sgd_mf import run_coa_sgd_mf_experiment
from experiments.run_hho_sgd_mf import run_hho_sgd_mf_experiment
from experiments.run_ml1m_experiments import run_coa_sgd_mf_ml1m, run_hho_sgd_mf_ml1m


def compare_coa_hho_100k(run_coa: bool = True, run_hho: bool = False):
    """
    Run and compare COA and HHO on MovieLens 100K.
    
    Args:
        run_coa: Whether to run COA-SGD-MF (default: True)
        run_hho: Whether to run HHO-SGD-MF (default: False, only runs if results don't exist)
    """
    print("=" * 60)
    print("MOVIELENS 100K: COA vs HHO Comparison")
    print("=" * 60)
    
    config = Config()
    
    # Run COA-SGD-MF if requested
    coa_summary_path = Path("results/coa_sgd_mf/summary.json")
    if run_coa:
        print("\n[1/2] Running COA-SGD-MF...")
        coa_results = run_coa_sgd_mf_experiment(config, verbose=True)
    elif not coa_summary_path.exists():
        print("\n[1/2] Running COA-SGD-MF (results not found)...")
        coa_results = run_coa_sgd_mf_experiment(config, verbose=True)
    else:
        print("\n[1/2] COA-SGD-MF results found, skipping run...")
    
    # Run HHO-SGD-MF only if requested or if results don't exist
    hho_summary_path = Path("results/hho_sgd_mf/summary.json")
    if run_hho:
        print("\n[2/2] Running HHO-SGD-MF...")
        hho_results = run_hho_sgd_mf_experiment(config, verbose=True)
    elif not hho_summary_path.exists():
        print("\n[2/2] HHO-SGD-MF results not found, skipping comparison...")
    else:
        print("\n[2/2] HHO-SGD-MF results found, skipping run...")
    
    # Load summaries for comparison
    if coa_summary_path.exists():
        with open(coa_summary_path, 'r') as f:
            coa_summary = json.load(f)
        
        if hho_summary_path.exists():
            with open(hho_summary_path, 'r') as f:
                hho_summary = json.load(f)
            
            print("\n" + "=" * 60)
            print("COMPARISON: MovieLens 100K")
            print("=" * 60)
            print(f"{'Metric':<20} {'COA-SGD-MF':<15} {'HHO-SGD-MF':<15} {'Difference':<15}")
            print("-" * 60)
            print(f"{'RMSE':<20} {coa_summary['rmse']:<15.6f} {hho_summary['rmse']:<15.6f} "
                  f"{coa_summary['rmse'] - hho_summary['rmse']:<15.6f}")
            print(f"{'MAE':<20} {coa_summary['mae']:<15.6f} {hho_summary['mae']:<15.6f} "
                  f"{coa_summary['mae'] - hho_summary['mae']:<15.6f}")
            print(f"{'Final Loss':<20} {coa_summary['final_loss']:<15.6f} {hho_summary.get('final_loss', 0):<15.6f} "
                  f"{coa_summary['final_loss'] - hho_summary.get('final_loss', 0):<15.6f}")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("COA-SGD-MF Results: MovieLens 100K")
            print("=" * 60)
            print(f"RMSE: {coa_summary['rmse']:.6f}")
            print(f"MAE: {coa_summary['mae']:.6f}")
            print(f"Final Loss: {coa_summary['final_loss']:.6f}")
            print("=" * 60)
            print("Note: HHO-SGD-MF results not found for comparison.")
    else:
        print("\nError: COA-SGD-MF results not found!")


def compare_coa_hho_1m(run_coa: bool = True, run_hho: bool = False):
    """
    Run and compare COA and HHO on MovieLens 1M.
    
    Args:
        run_coa: Whether to run COA-SGD-MF (default: True)
        run_hho: Whether to run HHO-SGD-MF (default: False, only runs if results don't exist)
    """
    print("\n" + "=" * 60)
    print("MOVIELENS 1M: COA vs HHO Comparison")
    print("=" * 60)
    
    config = Config(
        data_dir="data/ml-1m",
        train_split="train.dat",
        test_split="test.dat",
        latent_dim=10,
        learning_rate=0.01,
        regularization=0.01,
        n_iterations=100,
        random_seed=42
    )
    
    # Run COA-SGD-MF if requested
    coa_summary_path = Path("results/movielens-1m/coa_sgd_mf/summary.json")
    if run_coa:
        print("\n[1/2] Running COA-SGD-MF...")
        coa_results = run_coa_sgd_mf_ml1m(config, verbose=True)
    elif not coa_summary_path.exists():
        print("\n[1/2] Running COA-SGD-MF (results not found)...")
        coa_results = run_coa_sgd_mf_ml1m(config, verbose=True)
    else:
        print("\n[1/2] COA-SGD-MF results found, skipping run...")
    
    # Run HHO-SGD-MF only if requested or if results don't exist
    hho_summary_path = Path("results/movielens-1m/hho_sgd_mf/summary.json")
    if run_hho:
        print("\n[2/2] Running HHO-SGD-MF...")
        hho_results = run_hho_sgd_mf_ml1m(config, verbose=True)
    elif not hho_summary_path.exists():
        print("\n[2/2] HHO-SGD-MF results not found, skipping comparison...")
    else:
        print("\n[2/2] HHO-SGD-MF results found, skipping run...")
    
    # Load summaries for comparison
    if coa_summary_path.exists():
        with open(coa_summary_path, 'r') as f:
            coa_summary = json.load(f)
        
        if hho_summary_path.exists():
            with open(hho_summary_path, 'r') as f:
                hho_summary = json.load(f)
            
            print("\n" + "=" * 60)
            print("COMPARISON: MovieLens 1M")
            print("=" * 60)
            print(f"{'Metric':<20} {'COA-SGD-MF':<15} {'HHO-SGD-MF':<15} {'Difference':<15}")
            print("-" * 60)
            print(f"{'RMSE':<20} {coa_summary['rmse']:<15.6f} {hho_summary['rmse']:<15.6f} "
                  f"{coa_summary['rmse'] - hho_summary['rmse']:<15.6f}")
            print(f"{'MAE':<20} {coa_summary['mae']:<15.6f} {hho_summary['mae']:<15.6f} "
                  f"{coa_summary['mae'] - hho_summary['mae']:<15.6f}")
            print(f"{'Final Loss':<20} {coa_summary['final_loss']:<15.6f} {hho_summary.get('final_loss', 0):<15.6f} "
                  f"{coa_summary['final_loss'] - hho_summary.get('final_loss', 0):<15.6f}")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("COA-SGD-MF Results: MovieLens 1M")
            print("=" * 60)
            print(f"RMSE: {coa_summary['rmse']:.6f}")
            print(f"MAE: {coa_summary['mae']:.6f}")
            print(f"Final Loss: {coa_summary['final_loss']:.6f}")
            print("=" * 60)
            print("Note: HHO-SGD-MF results not found for comparison.")
    else:
        print("\nError: COA-SGD-MF results not found!")


def generate_comparison_table():
    """Generate comparison table for both datasets."""
    rows = []
    
    # MovieLens 100K
    for method in ['coa_sgd_mf', 'hho_sgd_mf']:
        summary_path = Path(f"results/{method}/summary.json")
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary = json.load(f)
                rows.append({
                    'Dataset': 'MovieLens 100K',
                    'Method': summary.get('method', ''),
                    'RMSE': summary.get('rmse', None),
                    'MAE': summary.get('mae', None),
                    'Final Loss': summary.get('final_loss', None)
                })
    
    # MovieLens 1M
    for method in ['coa_sgd_mf', 'hho_sgd_mf']:
        summary_path = Path(f"results/movielens-1m/{method}/summary.json")
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary = json.load(f)
                rows.append({
                    'Dataset': 'MovieLens 1M',
                    'Method': summary.get('method', ''),
                    'RMSE': summary.get('rmse', None),
                    'MAE': summary.get('mae', None),
                    'Final Loss': summary.get('final_loss', None)
                })
    
    if rows:
        df = pd.DataFrame(rows)
        table_path = Path("results/coa_hho_comparison.csv")
        df.to_csv(table_path, index=False)
        
        print("\n" + "=" * 60)
        print("COMPARISON TABLE: COA vs HHO")
        print("=" * 60)
        print(df.to_string(index=False))
        print("=" * 60)
        print(f"\nTable saved to: {table_path}")


def main(run_coa: bool = True, run_hho: bool = False):
    """
    Run all comparisons.
    
    Args:
        run_coa: Whether to run COA-SGD-MF (default: True)
        run_hho: Whether to run HHO-SGD-MF (default: False)
    """
    # Compare on MovieLens 100K
    compare_coa_hho_100k(run_coa=run_coa, run_hho=run_hho)
    
    # Compare on MovieLens 1M
    compare_coa_hho_1m(run_coa=run_coa, run_hho=run_hho)
    
    # Generate comparison table
    generate_comparison_table()
    
    print("\n" + "=" * 60)
    print("ALL COMPARISONS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Compare COA and HHO on MovieLens datasets')
    parser.add_argument('--run-coa', action='store_true', default=True,
                       help='Run COA-SGD-MF (default: True)')
    parser.add_argument('--no-run-coa', dest='run_coa', action='store_false',
                       help='Skip running COA-SGD-MF')
    parser.add_argument('--run-hho', action='store_true', default=False,
                       help='Run HHO-SGD-MF (default: False, only runs if results missing)')
    args = parser.parse_args()
    
    main(run_coa=args.run_coa, run_hho=args.run_hho)
