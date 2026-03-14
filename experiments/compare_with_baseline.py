"""
Compare COA-SGD-MF and HHO-SGD-MF with MF-SGD baseline on both datasets.
"""

import sys
from pathlib import Path
import pandas as pd
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_summary(filepath: Path) -> dict:
    """Load summary JSON file."""
    if filepath.exists():
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def compare_with_baseline_100k():
    """Compare methods with baseline on MovieLens 100K."""
    print("=" * 80)
    print("COMPARISON WITH BASELINE: MovieLens 100K")
    print("=" * 80)
    
    # Load summaries
    baseline_path = Path("results/mf_sgd/summary.json")
    coa_path = Path("results/coa_sgd_mf/summary.json")
    hho_path = Path("results/hho_sgd_mf/summary.json")
    
    baseline = load_summary(baseline_path)
    coa = load_summary(coa_path)
    hho = load_summary(hho_path)
    
    if not baseline:
        print("Error: Baseline (MF-SGD) results not found!")
        return
    
    print(f"\n{'Metric':<20} {'MF-SGD (Baseline)':<20} {'COA-SGD-MF':<20} {'HHO-SGD-MF':<20} {'COA vs Baseline':<20} {'HHO vs Baseline':<20}")
    print("-" * 120)
    
    # RMSE
    baseline_rmse = baseline.get('rmse', 0)
    coa_rmse = coa.get('rmse', 0) if coa else None
    hho_rmse = hho.get('rmse', 0) if hho else None
    
    coa_rmse_diff = (coa_rmse - baseline_rmse) if coa_rmse else None
    hho_rmse_diff = (hho_rmse - baseline_rmse) if hho_rmse else None
    
    coa_rmse_pct = ((coa_rmse_diff / baseline_rmse) * 100) if coa_rmse_diff is not None else None
    hho_rmse_pct = ((hho_rmse_diff / baseline_rmse) * 100) if hho_rmse_diff is not None else None
    
    coa_rmse_str = f"{coa_rmse:<20.6f}" if coa_rmse is not None else "N/A".ljust(20)
    hho_rmse_str = f"{hho_rmse:<20.6f}" if hho_rmse is not None else "N/A".ljust(20)
    coa_rmse_diff_str = f"{coa_rmse_diff:+.6f} ({coa_rmse_pct:+.2f}%)" if coa_rmse_diff is not None else "N/A".ljust(20)
    hho_rmse_diff_str = f"{hho_rmse_diff:+.6f} ({hho_rmse_pct:+.2f}%)" if hho_rmse_diff is not None else "N/A".ljust(20)
    
    print(f"{'RMSE':<20} {baseline_rmse:<20.6f} {coa_rmse_str} {hho_rmse_str} {coa_rmse_diff_str} {hho_rmse_diff_str}")
    
    # MAE
    baseline_mae = baseline.get('mae', 0)
    coa_mae = coa.get('mae', 0) if coa else None
    hho_mae = hho.get('mae', 0) if hho else None
    
    coa_mae_diff = (coa_mae - baseline_mae) if coa_mae else None
    hho_mae_diff = (hho_mae - baseline_mae) if hho_mae else None
    
    coa_mae_pct = ((coa_mae_diff / baseline_mae) * 100) if coa_mae_diff is not None else None
    hho_mae_pct = ((hho_mae_diff / baseline_mae) * 100) if hho_mae_diff is not None else None
    
    coa_mae_str = f"{coa_mae:<20.6f}" if coa_mae is not None else "N/A".ljust(20)
    hho_mae_str = f"{hho_mae:<20.6f}" if hho_mae is not None else "N/A".ljust(20)
    coa_mae_diff_str = f"{coa_mae_diff:+.6f} ({coa_mae_pct:+.2f}%)" if coa_mae_diff is not None else "N/A".ljust(20)
    hho_mae_diff_str = f"{hho_mae_diff:+.6f} ({hho_mae_pct:+.2f}%)" if hho_mae_diff is not None else "N/A".ljust(20)
    
    print(f"{'MAE':<20} {baseline_mae:<20.6f} {coa_mae_str} {hho_mae_str} {coa_mae_diff_str} {hho_mae_diff_str}")
    
    # Final Loss
    baseline_loss = baseline.get('final_loss', baseline.get('loss', 0))
    coa_loss = coa.get('final_loss', 0) if coa else None
    hho_loss = hho.get('final_loss', 0) if hho else None
    
    if baseline_loss and (coa_loss or hho_loss):
        coa_loss_diff = (coa_loss - baseline_loss) if coa_loss else None
        hho_loss_diff = (hho_loss - baseline_loss) if hho_loss else None
        
        coa_loss_pct = ((coa_loss_diff / baseline_loss) * 100) if coa_loss_diff is not None else None
        hho_loss_pct = ((hho_loss_diff / baseline_loss) * 100) if hho_loss_diff is not None else None
        
        coa_loss_str = f"{coa_loss:<20.6f}" if coa_loss is not None else "N/A".ljust(20)
        hho_loss_str = f"{hho_loss:<20.6f}" if hho_loss is not None else "N/A".ljust(20)
        coa_loss_diff_str = f"{coa_loss_diff:+.6f} ({coa_loss_pct:+.2f}%)" if coa_loss_diff is not None else "N/A".ljust(20)
        hho_loss_diff_str = f"{hho_loss_diff:+.6f} ({hho_loss_pct:+.2f}%)" if hho_loss_diff is not None else "N/A".ljust(20)
        
        print(f"{'Final Loss':<20} {baseline_loss:<20.6f} {coa_loss_str} {hho_loss_str} {coa_loss_diff_str} {hho_loss_diff_str}")
    
    print("=" * 120)
    
    # Summary
    print("\nSUMMARY:")
    print("-" * 80)
    if coa:
        if coa_rmse < baseline_rmse:
            print(f"[+] COA-SGD-MF improves RMSE by {abs(coa_rmse_pct):.2f}% vs baseline")
        else:
            print(f"[-] COA-SGD-MF increases RMSE by {abs(coa_rmse_pct):.2f}% vs baseline")
        
        if coa_mae < baseline_mae:
            print(f"[+] COA-SGD-MF improves MAE by {abs(coa_mae_pct):.2f}% vs baseline")
        else:
            print(f"[-] COA-SGD-MF increases MAE by {abs(coa_mae_pct):.2f}% vs baseline")
    
    if hho:
        if hho_rmse < baseline_rmse:
            print(f"[+] HHO-SGD-MF improves RMSE by {abs(hho_rmse_pct):.2f}% vs baseline")
        else:
            print(f"[-] HHO-SGD-MF increases RMSE by {abs(hho_rmse_pct):.2f}% vs baseline")
        
        if hho_mae < baseline_mae:
            print(f"[+] HHO-SGD-MF improves MAE by {abs(hho_mae_pct):.2f}% vs baseline")
        else:
            print(f"[-] HHO-SGD-MF increases MAE by {abs(hho_mae_pct):.2f}% vs baseline")
    
    print("=" * 80)


def compare_with_baseline_1m():
    """Compare methods with baseline on MovieLens 1M."""
    print("\n" + "=" * 80)
    print("COMPARISON WITH BASELINE: MovieLens 1M")
    print("=" * 80)
    
    # Load summaries
    baseline_path = Path("results/movielens-1m/mf_sgd/summary.json")
    coa_path = Path("results/movielens-1m/coa_sgd_mf/summary.json")
    hho_path = Path("results/movielens-1m/hho_sgd_mf/summary.json")
    
    baseline = load_summary(baseline_path)
    coa = load_summary(coa_path)
    hho = load_summary(hho_path)
    
    if not baseline:
        print("Error: Baseline (MF-SGD) results not found!")
        return
    
    print(f"\n{'Metric':<20} {'MF-SGD (Baseline)':<20} {'COA-SGD-MF':<20} {'HHO-SGD-MF':<20} {'COA vs Baseline':<20} {'HHO vs Baseline':<20}")
    print("-" * 120)
    
    # RMSE
    baseline_rmse = baseline.get('rmse', 0)
    coa_rmse = coa.get('rmse', 0) if coa else None
    hho_rmse = hho.get('rmse', 0) if hho else None
    
    coa_rmse_diff = (coa_rmse - baseline_rmse) if coa_rmse else None
    hho_rmse_diff = (hho_rmse - baseline_rmse) if hho_rmse else None
    
    coa_rmse_pct = ((coa_rmse_diff / baseline_rmse) * 100) if coa_rmse_diff is not None else None
    hho_rmse_pct = ((hho_rmse_diff / baseline_rmse) * 100) if hho_rmse_diff is not None else None
    
    coa_rmse_str = f"{coa_rmse:<20.6f}" if coa_rmse is not None else "N/A".ljust(20)
    hho_rmse_str = f"{hho_rmse:<20.6f}" if hho_rmse is not None else "N/A".ljust(20)
    coa_rmse_diff_str = f"{coa_rmse_diff:+.6f} ({coa_rmse_pct:+.2f}%)" if coa_rmse_diff is not None else "N/A".ljust(20)
    hho_rmse_diff_str = f"{hho_rmse_diff:+.6f} ({hho_rmse_pct:+.2f}%)" if hho_rmse_diff is not None else "N/A".ljust(20)
    
    print(f"{'RMSE':<20} {baseline_rmse:<20.6f} {coa_rmse_str} {hho_rmse_str} {coa_rmse_diff_str} {hho_rmse_diff_str}")
    
    # MAE
    baseline_mae = baseline.get('mae', 0)
    coa_mae = coa.get('mae', 0) if coa else None
    hho_mae = hho.get('mae', 0) if hho else None
    
    coa_mae_diff = (coa_mae - baseline_mae) if coa_mae else None
    hho_mae_diff = (hho_mae - baseline_mae) if hho_mae else None
    
    coa_mae_pct = ((coa_mae_diff / baseline_mae) * 100) if coa_mae_diff is not None else None
    hho_mae_pct = ((hho_mae_diff / baseline_mae) * 100) if hho_mae_diff is not None else None
    
    coa_mae_str = f"{coa_mae:<20.6f}" if coa_mae is not None else "N/A".ljust(20)
    hho_mae_str = f"{hho_mae:<20.6f}" if hho_mae is not None else "N/A".ljust(20)
    coa_mae_diff_str = f"{coa_mae_diff:+.6f} ({coa_mae_pct:+.2f}%)" if coa_mae_diff is not None else "N/A".ljust(20)
    hho_mae_diff_str = f"{hho_mae_diff:+.6f} ({hho_mae_pct:+.2f}%)" if hho_mae_diff is not None else "N/A".ljust(20)
    
    print(f"{'MAE':<20} {baseline_mae:<20.6f} {coa_mae_str} {hho_mae_str} {coa_mae_diff_str} {hho_mae_diff_str}")
    
    # Final Loss
    baseline_loss = baseline.get('final_loss', baseline.get('loss', 0))
    coa_loss = coa.get('final_loss', 0) if coa else None
    hho_loss = hho.get('final_loss', 0) if hho else None
    
    if baseline_loss and (coa_loss or hho_loss):
        coa_loss_diff = (coa_loss - baseline_loss) if coa_loss else None
        hho_loss_diff = (hho_loss - baseline_loss) if hho_loss else None
        
        coa_loss_pct = ((coa_loss_diff / baseline_loss) * 100) if coa_loss_diff is not None else None
        hho_loss_pct = ((hho_loss_diff / baseline_loss) * 100) if hho_loss_diff is not None else None
        
        coa_loss_str = f"{coa_loss:<20.6f}" if coa_loss is not None else "N/A".ljust(20)
        hho_loss_str = f"{hho_loss:<20.6f}" if hho_loss is not None else "N/A".ljust(20)
        coa_loss_diff_str = f"{coa_loss_diff:+.6f} ({coa_loss_pct:+.2f}%)" if coa_loss_diff is not None else "N/A".ljust(20)
        hho_loss_diff_str = f"{hho_loss_diff:+.6f} ({hho_loss_pct:+.2f}%)" if hho_loss_diff is not None else "N/A".ljust(20)
        
        print(f"{'Final Loss':<20} {baseline_loss:<20.6f} {coa_loss_str} {hho_loss_str} {coa_loss_diff_str} {hho_loss_diff_str}")
    
    print("=" * 120)
    
    # Summary
    print("\nSUMMARY:")
    print("-" * 80)
    if coa:
        if coa_rmse < baseline_rmse:
            print(f"[+] COA-SGD-MF improves RMSE by {abs(coa_rmse_pct):.2f}% vs baseline")
        else:
            print(f"[-] COA-SGD-MF increases RMSE by {abs(coa_rmse_pct):.2f}% vs baseline")
        
        if coa_mae < baseline_mae:
            print(f"[+] COA-SGD-MF improves MAE by {abs(coa_mae_pct):.2f}% vs baseline")
        else:
            print(f"[-] COA-SGD-MF increases MAE by {abs(coa_mae_pct):.2f}% vs baseline")
    
    if hho:
        if hho_rmse < baseline_rmse:
            print(f"[+] HHO-SGD-MF improves RMSE by {abs(hho_rmse_pct):.2f}% vs baseline")
        else:
            print(f"[-] HHO-SGD-MF increases RMSE by {abs(hho_rmse_pct):.2f}% vs baseline")
        
        if hho_mae < baseline_mae:
            print(f"[+] HHO-SGD-MF improves MAE by {abs(hho_mae_pct):.2f}% vs baseline")
        else:
            print(f"[-] HHO-SGD-MF increases MAE by {abs(hho_mae_pct):.2f}% vs baseline")
    
    print("=" * 80)


def generate_baseline_comparison_table():
    """Generate comparison table with baseline."""
    rows = []
    
    # MovieLens 100K
    baseline_100k = load_summary(Path("results/mf_sgd/summary.json"))
    coa_100k = load_summary(Path("results/coa_sgd_mf/summary.json"))
    hho_100k = load_summary(Path("results/hho_sgd_mf/summary.json"))
    
    if baseline_100k:
        rows.append({
            'Dataset': 'MovieLens 100K',
            'Method': 'MF-SGD (Baseline)',
            'RMSE': baseline_100k.get('rmse', None),
            'MAE': baseline_100k.get('mae', None),
            'Final Loss': baseline_100k.get('final_loss', None)
        })
    
    if coa_100k:
        rows.append({
            'Dataset': 'MovieLens 100K',
            'Method': 'COA-SGD-MF',
            'RMSE': coa_100k.get('rmse', None),
            'MAE': coa_100k.get('mae', None),
            'Final Loss': coa_100k.get('final_loss', None)
        })
    
    if hho_100k:
        rows.append({
            'Dataset': 'MovieLens 100K',
            'Method': 'HHO-SGD-MF',
            'RMSE': hho_100k.get('rmse', None),
            'MAE': hho_100k.get('mae', None),
            'Final Loss': hho_100k.get('final_loss', None)
        })
    
    # MovieLens 1M
    baseline_1m = load_summary(Path("results/movielens-1m/mf_sgd/summary.json"))
    coa_1m = load_summary(Path("results/movielens-1m/coa_sgd_mf/summary.json"))
    hho_1m = load_summary(Path("results/movielens-1m/hho_sgd_mf/summary.json"))
    
    if baseline_1m:
        rows.append({
            'Dataset': 'MovieLens 1M',
            'Method': 'MF-SGD (Baseline)',
            'RMSE': baseline_1m.get('rmse', None),
            'MAE': baseline_1m.get('mae', None),
            'Final Loss': baseline_1m.get('final_loss', None)
        })
    
    if coa_1m:
        rows.append({
            'Dataset': 'MovieLens 1M',
            'Method': 'COA-SGD-MF',
            'RMSE': coa_1m.get('rmse', None),
            'MAE': coa_1m.get('mae', None),
            'Final Loss': coa_1m.get('final_loss', None)
        })
    
    if hho_1m:
        rows.append({
            'Dataset': 'MovieLens 1M',
            'Method': 'HHO-SGD-MF',
            'RMSE': hho_1m.get('rmse', None),
            'MAE': hho_1m.get('mae', None),
            'Final Loss': hho_1m.get('final_loss', None)
        })
    
    if rows:
        df = pd.DataFrame(rows)
        table_path = Path("results/baseline_comparison.csv")
        df.to_csv(table_path, index=False)
        
        print("\n" + "=" * 80)
        print("COMPARISON TABLE: All Methods vs Baseline")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)
        print(f"\nTable saved to: {table_path}")


def main():
    """Run all baseline comparisons."""
    # Compare on MovieLens 100K
    compare_with_baseline_100k()
    
    # Compare on MovieLens 1M
    compare_with_baseline_1m()
    
    # Generate comparison table
    generate_baseline_comparison_table()
    
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

