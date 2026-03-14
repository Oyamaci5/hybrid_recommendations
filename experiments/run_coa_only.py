"""
Run COA-SGD-MF only on both MovieLens 100K and 1M datasets.
If HHO results exist, comparison will be shown.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.compare_coa_hho import compare_coa_hho_100k, compare_coa_hho_1m, generate_comparison_table


def main(run_coa: bool = True):
    """
    Run COA only on both datasets and compare with HHO if available.
    
    Args:
        run_coa: If True, run COA experiments. If False, use existing results.
    """
    print("=" * 60)
    print("RUNNING COA-SGD-MF ONLY")
    print("=" * 60)
    if run_coa:
        print("COA will be run on both datasets.")
    else:
        print("Using existing COA results (if available).")
    print("If HHO results exist, comparison will be shown.")
    print("=" * 60)
    
    # Run COA on MovieLens 100K (don't run HHO)
    compare_coa_hho_100k(run_coa=run_coa, run_hho=False)
    
    # Run COA on MovieLens 1M (don't run HHO)
    compare_coa_hho_1m(run_coa=run_coa, run_hho=False)
    
    # Generate comparison table
    generate_comparison_table()
    
    print("\n" + "=" * 60)
    print("COA EXPERIMENTS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run COA-SGD-MF only on MovieLens datasets')
    parser.add_argument('--skip-run', action='store_true', default=False,
                       help='Skip running COA, use existing results if available')
    args = parser.parse_args()
    
    main(run_coa=not args.skip_run)

