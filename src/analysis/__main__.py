"""
Main entry point for analysis module.
"""

from pathlib import Path
import sys

from . import plot_all, print_summary


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.analysis <run_directory>")
        print("\nExample:")
        print("  python -m src.analysis logs/ablation_no_planning/run_2025-11-13_12-00-00/")
        sys.exit(1)

    run_dir = Path(sys.argv[1])

    if not run_dir.exists():
        print(f"Error: {run_dir} does not exist")
        sys.exit(1)

    if not (run_dir / 'metrics.csv').exists():
        print(f"Error: No metrics.csv found in {run_dir}")
        sys.exit(1)

    print(f"Analyzing run: {run_dir}")
    print()

    # Generate plots
    plot_all(run_dir)

    # Print summary
    print_summary(run_dir)


if __name__ == '__main__':
    main()
