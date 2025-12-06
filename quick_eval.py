"""
Quick evaluation script

Runs a short evaluation similar to `main.py` but only on the first N
examples (default 5) from the datasets returned by
`read_split_datasets`. By default it runs the individual-method
evaluation (no RL). Pass `--run-rl` to also run the RL evaluation
on the small subset (be aware RL training still uses the same
train_agent implementation and may be slower).

Usage:
    python quick_eval.py            # runs individual-methods on 5 samples
    python quick_eval.py --n 3      # run on first 3 samples
    python quick_eval.py --run-rl   # also run RL (may be slow)
"""
import argparse
from dataset_formatter import read_split_datasets


def main(n=5, run_rl=False):
    # Read datasets (same defaults as main.py)
    train_dataset, test_dataset = read_split_datasets(autofj=True, ss=True, wt=True, kbwt=True)

    if not test_dataset:
        print("No test dataset available (read_split_datasets returned empty).")
        return

    n = max(1, int(n))
    test_subset = test_dataset[:n]
    train_subset = train_dataset[:n] if train_dataset else []

    print(f"Running quick evaluation on {len(test_subset)} samples (from test set)")

    # Import main's evaluation helpers (safe: main.py guards execution under __main__)
    try:
        from main import evaluate_individual_methods, evaluate_rl_method
    except Exception:
        # If importing main fails for any reason, provide a minimal fallback
        print("Could not import evaluation helpers from main.py. Exiting.")
        return

    # Run individual-method evaluation on the small test subset
    print("\n--- Individual methods evaluation (quick) ---")
    evaluate_individual_methods(test_subset)

    if run_rl:
        print("\n--- RL evaluation (quick) ---")
        # Train/evaluate RL on the small subsets (may still be slow)
        evaluate_rl_method(train_subset, test_subset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick evaluation (3-5 samples)")
    parser.add_argument("--n", type=int, default=5, help="Number of samples to evaluate")
    parser.add_argument("--run-rl", action="store_true", help="Also run RL evaluation (slower)")
    args = parser.parse_args()

    main(n=args.n, run_rl=args.run_rl)
