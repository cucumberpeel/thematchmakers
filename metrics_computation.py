"""
Compute and report metrics across all datasets.

Runs evaluation on autofj, ss, wt, and kbwt datasets, computes metrics for each,
aggregates overall statistics, and writes results to metrics_output.txt.

Usage:
    python metrics_computation.py
"""
import os
import sys
from datetime import datetime
from dataset_formatter import read_split_datasets
from metrics import compute_classification_metrics, compute_aggregate_metrics
from algorithms import (lexical_algorithm, semantic_algorithm, llm_reasoning_algorithm,
shingles_algorithm, regex_algorithm, identity_algorithm,
accent_fold_algorithm,)


general_costs = {'light': 0.075, 'moderate': 0.15, 'expensive': 0.3}
primitives = [
    ("lexical", lexical_algorithm, 'light'), 
    ("semantic", semantic_algorithm, 'moderate'), 
    ("llm", llm_reasoning_algorithm, 'expensive'),
    ("shingles", shingles_algorithm, 'light'),
    ("regex", regex_algorithm, 'moderate'),
    ("identity", identity_algorithm, 'light'),
    ("accent_fold", accent_fold_algorithm, 'light')
]


DATASETS = ["autofj", "ss", "wt", "kbwt"]
DATASET_FLAGS = {
    "autofj": {"autofj": True,  "ss": False, "wt": False, "kbwt": False},
    "ss":     {"autofj": False, "ss": True,  "wt": False, "kbwt": False},
    "wt":     {"autofj": False, "ss": False, "wt": True,  "kbwt": False},
    "kbwt":   {"autofj": False, "ss": False, "wt": False, "kbwt": True},
}


def evaluate_dataset(dataset_name: str, method_name: str, method_func):
    """Evaluate a single method on a single dataset."""
    flags = DATASET_FLAGS[dataset_name]
    train_dataset, test_dataset = read_split_datasets(
        autofj=flags["autofj"],
        ss=flags["ss"],
        wt=flags["wt"],
        kbwt=flags["kbwt"],
    )
    
    if not test_dataset:
        return None
    
    golds, preds = [], []
    for sample in test_dataset:
        source_value = sample["source_value"]
        target_values = sample["target_values"]
        gold_value = sample["gold_value"]
        
        if method_name == "llm":
            prediction = method_func(source_value, target_values, sample["source_column"], sample["target_column"])
        else:
            try:
                prediction = method_func(source_value, target_values)
            except Exception:
                prediction = None
        
        golds.append(gold_value)
        preds.append(prediction)
    
    return compute_classification_metrics(golds, preds)


def main():
    output_path = "metrics_output.txt"
    
    # Store results: dataset -> method -> metrics
    all_results = {ds: {} for ds in DATASETS}
    
    print("=" * 80)
    print("RUNNING METRICS COMPUTATION ACROSS ALL DATASETS")
    print("=" * 80)
    
    for method_name, method_func, _ in primitives:
        print(f"\nEvaluating method: {method_name}")
        method_dataset_results = {}
        
        for dataset_name in DATASETS:
            print(f"  Dataset: {dataset_name}...", end=" ")
            metrics = evaluate_dataset(dataset_name, method_name, method_func)
            
            if metrics is None:
                print("SKIPPED (no data)")
                continue
            
            all_results[dataset_name][method_name] = metrics
            method_dataset_results[dataset_name] = metrics
            print(f"F1={metrics['f1']:.3f}")
        
        # Compute aggregate for this method across datasets
        if method_dataset_results:
            agg = compute_aggregate_metrics(method_dataset_results)
            print(f"  → Aggregate F1: {agg['f1_mean']:.3f} ± {agg['f1_std']:.3f}")
    
    # Write results to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"METRICS COMPUTATION REPORT\n")
        f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n")
        f.write("=" * 80 + "\n\n")
        
        # Per-dataset, per-method breakdown
        for dataset_name in DATASETS:
            f.write(f"\n{'='*80}\n")
            f.write(f"DATASET: {dataset_name.upper()}\n")
            f.write(f"{'='*80}\n\n")
            
            if not all_results[dataset_name]:
                f.write("  No results available.\n")
                continue
            
            f.write(f"{'Method':<15} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Cov':>7}\n")
            f.write("-" * 80 + "\n")
            
            for method_name in [m[0] for m in primitives]:
                if method_name not in all_results[dataset_name]:
                    continue
                m = all_results[dataset_name][method_name]
                f.write(
                    f"{method_name:<15} "
                    f"{m['accuracy']:>7.3f} "
                    f"{m['precision']:>7.3f} "
                    f"{m['recall']:>7.3f} "
                    f"{m['f1']:>7.3f} "
                    f"{m['coverage']:>7.3f}\n"
                )
        
        # Overall aggregate per method
        f.write(f"\n\n{'='*80}\n")
        f.write("AGGREGATE METRICS ACROSS DATASETS (per method)\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"{'Method':<15} {'Acc Mean':>10} {'Prec Mean':>10} {'Rec Mean':>10} {'F1 Mean':>10}\n")
        f.write("-" * 80 + "\n")
        
        for method_name, _, _ in primitives:
            method_dataset_results = {}
            for ds in DATASETS:
                if method_name in all_results[ds]:
                    method_dataset_results[ds] = all_results[ds][method_name]
            
            if method_dataset_results:
                agg = compute_aggregate_metrics(method_dataset_results)
                f.write(
                    f"{method_name:<15} "
                    f"{agg['accuracy_mean']:>10.3f} "
                    f"{agg['precision_mean']:>10.3f} "
                    f"{agg['recall_mean']:>10.3f} "
                    f"{agg['f1_mean']:>10.3f}\n"
                )
    
    print(f"\n{'='*80}")
    print(f"Metrics written to: {output_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
