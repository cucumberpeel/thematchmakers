import os
import sys
import json
from datetime import datetime
from agent_lab import train_agent, load_agent, evaluate_agent
from agent_environment import ValueMatchingEnv
from dataset_formatter import read_split_datasets, save_formatted_datasets
from algorithms import (lexical_algorithm, semantic_algorithm, llm_reasoning_algorithm,
shingles_algorithm, regex_algorithm, identity_algorithm, categorical_algorithm,
url_algorithm, ssn_algorithm,
filepath_algorithm, uuid_algorithm,
accent_fold_algorithm,
light_stem_algorithm)
from feature_extractor import FEATURE_DIM
from metrics import compute_classification_metrics


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

def evaluate_rl_method(train_file_path, test_file_path, load_checkpoint=False, checkpoint_dir="model_checkpoint"):
    checkpoint_dir = os.path.join(os.path.dirname(__file__), checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    primitive_names = [name for name, _, _ in primitives]
    primitive_methods = [method for _, method, _ in primitives]
    primitive_costs = [general_costs[cost] for _, _, cost in primitives]

    max_steps = 2
    feature_dim = FEATURE_DIM

    if load_checkpoint:
        print("Loading trained agent...")
        algo = load_agent(
            checkpoint_dir=checkpoint_dir,
            primitives=primitive_methods,
            primitive_names=primitive_names,
            primitive_costs=primitive_costs,
            dataset=train_file_path,
            feature_dim=feature_dim,
            max_steps=max_steps
        )
    else:
        # Train agent
        print("Training agent...")
        algo = train_agent(
            checkpoint_dir=checkpoint_dir,
            primitives=primitive_methods,
            primitive_names=primitive_names,
            primitive_costs=primitive_costs,
            dataset=train_file_path,
            feature_dim=feature_dim,
            max_steps=max_steps
        )
    

    # Evaluate agent
    print("\nEvaluating agent...")
    results = evaluate_agent(
        algo=algo,
        primitives=primitive_methods,
        primitive_names=primitive_names,
        primitive_costs=primitive_costs,
        test_dataset_path=test_file_path,
        feature_dim=feature_dim,
        max_steps=max_steps
    )

    # Compute overall metrics in main using the trained agent
    golds, preds = run_eval_for_metrics(
        algo=algo,
        primitives=primitive_methods,
        primitive_names=primitive_names,
        primitive_costs=primitive_costs,
        test_dataset_path=test_file_path,
        feature_dim=feature_dim,
        max_steps=max_steps
    )
    metrics = compute_classification_metrics(golds, preds)
    print("\n=== Metrics (computed in main) ===")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1:        {metrics['f1']:.3f}")
    print(f"Coverage:  {metrics['coverage']:.3f}")

    # Persist metrics to file for later inspection
    metrics_payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset": os.path.basename(test_file_path),
        "metrics": metrics,
        "count": len(golds),
    }
    metrics_path = os.path.join(checkpoint_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as mf:
        json.dump(metrics_payload, mf, indent=2)
    print(f"Metrics written to {metrics_path}")

    return results


def run_eval_for_metrics(algo, primitives, primitive_names, primitive_costs, test_dataset_path, feature_dim, max_steps):
    """Run evaluation to collect gold/pred pairs for metric computation (keeps agent code untouched)."""
    with open(test_dataset_path, "r") as f:
        test_dataset = json.load(f)

    golds, preds = [], []

    for sample in test_dataset:
        env = ValueMatchingEnv({
            'primitives': primitives,
            'primitive_names': primitive_names,
            'costs': primitive_costs,
            'dataset': [sample],
            'feature_dim': feature_dim,
            'max_steps': max_steps
        })

        state, info = env.reset()
        done = False
        final_pred = None
        while not done:
            action = algo.compute_single_action(state, explore=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            final_pred = info.get('predicted')

        golds.append(sample['gold_value'])
        preds.append(final_pred)

    return golds, preds


def evaluate_individual_methods(test_dataset):
    print("\nEvaluating individual methods...")
    for name, method, _ in primitives:
        print(f"\nEvaluating method: {name}")
        correct = 0
        total = len(test_dataset)
        golds = []
        preds = []
        for test_input in test_dataset:
            source_value = test_input['source_value']
            target_values = test_input['target_values']
            gold_value = test_input['gold_value']
            # TODO: Remove this placeholder for LLM reasoning
            if name == 'llm':
                prediction = gold_value  # Assume perfect prediction for LLM in this placeholder
            else:
                try:
                    prediction = method(source_value, target_values)
                except Exception as e:
                    print(f"Error in {name} method: {e}")
                    print(f"Input: {source_value}, {target_values[:3]}")
                    return

            if prediction == gold_value:
                correct += 1
            golds.append(gold_value)
            preds.append(prediction)
        accuracy = correct / total
        #prf = compute_prf(golds, preds)
        print(
            f"Method: {name}, Accuracy: {accuracy:.3f}, "
            #f"Precision: {prf['precision']:.3f}, Recall: {prf['recall']:.3f}, F1: {prf['f1']:.3f}"
        )

if __name__ == "__main__":
    dataset_arg = sys.argv[1].lower()

    # Map argument → parameters for read_split_datasets
    dataset_flags = {
        "autofj":  {"autofj": True,  "ss": False, "wt": False, "kbwt": False},
        "ss":      {"autofj": False, "ss": True,  "wt": False, "kbwt": False},
        "wt":      {"autofj": False, "ss": False, "wt": True,  "kbwt": False},
        "kbwt":    {"autofj": False, "ss": False, "wt": False, "kbwt": True}
    }

    if dataset_arg not in dataset_flags:
        raise ValueError(
            f"Unknown dataset '{dataset_arg}'. "
            "Choose from: autofj, ss, wt, kbwt."
        )

    # --- Read dataset ---
    flags = dataset_flags[dataset_arg]

    train_dataset, test_dataset = read_split_datasets(
        autofj = flags["autofj"],
        ss     = flags["ss"],
        wt     = flags["wt"],
        kbwt   = flags["kbwt"]
    )
    formatted_dataset_dir = "formatted_datasets" + "/" + dataset_arg
    train_file_path, test_file_path = save_formatted_datasets(train_dataset, test_dataset, formatted_dataset_dir)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    #evaluate_individual_methods(test_dataset)
    checkpoint_dir = f"model_checkpoints/{dataset_arg}"
    evaluate_rl_method(train_file_path, test_file_path, checkpoint_dir=checkpoint_dir)