import os
from agent_lab import train_agent, load_agent, evaluate_agent
from dataset_formatter import read_split_datasets
from algorithms import (lexical_algorithm, semantic_algorithm, llm_reasoning_algorithm,
shingles_algorithm, regex_algorithm, ip_matcher, date_algorithm,
numeric_family_algorithm, identity_algorithm, categorical_algorithm, boolean_algorithm, phone_algorithm, email_algorithm,
url_algorithm, geo_pair_algorithm, zip_algorithm, credit_card_algorithm, ssn_algorithm, isbn_algorithm, mac_address_algorithm,
filepath_algorithm, uuid_algorithm,
unit_normalizer_algorithm, accent_fold_algorithm,
light_stem_algorithm)
from feature_extractor import FEATURE_DIM


general_costs = {'light': 0.05, 'moderate': 0.1, 'expensive': 0.7}
primitives = [
    ("lexical", lexical_algorithm, 'light'), 
    ("semantic", semantic_algorithm, 'moderate'), 
    ("llm", llm_reasoning_algorithm, 'expensive'),
    ("shingles", shingles_algorithm, 'light'),
    ("regex", regex_algorithm, 'moderate'),
    ("ip", ip_matcher, 'light'),
    ("date", date_algorithm, 'light'),
    ("numeric_family", numeric_family_algorithm, 'light'),
    ("identity", identity_algorithm, 'light'),
    ("categorical", categorical_algorithm, 'light'),
    ("boolean", boolean_algorithm, 'light'),
    ("phone", phone_algorithm, 'light'),
    ("email", email_algorithm, 'light'),
    ("url", url_algorithm, 'light'),
    ("geo_pair", geo_pair_algorithm, 'light'),
    ("zip", zip_algorithm, 'light'),
    ("credit_card", credit_card_algorithm, 'light'),
    ("ssn", ssn_algorithm, 'light'),
    ("isbn", isbn_algorithm, 'light'),
    ("mac_address", mac_address_algorithm, 'light'),
    ("filepath", filepath_algorithm, 'light'),
    ("uuid", uuid_algorithm, 'light'),
    ("unit_normalizer", unit_normalizer_algorithm, 'light'),
    ("accent_fold", accent_fold_algorithm, 'light'),
    ("light_stem", light_stem_algorithm, 'light')
]

def evaluate_rl_method(train_dataset, test_dataset, load_checkpoint=True, checkpoint_dir="model_checkpoint"):
    checkpoint_dir = os.path.join(os.path.dirname(__file__), checkpoint_dir)
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
            primitive_costs=primitive_costs,
            dataset=train_dataset,
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
            dataset=train_dataset,
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
        test_dataset=test_dataset,
        feature_dim=feature_dim,
        max_steps=max_steps
    )


def evaluate_individual_methods(test_dataset):
    print("\nEvaluating individual methods...")
    for name, method, _ in primitives:
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
    train_dataset, test_dataset = read_split_datasets(autofj=True, ss=False, wt=False, kbwt=False)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    #evaluate_individual_methods(test_dataset)
    evaluate_rl_method(train_dataset, test_dataset)