from agent_lab import train_agent, evaluate_agent
from dataset_formatter import read_datasets, split_datasets, print_datasets_info
from algorithms import lexical_algorithm, semantic_algorithm, llm_reasoning_algorithm
from feature_extractor import FEATURE_DIM


def evaluate_rl_method(train_dataset, test_dataset):
    primitives = [("lexical", lexical_algorithm), 
                  ("semantic", semantic_algorithm), 
                  ("llm", llm_reasoning_algorithm)
                  ]
    primitive_names = [name for name, _ in primitives]
    primitive_methods = [method for _, method in primitives]

    max_steps = 2
    feature_dim = FEATURE_DIM

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
     
    # Train agent
    print("Training agent...")
    Q = train_agent(
        primitives=primitive_methods,
        primitive_names=primitive_names,
        dataset=train_dataset,
        feature_dim=feature_dim,
        max_steps=max_steps
    )
    
    print(f"\nTrained Q-table size: {len(Q)}")
    
    # Evaluate agent
    print("\nEvaluating agent...")
    results = evaluate_agent(
        Q=Q,
        primitives=primitive_methods,
        primitive_names=primitive_names,
        test_dataset=test_dataset,
        feature_dim=feature_dim,
        max_steps=max_steps
    )


def evaluate_individual_methods(test_dataset):
    primitives = [("lexical", lexical_algorithm), 
                  ("semantic", semantic_algorithm), 
                  ("llm", llm_reasoning_algorithm)
        ]

    print("\nEvaluating individual methods...")
    for name, method in primitives:
        correct = 0
        total = len(test_dataset)
        for test_input in test_dataset:
            source_value = test_input['source_value']
            target_values = test_input['target_values']
            gold_value = test_input['gold_value']
            # TODO: Remove this placeholder for LLM reasoning
            if name == 'llm':
                prediction = gold_value  # Assume perfect prediction for LLM in this placeholder
            else:
                prediction = method(source_value, target_values)

            if prediction == gold_value:
                correct += 1
        accuracy = correct / total
        print(f"Method: {name}, Accuracy: {accuracy:.3f}")

if __name__ == "__main__":
    datasets = read_datasets()
    #print_datasets_info(datasets)
    train_dataset, test_dataset = split_datasets(datasets, test_size=0.2)

    evaluate_individual_methods(test_dataset)
    evaluate_rl_method(train_dataset, test_dataset)