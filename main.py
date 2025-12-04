from agent_lab import train_agent, evaluate_agent
from dataset_formatter import read_datasets, split_datasets, print_datasets_info
from algorithms import (lexical_algorithm, semantic_algorithm, llm_reasoning_algorithm,
shingles_algorithm, regex_algorithm, ip_country_matcher, ip_subnet_matcher, date_algorithm,
numeric_algorithm, identity_algorithm, length_algorithm, prefix_algorithm, suffix_algorithm,
range_algorithm, categorical_algorithm, boolean_algorithm, phone_algorithm, email_algorithm,
url_algorithm, geo_pair_algorithm, currency_algorithm, name_algorithm, address_algorithm,
zip_algorithm, credit_card_algorithm, ssn_algorithm, isbn_algorithm, mac_address_algorithm,
color_algorithm, rgb_algorithm, hex_algorithm, percentage_algorithm, filepath_algorithm, uuid_algorithm,
inches_to_cm_algorithm, cm_to_inches_algorithm, weight_kg_to_lb_algorithm, weight_lb_to_kg_algorithm,
temperature_c_to_f_algorithm, temperature_f_to_c_algorithm, accent_fold_algorithm,
light_stem_algorithm)


from feature_extractor import FEATURE_DIM


def evaluate_rl_method(train_dataset, test_dataset):
    primitives = [("lexical", lexical_algorithm), 
                  ("semantic", semantic_algorithm), 
                  ("llm", llm_reasoning_algorithm),
                  ("shingles", shingles_algorithm),
                  ("regex", regex_algorithm),
                    ("ip_country", ip_country_matcher),
                    ("ip_subnet", ip_subnet_matcher),
                    ("date", date_algorithm),
                    ("numeric", numeric_algorithm),
                    ("identity", identity_algorithm),
                    ("length", length_algorithm),
                    ("prefix", prefix_algorithm),
                    ("suffix", suffix_algorithm),
                    ("range", range_algorithm),
                    ("categorical", categorical_algorithm),
                    ("boolean", boolean_algorithm),
                    ("phone", phone_algorithm),
                    ("email", email_algorithm),
                    ("url", url_algorithm),
                    ("geo_pair", geo_pair_algorithm),
                    ("currency", currency_algorithm),
                    ("name", name_algorithm),
                    ("address", address_algorithm),
                    ("zip", zip_algorithm),
                    ("credit_card", credit_card_algorithm),
                    ("ssn", ssn_algorithm),
                    ("isbn", isbn_algorithm),
                    ("mac_address", mac_address_algorithm),
                    ("color", color_algorithm),
                    ("rgb", rgb_algorithm),
                    ("hex", hex_algorithm),
                    ("percentage", percentage_algorithm),
                    ("filepath", filepath_algorithm),
                    ("uuid", uuid_algorithm),
                    ("inches_to_cm", inches_to_cm_algorithm),
                    ("cm_to_inches", cm_to_inches_algorithm),
                    ("weight_kg_to_lb", weight_kg_to_lb_algorithm),
                    ("weight_lb_to_kg", weight_lb_to_kg_algorithm),
                    ("temperature_c_to_f", temperature_c_to_f_algorithm),
                    ("temperature_f_to_c", temperature_f_to_c_algorithm),
                    ("accent_fold", accent_fold_algorithm),
                    ("light_stem", light_stem_algorithm)                
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
                  ("llm", llm_reasoning_algorithm),
                    ("shingles", shingles_algorithm),
                    ("regex", regex_algorithm),
                    ("ip_country", ip_country_matcher),
                    ("ip_subnet", ip_subnet_matcher),
                    ("date", date_algorithm),
                    ("numeric", numeric_algorithm),
                    ("identity", identity_algorithm),
                    ("length", length_algorithm),
                    ("prefix", prefix_algorithm),
                    ("suffix", suffix_algorithm),
                    ("range", range_algorithm),
                    ("categorical", categorical_algorithm),
                    ("boolean", boolean_algorithm),
                    ("phone", phone_algorithm),
                    ("email", email_algorithm),
                    ("url", url_algorithm),
                    ("geo_pair", geo_pair_algorithm),
                    ("currency", currency_algorithm),
                    ("name", name_algorithm),
                    ("address", address_algorithm),
                    ("zip", zip_algorithm),
                    ("credit_card", credit_card_algorithm),
                    ("ssn", ssn_algorithm),
                    ("isbn", isbn_algorithm),
                    ("mac_address", mac_address_algorithm),
                    ("color", color_algorithm),
                    ("rgb", rgb_algorithm),
                    ("hex", hex_algorithm),
                    ("percentage", percentage_algorithm),
                    ("filepath", filepath_algorithm),
                    ("uuid", uuid_algorithm),
                    ("inches_to_cm", inches_to_cm_algorithm),
                    ("cm_to_inches", cm_to_inches_algorithm),
                    ("weight_kg_to_lb", weight_kg_to_lb_algorithm),
                    ("weight_lb_to_kg", weight_lb_to_kg_algorithm),
                    ("temperature_c_to_f", temperature_c_to_f_algorithm),
                    ("temperature_f_to_c", temperature_f_to_c_algorithm),
                    ("accent_fold", accent_fold_algorithm),
                    ("light_stem", light_stem_algorithm)
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
                try:
                    prediction = method(source_value, target_values)
                except Exception as e:
                    print(f"Error in {name} method: {e}")
                    print(f"Input: {source_value}, {target_values[:3]}")
                    return

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