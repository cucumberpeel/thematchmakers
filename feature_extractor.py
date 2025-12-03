import numpy as np

FEATURE_DIM = 8

def compute_features(source_value, target_values):
    
    # String characteristics
    length_source = len(source_value)
    num_words_source = len(source_value.split())
    is_acronym_source = source_value.isupper()
    digit_ratio_source = sum(c.isdigit() for c in source_value) / length_source
    letter_ratio_source = sum(c.isalpha() for c in source_value) / length_source

    # Candidate set characteristics
    num_candidates = len(target_values)
    avg_candidate_length = np.mean([len(t) for t in target_values])
    std_candidate_length = np.std([len(t) for t in target_values])

    features = [
        length_source,
        num_words_source,
        is_acronym_source,
        digit_ratio_source,
        letter_ratio_source,
        num_candidates,
        avg_candidate_length,
        std_candidate_length
    ]

    return features