import numpy as np
import re

FEATURE_DIM = 8

def looks_like_email(s):
    return "@" in s and "." in s

def looks_like_phone_number(s):
    digits = [c for c in s if c.isdigit()]
    return len(digits) >= 7

def looks_like_url(s):
    return s.startswith("http://") or s.startswith("https://")

def looks_like_date(s):
    date_patterns = [
        r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
        r'^\d{2}/\d{2}/\d{4}$',  # MM/DD/YYYY
        r'^\d{2}-\d{2}-\d{4}$',  # MM-DD-YYYY
        r'^\d{4}/\d{2}/\d{2}$'   # YYYY/MM/DD
    ]
    for pattern in date_patterns:
        if re.match(pattern, s):
            return True
    return False

def looks_like_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def looks_like_ip_address(s):
    ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    return re.match(ip_pattern, s) is not None

def looks_like_currency(s):
    currency_symbols = ['$', '€', '£', '¥', '₹']
    return any(s.startswith(sym) for sym in currency_symbols)

def looks_like_zip_code(s):
    zip_pattern = r'^\d{5}(-\d{4})?$'
    return re.match(zip_pattern, s) is not None

def looks_like_name(s):
    return all(part.isalpha() for part in s.split() if part)

def looks_like_address(s):
    address_keywords = ['St', 'Street', 'Ave', 'Avenue', 'Blvd', 'Boulevard', 'Rd', 'Road', 'Ln', 'Lane', 'Dr', 'Drive']
    return any(keyword in s for keyword in address_keywords)

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