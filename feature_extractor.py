import numpy as np
import re
from difflib import SequenceMatcher

try:
    from rapidfuzz import fuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False

try:
    import spacy
    _HAS_SPACY = True
except Exception:
    _HAS_SPACY = False

FEATURE_DIM = 16  # updated to include entity features
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


from collections import Counter
# Lazy-load spacy model on first use
_SPACY_NLP = None
_ALL_NER_LABELS = {}

def _get_nlp():
    global _SPACY_NLP
    global _ALL_NER_LABELS
    if _SPACY_NLP is None and _HAS_SPACY:
        try:
            _SPACY_NLP = spacy.load("en_core_web_sm")
            print("[feature_extractor] Spacy model 'en_core_web_sm' loaded successfully.")
            _ALL_NER_LABELS = {l: i for i, l in enumerate(_SPACY_NLP.get_pipe("ner").labels)}
        except Exception as e:
            print(f"[feature_extractor] Failed to load spacy model: {e}")
            pass
    return _SPACY_NLP

def get_entity(text):
    """Return the most common named entity recognized by spacy."""
    nlp = _get_nlp()

    if nlp is None:
        return -1
    doc = nlp(str(text))
    entity_ids = [_ALL_NER_LABELS.get(ent.label_, -1) for ent in doc.ents]
    if len(entity_ids) == 0:
        return -1
    
    most_common = Counter(entity_ids).most_common(1)[0][0]
    return most_common

def count_entities(text):
    """Return count of named entities in text."""
    nlp = _get_nlp()
    if nlp is None:
        return 0
    doc = nlp(str(text))
    return len(doc.ents)

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
    is_date_source = looks_like_date(source_value)
    is_numeric_source = looks_like_numeric(source_value)
    is_name_source = looks_like_name(source_value)
    is_address_source = looks_like_address(source_value)

    # Entity recognition features (spacy)
    source_entity = get_entity(source_value)

    # Candidate set characteristics
    num_candidates = len(target_values)
    avg_candidate_length = np.mean([len(t) for t in target_values])

    # Entity features for candidates
    candidate_entities = [get_entity(t) for t in target_values[:5]]
    if len(candidate_entities) == 0:
        candidate_entity = -1
    else:
        candidate_entity = Counter(candidate_entities).most_common(1)[0][0]

    features = [
        num_candidates,
        avg_candidate_length,
        is_date_source,
        is_numeric_source,
        is_name_source,
        is_address_source,
        # Entity features
        source_entity,
        candidate_entity,
    ]

    # --- Pairwise lexical features (aggregated across candidate set) ---
    def _norm(s):
        return s.strip()

    def _tokens(s):
        return [t for t in re.split(r"\W+", s.lower()) if t]

    def _lev_sim(a, b):
        if _HAS_RAPIDFUZZ:
            try:
                return fuzz.ratio(a, b) / 100.0
            except Exception:
                pass
        # Fallback to SequenceMatcher ratio in [0,1]
        return SequenceMatcher(None, a, b).ratio()

    def _lcs_norm(a, b):
        # longest common substring length normalized by max len
        if not a or not b:
            return 0.0
        la, lb = len(a), len(b)
        # dynamic programming for longest common substring
        dp = [0] * (lb + 1)
        best = 0
        for i in range(1, la + 1):
            prev = 0
            for j in range(1, lb + 1):
                temp = dp[j]
                if a[i-1] == b[j-1]:
                    dp[j] = prev + 1
                    if dp[j] > best:
                        best = dp[j]
                else:
                    dp[j] = 0
                prev = temp
        return best / max(la, lb)

    lev_sims = []
    token_jaccards = []
    prefix_matches = []
    suffix_matches = []
    lower_exact_matches = []
    lcs_norms = []

    s_norm = _norm(source_value)
    s_lower = s_norm.lower()
    s_tokens = _tokens(s_norm)
    s_first = s_tokens[0] if s_tokens else ""
    s_last = s_tokens[-1] if s_tokens else ""

    for t in target_values:
        t_norm = _norm(t)
        t_lower = t_norm.lower()
        t_tokens = _tokens(t_norm)

        # exact (case-sensitive) is already implicit in candidates but keep lower-case check
        lower_exact_matches.append(1.0 if s_lower == t_lower else 0.0)

        # Levenshtein / fuzzy similarity
        lev_sims.append(_lev_sim(s_norm, t_norm))

        # token Jaccard
        union = set(s_tokens) | set(t_tokens)
        inter = set(s_tokens) & set(t_tokens)
        token_jaccards.append(len(inter) / (len(union) + 1e-9))

        # prefix / suffix token matches
        t_first = t_tokens[0] if t_tokens else ""
        t_last = t_tokens[-1] if t_tokens else ""
        prefix_matches.append(1.0 if s_first and (s_first == t_first) else 0.0)
        suffix_matches.append(1.0 if s_last and (s_last == t_last) else 0.0)

        # longest common substring normalized
        lcs_norms.append(_lcs_norm(s_norm, t_norm))

    # Aggregate statistics across candidate set: max and mean for each numeric metric
    def _agg(nums):
        if not nums:
            return 0.0, 0.0
        arr = np.array(nums, dtype=float)
        return float(np.max(arr)), float(np.mean(arr))

    lev_max, lev_mean = _agg(lev_sims)
    jaccard_max, jaccard_mean = _agg(token_jaccards)
    lcs_max, lcs_mean = _agg(lcs_norms)

    pairwise_features = [
        lev_max,
        lev_mean,
        jaccard_max,
        jaccard_mean,
        lcs_max,
        lcs_mean
    ]

    features.extend(pairwise_features)

    # --- Char n-gram TF-IDF cosine similarities (char_wb) ---
    try:
        texts = [s_norm] + [_norm(t) for t in target_values]
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
        X = vec.fit_transform(texts)
        if X.shape[0] > 1:
            # cosine_similarity between source (row 0) and each candidate (rows 1:)
            cosines = cosine_similarity(X[0], X[1:]).ravel()
            tfidf_max = float(np.max(cosines))
            tfidf_mean = float(np.mean(cosines))
        else:
            tfidf_max, tfidf_mean = 0.0, 0.0
    except Exception:
        tfidf_max, tfidf_mean = 0.0, 0.0

    features.append(tfidf_max)
    features.append(tfidf_mean)

    return features