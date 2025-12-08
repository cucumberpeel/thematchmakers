import re
from difflib import SequenceMatcher
from urllib.parse import urlparse, urlunparse
import unicodedata, re


import pandas as pd
try:
    import bdikit as bdi
except Exception:
    bdi = None


def lexical_algorithm(source, targets):
    # Use bdikit edit-distance matcher if available, otherwise fallback
    # to a simple edit-distance proxy using SequenceMatcher.
    if bdi is not None:
        try:
            source_column = 'source'
            target_column = 'target'
            source_dataset = pd.DataFrame({source_column: [source] })
            target_dataset = pd.DataFrame({ target_column: targets })

            matches = bdi.match_values(
                                source_dataset,
                                target_dataset,
                                attribute_matches=(source_column, target_column),
                                method="edit_distance",
                            )

            return matches["target_value"].iloc[0]
        except Exception:
            pass

    # Fallback: choose candidate with highest SequenceMatcher ratio
    try:
        s = str(source)
        best, best_score = None, -1.0
        for t in targets:
            score = SequenceMatcher(None, s, str(t)).ratio()
            if score > best_score:
                best, best_score = t, score
        return best
    except Exception:
        return None


def semantic_algorithm(source, targets):
    dummy = True
    if dummy:
        return targets[0]  # Dummy
    source_column = 'source'
    target_column = 'target'
    source_dataset = pd.DataFrame({source_column: [source] })
    target_dataset = pd.DataFrame({ target_column: targets })
    
    matches = bdi.match_values(
                            source_dataset,
                            target_dataset,
                            attribute_matches=(source_column, target_column),
                            method="embedding",
                        )
    return matches["target_value"].iloc[0]
    

def llm_reasoning_algorithm(source, targets):
    # LLM reasoning is disabled for now to avoid oracle behaviour during
    # training. The original implementation (using bdikit) is commented
    # out below for reference. Instead return a neutral response with a
    # zero reward so the RL agent does not learn to prefer the LLM action.
    #
    # Original implementation:
    # source_column = 'source'
    # target_column = 'target'
    # source_dataset = pd.DataFrame({source_column: [source] })
    # target_dataset = pd.DataFrame({ target_column: targets })
    # print("WARNING: Use LLM reasoning only in production")
    # matches = bdi.match_values(
    #                         source_dataset,
    #                         target_dataset,
    #                         attribute_matches=(source_column, target_column),
    #                         method="llm",
    #                         method_args={"model_name": "deepinfra/openai/gpt-oss-120b"}
    #                     )
    # return matches["target_value"].iloc[0]

    # Neutral stub: no prediction, zero score, and explicit zero reward
    return None


def shingles_algorithm(source, targets, k=3):
    def norm(s):
        s = "" if s is None else str(s).lower().strip()
        s = re.sub(r"[^\w\s]", " ", s)
        return re.sub(r"\s+", " ", s)

    def shingles(s):
        s = norm(s)
        if not s: return set()
        if len(s) < k: return {s}
        return {s[i:i+k] for i in range(len(s)-k+1)}

    s_sh = shingles(source)
    best, best_score = None, -1.0
    for t in targets:
        t_sh = shingles(t)
        if not s_sh and not t_sh:
            score = 1.0
        elif not s_sh or not t_sh:
            score = 0.0
        else:
            inter = len(s_sh & t_sh); union = len(s_sh | t_sh)
            score = inter / union if union else 0.0
        if score > best_score:
            best, best_score = t, score
    return best



def regex_algorithm(source, targets, flags=re.IGNORECASE):
    if source is None:
        return None
    def _normalize(s):
        if s is None:
            return ""
        s = str(s)
        # Unicode fold + remove diacritics
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        # Lowercase and collapse whitespace
        s = s.casefold()
        s = re.sub(r"[\s\u00A0]+", " ", s)
        return s.strip()

    src_raw = " ".join(str(source).strip().split())
    src_norm = _normalize(src_raw)

    # 1) Try a flexible exact-match regex on the raw source (allow \s+ for spaces)
    try:
        pat_raw = re.escape(src_raw).replace(r"\ ", r"\s+")
        rx_raw = re.compile(pat_raw, flags)
        for t in targets:
            if t is None:
                continue
            try:
                if rx_raw.search(str(t)):
                    return t
            except re.error:
                continue
    except re.error:
        pass

    # 2) Try matching on normalized (diacritics-folded, lowercased) targets
    targets_norm = [None if t is None else _normalize(t) for t in targets]
    if src_norm:
        # exact substring on normalized
        for t_raw, t_norm in zip(targets, targets_norm):
            if not t_norm:
                continue
            if src_norm in t_norm:
                return t_raw

        # 3) Token-subsequence regex: tokens in order but allowing intervening words
        tokens = [tok for tok in re.split(r"\W+", src_norm) if tok]
        if tokens:
            # build pattern: \btoken1\b.*?\btoken2\b.*?\b...
            escaped = [re.escape(t) for t in tokens]
            pat_tokens = r"\b" + r"\b.*?\b".join(escaped) + r"\b"
            try:
                rx_tokens = re.compile(pat_tokens, flags | re.DOTALL)
                for t_raw, t_norm in zip(targets, targets_norm):
                    if not t_norm:
                        continue
                    if rx_tokens.search(t_norm):
                        return t_raw
            except re.error:
                pass

        # 4) Token overlap (Jaccard) fallback: choose target with highest overlap
        src_tokens_set = set(tokens)
        best_idx, best_j = -1, 0.0
        for i, t_norm in enumerate(targets_norm):
            if not t_norm:
                continue
            tgt_tokens = set([tok for tok in re.split(r"\W+", t_norm) if tok])
            if not tgt_tokens:
                continue
            inter = src_tokens_set & tgt_tokens
            union = src_tokens_set | tgt_tokens
            j = len(inter) / (len(union) + 1e-9)
            if j > best_j:
                best_j, best_idx = j, i
        if best_idx >= 0 and best_j >= 0.4:
            return targets[best_idx]

        # 5) Fuzzy fallback using SequenceMatcher on normalized strings
        best_idx, best_score = -1, 0.0
        for i, t_norm in enumerate(targets_norm):
            if not t_norm:
                continue
            score = SequenceMatcher(None, src_norm, t_norm).ratio()
            if score > best_score:
                best_score, best_idx = score, i
        if best_idx >= 0 and best_score >= 0.6:
            return targets[best_idx]

    return None


def identity_algorithm(source, targets):
    if source in targets:
        return source
    return None


# Removed: length_algorithm, prefix_algorithm, suffix_algorithm, range_algorithm
# These simple surface-form heuristics were removed to avoid duplication
# with more robust primitives (unit_normalizer, numeric_family, regex, etc.).

def categorical_algorithm(source, targets):
    src_cat = str(source).strip().lower()
    for t in targets:
        if str(t).strip().lower() == src_cat:
            return t
    return None

def boolean_algorithm(source, targets):
    # Removed per user request
    return None

def phone_algorithm(source, targets):
    # Removed per user request
    return None

def email_algorithm(source, targets):
    # Removed per user request
    return None


def url_algorithm(source, targets):
    def norm(u):
        try:
            p = urlparse(str(u).strip())
            scheme = (p.scheme or "http").lower()
            host = p.netloc.lower().lstrip("www.")
            path = (p.path or "/").rstrip("/") or "/"
            return urlunparse((scheme, host, path, "", "", ""))
        except:
            return None
    s = norm(source)
    if not s: return None
    for t in targets:
        if norm(t) == s:
            return t
    return None


def name_algorithm(source, targets):
    def normalize_name(n):
        n = str(n).strip().lower()
        n = re.sub(r"\s+", " ", n)
        return n
    src_name = normalize_name(source)
    for t in targets:
        tgt_name = normalize_name(t)
        if tgt_name == src_name:
            return t
    return None
# Removed: name_algorithm and address_algorithm
# These simple canonicalizers were removed to reduce primitive count
# and rely on more flexible text/regex methods.



def ssn_algorithm(source, targets):
    def normalize_ssn(s):
        s = str(s).strip()
        s = re.sub(r"\D", "", s)
        return s
    src_ssn = normalize_ssn(source)
    for t in targets:
        tgt_ssn = normalize_ssn(t)
        if tgt_ssn == src_ssn:
            return t
    return None

def isbn_algorithm(source, targets):
    # Removed per user request
    return None


def mac_address_algorithm(source, targets):
    # Removed per user request
    return None


def color_algorithm(source, targets):
    def normalize_color(c):
        c = str(c).strip().lower()
        c = re.sub(r"\s+", " ", c)
        return c
    src_color = normalize_color(source)
    for t in targets:
        tgt_color = normalize_color(t)
        if tgt_color == src_color:
            return t
    return None
# Removed: color_algorithm and rgb_algorithm
# Use hex_color_algorithm or more advanced color matching if needed.

# Removed: hex_color_algorithm
# Hex colour canonicalization removed in favor of a single `hex_algorithm` style
# primitive or external color-matching helpers when needed.

def filepath_algorithm(source, targets):
    def normalize_filepath(f):
        f = str(f).strip().lower()
        f = re.sub(r"\\+", "/", f)
        f = re.sub(r"/+", "/", f)
        return f
    src_fp = normalize_filepath(source)
    for t in targets:
        tgt_fp = normalize_filepath(t)
        if tgt_fp == src_fp:
            return t
    return None

def uuid_algorithm(source, targets):
    def normalize_uuid(u):
        u = str(u).strip().lower()
        u = re.sub(r"[^0-9a-f]", "", u)
        return u
    src_uuid = normalize_uuid(source)
    for t in targets:
        tgt_uuid = normalize_uuid(t)
        if tgt_uuid == src_uuid:
            return t
    return None

def hex_algorithm(source, targets):
    def normalize_hex(h):
        h = str(h).strip().lower()
        h = re.sub(r"[^0-9a-f]", "", h)
        return h
    src_hex = normalize_hex(source)
    for t in targets:
        tgt_hex = normalize_hex(t)
        if tgt_hex == src_hex:
            return t
    return None

# Removed: hex_algorithm
# Use external color/hex matching utilities or keep hex normalization
# in feature engineering if needed.

def accent_fold_algorithm(source, targets):
    def fold(s):
        s = unicodedata.normalize("NFKD", str(s))
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        s = s.casefold()
        s = re.sub(r"\s+", " ", s).strip()
        return s
    s = fold(source)
    for t in targets:
        if fold(t) == s:
            return t
    return None

def light_stem_algorithm(source, targets):
    def light(s):
        s = str(s).casefold()
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        # ultra-light “stem”: drop common plural/verb suffixes (very rough)
        s = re.sub(r"(es|s|ed|ing)\b", "", s)
        return s
    s = light(source)
    for t in targets:
        if light(t) == s:
            return t
    return None





