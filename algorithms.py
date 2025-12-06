import re
from difflib import SequenceMatcher
import ipaddress
import math
from urllib.parse import urlparse, urlunparse
import unicodedata, re
import warnings


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

def ip_matcher(source, targets, cidr_map=None, prefix=24, **kw):
    """Unified IP matcher.

    - If `cidr_map` is provided: treat `targets` as country labels and return
      the country whose CIDR contains `source` (longest-prefix match).
    - If `cidr_map` is not provided: treat `targets` as IP addresses and
      return the first target in the same `/prefix` subnet as `source`.

    Always return a single matched value or `None`.
    """
    # validate source
    try:
        ip_addr = ipaddress.ip_address(str(source))
    except Exception:
        return None

    # CIDR map mode: return country label
    if cidr_map:
        best_country, best_prefix = None, -1
        for country, cidrs in cidr_map.items():
            for c in cidrs:
                try:
                    net = ipaddress.ip_network(c, strict=False)
                except Exception:
                    continue
                if ip_addr in net and net.prefixlen > best_prefix:
                    best_prefix, best_country = net.prefixlen, country
        return best_country if best_country in set(map(str, targets)) else None

    # Subnet mode: match IP targets by subnet
    try:
        net_src = ipaddress.ip_network(f"{source}/{prefix}", strict=False)
    except Exception:
        return None

    for t in targets:
        try:
            if ipaddress.ip_network(f"{t}/{prefix}", strict=False) == net_src:
                return t
        except Exception:
            continue
    return None

def date_algorithm(source, targets, date_only=True, tolerance=None):
    """
    - date_only=True → compare calendar dates (ignore time-of-day)
    - tolerance=None → exact equality; else pd.Timedelta like '1D' or '2H'
    """
    try:
        # Suppress pandas' "Could not infer format" UserWarning which is
        # noisy when parsing heterogeneous date formats. We only ignore
        # that specific warning here; other warnings will still surface.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Could not infer format",
                category=UserWarning,
            )
            s = pd.to_datetime(pd.Series([source]), errors="coerce", utc=True)
            t = pd.to_datetime(pd.Series(list(targets)), errors="coerce", utc=True)

        if date_only:
            s = s.dt.normalize()
            t = t.dt.normalize()

        s_val = s.iloc[0]
        if pd.isna(s_val):
            return None

        if tolerance is None:
            mask = t.eq(s_val)
            if mask.any():
                idx = int(mask.idxmax())
                return targets[idx]
            return None
        else:
            diff = (t - s_val).abs()
            ok = diff <= pd.to_timedelta(tolerance)
            if ok.any():
                idx = int(diff[ok].idxmin())
                return targets[idx]
            return None
    except Exception:
        # Swallow any parsing/processing errors and return None instead of
        # allowing an exception to propagate and trigger error printing elsewhere.
        return None


def numeric_family_algorithm(source, targets, abs_tol=None, rel_tol=1e-3):
    """Unified numeric matcher.

    Handles plain numbers, currency-formatted values (e.g. "$1,234.56"),
    and percentages (e.g. "12%" or "12.3 %"). The function strips common
    symbols (`,`, `$`, `€`, `£`, `%`), parses numeric values (including
    parentheses-as-negative), and then compares values using an absolute
    and relative tolerance. If `abs_tol` is None, sensible defaults are used
    per-type (currency/percent vs raw number).
    Returns the first matching target that satisfies the tolerance, or None.
    """
    def parse(x):
        if x is None:
            return None, None
        s = str(x).strip()
        if s == "":
            return None, None

        is_percent = "%" in s
        is_currency = any(ch in s for ch in ("$", "€", "£"))

        # remove thousands separators and currency/percent symbols
        s2 = s.replace(",", "")
        s2 = s2.replace("$", "").replace("€", "").replace("£", "")
        s2 = s2.replace("%", "")

        # parentheses as negative: (1,234) -> -1234
        if re.match(r"^\(.*\)$", s2):
            s2 = "-" + s2[1:-1].strip()

        try:
            val = float(s2)
        except Exception:
            return None, None

        typ = 'number'
        if is_currency:
            typ = 'currency'
        elif is_percent:
            typ = 'percent'
            val = val / 100.0

        return val, typ

    s_val, s_type = parse(source)
    if s_val is None:
        return None

    # default absolute tolerances when not provided
    def default_abs(t):
        if t == 'currency':
            return 1e-2
        if t == 'percent':
            return 1e-3
        return 1e-6

    for t in targets:
        t_val, t_type = parse(t)
        if t_val is None:
            continue

        # Decide comparison type: if either is currency -> currency, elif either percent -> percent
        comp_type = 'number'
        if 'currency' in (s_type, t_type):
            comp_type = 'currency'
        elif 'percent' in (s_type, t_type):
            comp_type = 'percent'

        tol_abs = default_abs(comp_type) if abs_tol is None else abs_tol
        tol_rel = rel_tol

        err = abs(s_val - t_val)
        if err <= max(tol_abs, tol_rel * max(abs(s_val), abs(t_val))):
            return t
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
    true_set = {"true", "1", "yes", "y", "t"}
    false_set = {"false", "0", "no", "n", "f"}
    src_val = str(source).strip().lower()
    if src_val in true_set:
        src_bool = True
    elif src_val in false_set:
        src_bool = False
    else:
        return None
    for t in targets:
        tgt_val = str(t).strip().lower()
        if tgt_val in true_set and src_bool is True:
            return t
        elif tgt_val in false_set and src_bool is False:
            return t
    return None

def phone_algorithm(source, targets):
    def normalize_phone(p):
        digits = re.sub(r"\D", "", str(p))
        return digits[-10:] if len(digits) >= 10 else None
    src_phone = normalize_phone(source)
    if src_phone is None:
        return None
    for t in targets:
        tgt_phone = normalize_phone(t)
        if tgt_phone == src_phone:
            return t
    return None

def email_algorithm(source, targets):
    def norm(e):
        e = str(e).strip().lower()
        if "@" not in e: return None
        local, dom = e.split("@", 1)
        if dom in ("gmail.com", "googlemail.com"):
            local = local.split("+", 1)[0].replace(".", "")
        return f"{local}@{dom}"
    s = norm(source)
    if not s: return None
    for t in targets:
        if norm(t) == s:
            return t
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

def zip_algorithm(source, targets):
    def norm(z):
        d = re.sub(r"\D", "", str(z))
        return d[:5] if len(d)>=5 else None
    s = norm(source)
    if not s: return None
    for t in targets:
        if norm(t) == s:
            return t
    return None


def credit_card_algorithm(source, targets):
    def norm_cc(c):
        digits = re.sub(r"\D", "", str(c))
        return digits if 12 <= len(digits) <= 19 and _luhn_ok(digits) else None
    def _luhn_ok(d):
        s = 0; alt = False
        for ch in d[::-1]:
            n = int(ch)
            if alt:
                n = n*2; n -= 9 if n > 9 else 0
            s += n; alt = not alt
        return s % 10 == 0
    s = norm_cc(source)
    if not s: return None
    for t in targets:
        if norm_cc(t) == s:
            return t
    return None


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
    def norm(i):
        s = re.sub(r"[\s-]", "", str(i)).upper()
        if _isbn13_ok(s): return _to_isbn13(s)  # canonicalize to 13
        if _isbn10_ok(s): return _to_isbn13(s)
        return None
    def _isbn10_ok(s):
        if len(s)!=10: return False
        total=0
        for i,ch in enumerate(s[:9],1):
            if not ch.isdigit(): return False
            total += i*int(ch)
        checksum = s[9]
        total += 10 if checksum=="X" else (int(checksum) if checksum.isdigit() else 0)
        return total % 11 == 0
    def _isbn13_ok(s):
        if len(s)!=13 or not s.isdigit(): return False
        total = sum((int(d)*(1 if i%2==0 else 3)) for i,d in enumerate(s[:12]))
        return (10 - total%10) % 10 == int(s[12])
    def _to_isbn13(s):
        if len(s)==13: return s
        # convert 10->13
        core = "978"+s[:-1]
        total = sum((int(d)*(1 if i%2==0 else 3)) for i,d in enumerate(core))
        chk = (10 - total%10) % 10
        return core + str(chk)
    s = norm(source)
    if not s: return None
    for t in targets:
        if norm(t) == s:
            return t
    return None


def mac_address_algorithm(source, targets):
    def norm(m):
        m = re.sub(r"[^0-9a-fA-F]", "", str(m))
        if len(m) != 12: return None
        m = m.lower()
        return ":".join(m[i:i+2] for i in range(0, 12, 2))
    s = norm(source)
    if not s: return None
    for t in targets:
        if norm(t) == s:
            return t
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


def geo_pair_algorithm(source, targets, tol_m=50.0):
    """
    source: "lat,lon" or (lat,lon); targets list of same
    """
    def parse(x):
        if isinstance(x, (tuple, list)) and len(x)==2: return float(x[0]), float(x[1])
        parts = str(x).split(",")
        if len(parts)!=2: return None
        try: return float(parts[0]), float(parts[1])
        except: return None
    def haversine(p,q):
        R=6371000.0
        lat1,lon1 = map(math.radians, p)
        lat2,lon2 = map(math.radians, q)
        dlat=lat2-lat1; dlon=lon2-lon1
        a=math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        return 2*R*math.asin(math.sqrt(a))
    s = parse(source)
    if not s: return None
    best=None; best_d=1e18
    for t in targets:
        q = parse(t)
        if not q: continue
        d = haversine(s,q)
        if d <= tol_m and d < best_d:
            best, best_d = t, d
    return best

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


def unit_normalizer_algorithm(source, targets, abs_tol=0.1, rel_tol=1e-2):
    """Unified unit-aware matcher.

    Detects numeric values and units in `source` and `targets`. Supports
    length (in, cm), weight (kg, lb) and temperature (C, F) conversions.
    Returns the first target that, after normalization/conversion, matches
    the source within the provided tolerances. If no unit information is
    found or no match within tolerance, returns None.
    """
    def parse_value_unit(x):
        if x is None:
            return None, None
        s = str(x).strip()
        # find first numeric substring
        m = re.search(r"[+-]?\d+[\d,\.]*?(?:[eE][+-]?\d+)?", s)
        if not m:
            return None, None
        num_str = m.group(0)
        # normalize number (remove commas)
        num_norm = num_str.replace(",", "")
        try:
            val = float(num_norm)
        except Exception:
            return None, None

        # unit is the remainder after the number
        unit_part = s[m.end():].strip().lower()
        # also check prefix unit like 'kg 5' (rare)
        if not unit_part:
            prefix = s[:m.start()].strip().lower()
            if prefix:
                unit_part = prefix

        # clean unit string
        unit_part = re.sub(r'[^a-zA-Z°"\'º%]+', '', unit_part)
        return val, unit_part or None

    def unit_type_and_canonical(u):
        if not u:
            return None, None
        u = u.lower().replace('.', '').replace('\u00ba', 'o')
        # length
        if any(k in u for k in ["cm", "cent", "centi"]):
            return 'length', 'cm'
        if any(k in u for k in ["in", 'inch', '″', '"', "inch"]):
            return 'length', 'in'
        # weight
        if any(k in u for k in ["kg", "kilog", "kilogram"]):
            return 'weight', 'kg'
        if any(k in u for k in ["lb", "lbs", "pound", "pounder"]):
            return 'weight', 'lb'
        # temperature
        if 'c' == u or 'c' in u or '°c' in u or 'degc' in u:
            return 'temp', 'C'
        if 'f' == u or 'f' in u or '°f' in u or 'degf' in u:
            return 'temp', 'F'
        return None, None

    def convert_to(val, from_unit, to_unit, utype):
        if val is None or from_unit is None or to_unit is None:
            return None
        if utype == 'length':
            if from_unit == to_unit:
                return val
            if from_unit == 'in' and to_unit == 'cm':
                return val * 2.54
            if from_unit == 'cm' and to_unit == 'in':
                return val / 2.54
        if utype == 'weight':
            if from_unit == to_unit:
                return val
            if from_unit == 'kg' and to_unit == 'lb':
                return val * 2.20462
            if from_unit == 'lb' and to_unit == 'kg':
                return val / 2.20462
        if utype == 'temp':
            if from_unit == to_unit:
                return val
            if from_unit == 'C' and to_unit == 'F':
                return val * 9.0 / 5.0 + 32.0
            if from_unit == 'F' and to_unit == 'C':
                return (val - 32.0) * 5.0 / 9.0
        return None

    s_val, s_unit_raw = parse_value_unit(source)
    s_type, s_unit = unit_type_and_canonical(s_unit_raw)

    for t in targets:
        t_val, t_unit_raw = parse_value_unit(t)
        t_type, t_unit = unit_type_and_canonical(t_unit_raw)

        # If both have recognized unit types and they match, try direct compare
        if s_val is not None and t_val is not None and s_type and t_type and s_type == t_type:
            utype = s_type
            # choose sensible tolerances
            tol_abs = abs_tol if utype in ('length', 'weight') else 0.5
            tol_rel = rel_tol

            # convert source to target unit and compare
            if s_unit and t_unit:
                s_in_t = convert_to(s_val, s_unit, t_unit, utype)
                if s_in_t is not None:
                    err = abs(s_in_t - t_val)
                    if err <= max(tol_abs, tol_rel * max(abs(s_in_t), abs(t_val))):
                        return t
                # convert target to source unit and compare
                t_in_s = convert_to(t_val, t_unit, s_unit, utype)
                if t_in_s is not None:
                    err = abs(t_in_s - s_val)
                    if err <= max(tol_abs, tol_rel * max(abs(t_in_s), abs(s_val))):
                        return t
            else:
                # If units missing but same type, compare numeric values scaled if needed
                err = abs(s_val - t_val)
                if err <= max(tol_abs, tol_rel * max(abs(s_val), abs(t_val))):
                    return t

    return None


# The old specific unit conversion functions were removed in favor of
# `unit_normalizer_algorithm`. Call that single algorithm where needed.

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





