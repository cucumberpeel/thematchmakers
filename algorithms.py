import re
import ipaddress
import math
from urllib.parse import urlparse, urlunparse
import unicodedata, re


import pandas as pd
try:
    import bdikit as bdi
except Exception:
    bdi = None


def lexical_algorithm(source, targets):
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
    source_column = 'source'
    target_column = 'target'
    source_dataset = pd.DataFrame({source_column: [source] })
    target_dataset = pd.DataFrame({ target_column: targets })
    print("WARNING: Use LLM reasoning only in production")
    matches = bdi.match_values(
                            source_dataset,
                            target_dataset,
                            attribute_matches=(source_column, target_column),
                            method="llm",
                            method_args={"model_name": "deepinfra/openai/gpt-oss-120b"}
                        )

    return matches["target_value"].iloc[0]


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
    # escape source but allow flexible whitespace
    src = " ".join(str(source).strip().split())
    pat = re.escape(src).replace(r"\ ", r"\s+")
    try:
        rx = re.compile(pat, flags)
    except re.error:
        return None
    for t in targets:
        if t is None: 
            continue
        if rx.search(str(t)):
            return t
    return None

def ip_country_matcher(ip, targets, cidr_map=None, **kw):
    # targets is usually a list of country labels; use cidr_map to decide which one
    try:
        ip_addr = ipaddress.ip_address(str(ip))
    except Exception:
        return {"prediction": None, "score": 0.0, "meta": {"error": "bad ip"}}

    best_country, best_prefix = None, -1
    if not cidr_map:
        return {"prediction": None, "score": 0.0, "meta": {"reason": "no cidr_map"}}

    for country, cidrs in cidr_map.items():
        for c in cidrs:
            try:
                net = ipaddress.ip_network(c, strict=False)
            except Exception:
                continue
            if ip_addr in net and net.prefixlen > best_prefix:
                best_prefix, best_country = net.prefixlen, country

    score = 1.0 if best_country in set(map(str, targets)) else 0.0
    pred = best_country if score > 0 else None
    return {"prediction": pred, "score": float(score), "meta": {"prefixlen": best_prefix}}

def ip_subnet_matcher(source, targets, prefix=24, **kw):
    # here targets are other IPs; match if in same subnet
    try:
        net_src = ipaddress.ip_network(f"{source}/{prefix}", strict=False)
    except Exception:
        return {"prediction": None, "score": 0.0, "meta": {"error": "bad ip"}}
    for t in targets:
        try:
            if ipaddress.ip_network(f"{t}/{prefix}", strict=False) == net_src:
                return {"prediction": t, "score": 1.0, "meta": {"prefix": prefix}}
        except Exception:
            continue
    return {"prediction": None, "score": 0.0, "meta": {"prefix": prefix}}

def date_algorithm(source, targets, date_only=True, tolerance=None):
    """
    - date_only=True → compare calendar dates (ignore time-of-day)
    - tolerance=None → exact equality; else pd.Timedelta like '1D' or '2H'
    """
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

def numeric_algorithm(source, targets, abs_tol=1e-6, rel_tol=1e-3):
    def to_float(x):
        try:
            return float(str(x).replace(",", ""))
        except:
            return None
    s = to_float(source)
    if s is None:
        return None
    best, best_err = None, float("inf")
    for t in targets:
        v = to_float(t)
        if v is None:
            continue
        err = abs(s - v)
        if err <= max(abs_tol, rel_tol * max(abs(s), abs(v))) and err < best_err:
            best, best_err = t, err
    return best


def identity_algorithm(source, targets):
    if source in targets:
        return source
    return None

def length_algorithm(source, targets):
    try:
        src_len = len(str(source))
    except Exception:
        return None
    for t in targets:
        try:
            if len(str(t)) == src_len:
                return t
        except Exception:
            continue
    return None

def prefix_algorithm(source, targets, prefix_len=3):
    src_prefix = str(source)[:prefix_len]
    for t in targets:
        if str(t).startswith(src_prefix):
            return t
    return None

def suffix_algorithm(source, targets, suffix_len=3):
    src_suffix = str(source)[-suffix_len:]
    for t in targets:
        if str(t).endswith(src_suffix):
            return t
    return None


def range_algorithm(source, targets, range_tol=5):
    try:
        src_val = float(str(source).replace(",", ""))
    except Exception:
        return None
    for t in targets:
        try:
            tgt_val = float(str(t).replace(",", ""))
            if abs(src_val - tgt_val) <= range_tol:
                return t
        except Exception:
            continue
    return None

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

def address_algorithm(source, targets):
    def normalize_address(a):
        a = str(a).strip().lower()
        a = re.sub(r"\s+", " ", a)
        a = re.sub(r"[^\w\s]", "", a)
        return a
    src_addr = normalize_address(source)
    for t in targets:
        tgt_addr = normalize_address(t)
        if tgt_addr == src_addr:
            return t
    return None

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

def rgb_algorithm(source, targets):
    def normalize_rgb(r):
        r = str(r).strip().lower()
        r = re.sub(r"\s+", "", r)
        return r
    src_rgb = normalize_rgb(source)
    for t in targets:
        tgt_rgb = normalize_rgb(t)
        if tgt_rgb == src_rgb:
            return t
    return None

def hex_color_algorithm(source, targets):
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


def currency_algorithm(source, targets, abs_tol=1e-2, rel_tol=1e-3):
    def to_float(x):
        try:
            s = str(x).replace(",", "").replace("$", "").replace("€", "").replace("£", "")
            return float(s)
        except:
            return None
    s = to_float(source)
    if s is None:
        return None
    best, best_err = None, float("inf")
    for t in targets:
        v = to_float(t)
        if v is None:
            continue
        err = abs(s - v)
        if err <= max(abs_tol, rel_tol * max(abs(s), abs(v))) and err < best_err:
            best, best_err = t, err
    return best

def percentage_algorithm(source, targets, abs_tol=1e-2, rel_tol=1e-3):
    def to_float(x):
        try:
            s = str(x).replace("%", "").replace(",", "")
            return float(s) / 100.0
        except:
            return None
    s = to_float(source)
    if s is None:
        return None
    best, best_err = None, float("inf")
    for t in targets:
        v = to_float(t)
        if v is None:
            continue
        err = abs(s - v)
        if err <= max(abs_tol, rel_tol * max(abs(s), abs(v))) and err < best_err:
            best, best_err = t, err
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

def inches_to_cm_algorithm(source, targets, abs_tol=0.1, rel_tol=1e-2):
    def to_cm(x):
        try:
            val = float(str(x).replace(",", ""))
            return val * 2.54
        except:
            return None
    s = to_cm(source)
    if s is None:
        return None
    best, best_err = None, float("inf")
    for t in targets:
        v = to_cm(t)
        if v is None:
            continue
        err = abs(s - v)
        if err <= max(abs_tol, rel_tol * max(abs(s), abs(v))) and err < best_err:
            best, best_err = t, err
    return best

def cm_to_inches_algorithm(source, targets, abs_tol=0.1, rel_tol=1e-2):
    def to_inches(x):
        try:
            val = float(str(x).replace(",", ""))
            return val / 2.54
        except:
            return None
    s = to_inches(source)
    if s is None:
        return None
    best, best_err = None, float("inf")
    for t in targets:
        v = to_inches(t)
        if v is None:
            continue
        err = abs(s - v)
        if err <= max(abs_tol, rel_tol * max(abs(s), abs(v))) and err < best_err:
            best, best_err = t, err
    return best

def weight_kg_to_lb_algorithm(source, targets, abs_tol=0.1, rel_tol=1e-2):
    def to_lb(x):
        try:
            val = float(str(x).replace(",", ""))
            return val * 2.20462
        except:
            return None
    s = to_lb(source)
    if s is None:
        return None
    best, best_err = None, float("inf")
    for t in targets:
        v = to_lb(t)
        if v is None:
            continue
        err = abs(s - v)
        if err <= max(abs_tol, rel_tol * max(abs(s), abs(v))) and err < best_err:
            best, best_err = t, err
    return best

def weight_lb_to_kg_algorithm(source, targets, abs_tol=0.1, rel_tol=1e-2):
    def to_kg(x):
        try:
            val = float(str(x).replace(",", ""))
            return val / 2.20462
        except:
            return None
    s = to_kg(source)
    if s is None:
        return None
    best, best_err = None, float("inf")
    for t in targets:
        v = to_kg(t)
        if v is None:
            continue
        err = abs(s - v)
        if err <= max(abs_tol, rel_tol * max(abs(s), abs(v))) and err < best_err:
            best, best_err = t, err
    return best

def temperature_c_to_f_algorithm(source, targets, abs_tol=0.5, rel_tol=1e-2):
    def to_f(x):
        try:
            val = float(str(x).replace(",", ""))
            return val * 9.0 / 5.0 + 32.0
        except:
            return None
    s = to_f(source)
    if s is None:
        return None
    best, best_err = None, float("inf")
    for t in targets:
        v = to_f(t)
        if v is None:
            continue
        err = abs(s - v)
        if err <= max(abs_tol, rel_tol * max(abs(s), abs(v))) and err < best_err:
            best, best_err = t, err
    return best

def temperature_f_to_c_algorithm(source, targets, abs_tol=0.5, rel_tol=1e-2):
    def to_c(x):
        try:
            val = float(str(x).replace(",", ""))
            return (val - 32.0) * 5.0 / 9.0
        except:
            return None
    s = to_c(source)
    if s is None:
        return None
    best, best_err = None, float("inf")
    for t in targets:
        v = to_c(t)
        if v is None:
            continue
        err = abs(s - v)
        if err <= max(abs_tol, rel_tol * max(abs(s), abs(v))) and err < best_err:
            best, best_err = t, err
    return best

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





