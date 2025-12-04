import pandas as pd
from algorithms import shingles_algorithm, regex_algorithm


def evaluate(gt_path="data/Hospital/gt.csv", k=4, sample_mismatches=10):
    df = pd.read_csv(gt_path)
    # Ensure columns exist
    if "title_l" not in df.columns or "title_r" not in df.columns:
        raise ValueError("gt.csv must contain 'title_l' and 'title_r' columns")

    candidates = df["title_r"].astype(str).tolist()

    total = 0
    shingle_hits = 0
    regex_hits = 0
    regex_none = 0

    shingle_mismatches = []
    regex_mismatches = []

    for _, row in df.iterrows():
        source = str(row["title_l"]) if pd.notna(row["title_l"]) else ""
        gold = str(row["title_r"]) if pd.notna(row["title_r"]) else ""
        total += 1

        pred_shingle = shingles_algorithm(source, candidates, k=k)
        pred_regex = regex_algorithm(source, candidates)

        if pred_shingle == gold:
            shingle_hits += 1
        else:
            if len(shingle_mismatches) < sample_mismatches:
                shingle_mismatches.append((source, gold, pred_shingle))

        if pred_regex == gold:
            regex_hits += 1
        else:
            if pred_regex is None:
                regex_none += 1
            if len(regex_mismatches) < sample_mismatches:
                regex_mismatches.append((source, gold, pred_regex))

    print(f"Evaluated {total} examples")
    print(f"Shingles (k={k}) accuracy: {shingle_hits}/{total} = {shingle_hits/total:.4f}")
    print(f"Regex accuracy: {regex_hits}/{total} = {regex_hits/total:.4f}")
    print(f"Regex returned None for {regex_none} examples")

    if shingle_mismatches:
        print("\nSample shingles mismatches (source | gold | prediction):")
        for s, g, p in shingle_mismatches:
            print(f"- {s} | {g} | {p}")

    if regex_mismatches:
        print("\nSample regex mismatches (source | gold | prediction_or_None):")
        for s, g, p in regex_mismatches:
            print(f"- {s} | {g} | {p}")


if __name__ == "__main__":
    evaluate()
