import os
import pandas as pd
from typing import List, Dict, Optional

def load_benchmark_data(
    name: str,
    source: str = 'title_l',
    target: str = 'title_r',
    gold: Optional[str] = None
) -> List[Dict]:
    """
    Loads AutoFJ benchmark data from dataset name.
    Returns list of dicts with source value, gold value, and list of target values.
    """
    try:
        raw_data = pd.read_csv(os.path.join("data", name, "gt.csv"))
    except Exception as e:
        print(f"Error loading data from {name}: {e}")
        return []
    
    if gold is None:
        gold = target
    
    all_targets = raw_data[target].unique().tolist()
    
    dataset = []
    for _, row in raw_data.iterrows():
        dataset.append({
            'source': row[source],
            'gold': row[gold],
            'targets': all_targets,
        })
    
    return dataset
