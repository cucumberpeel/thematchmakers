import pandas as pd
from typing import List, Dict, Optional

def load_data(
    path: str,
    source: str = 'title_l',
    target: str = 'title_r',
    gold: Optional[str] = None
) -> List[Dict]:
    try:
        raw_data = pd.read_csv(path)
    except Exception as e:
        print(f"Error loading data from {path}: {e}")
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