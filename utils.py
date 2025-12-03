import os
import pandas as pd
from typing import List, Dict, Optional

def load_autofj_data(
    name: str,
    source: str = 'title_l',
    target: str = 'title_r',
    gold: Optional[str] = None
) -> List[Dict]:
    """
    Loads AutoFJ benchmark dataset from dataset name.
    Returns list of dicts with source value, gold value, and list of target values.

    Args:
        name: Name of the dataset
        source: Name of the source column
        target: Name of the target column
        gold: Name of the gold column
    """
    try:
        raw_data = pd.read_csv(os.path.join("data", "benchmark", name, "gt.csv"))
    except Exception as e:
        print(f"Error loading data from {name}: {e}")
        return []
    
    if gold is None:
        gold = target
    
    all_targets = raw_data[target].unique().tolist()
    
    dataset = []
    for _, row in raw_data.iterrows():
        dataset.append({
                'source_column': source,
                'target_column': target,
                'source_value': row[source],
                'gold_value': row[target],
                'target_values': all_targets,
            })
    
    return dataset

def load_dtt_data(
    name: str,
    source: Optional[str] = None,
    target: Optional[str] = None,
    gold: Optional[str] = None
) -> List[Dict]:
    """
    Loads DTT dataset from dataset name.
    Returns list of dicts with source value, gold value, and list of target values.

    Args:
        name: Name of the dataset
        source: Name of the source column
        target: Name of the target column
        gold: Name of the gold column
    """
    try:
        raw_data = pd.read_csv(os.path.join("data", "DXF", name, "ground truth.csv"))
    except Exception as e:
        print(f"Error loading data from {name}: {e}")
        return []
    
    if source is None or target is None:
        source_cols = [col for col in raw_data.columns if col.startswith("source")]
        target_cols = [col for col in raw_data.columns if col.startswith("target")]
        if len(source_cols) > 1 or len(target_cols) > 1:
            print(f"Error locating source-target columns: {raw_data.columns}")
            return []
        source = source_cols[0]
        target = target_cols[0]
    
    if gold is None:
        gold = target
    
    all_targets = raw_data[target].unique().tolist()
    
    dataset = []
    for _, row in raw_data.iterrows():
        dataset.append({
                'source_column': source,
                'target_column': target,
                'source_value': row[source],
                'gold_value': row[target],
                'target_values': all_targets,
            })
    
    return dataset