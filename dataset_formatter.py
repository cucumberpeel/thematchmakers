import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List
from sklearn.model_selection import train_test_split

# def read_datasets_old():
#     formatted_datasets = []
#     dataset_paths = ['../data/Hospital/gt.csv', '../data/Country/gt.csv']

#     for dataset_path in dataset_paths:
#         raw_dataset = pd.read_csv(dataset_path)
#         source_col = 'title_l'
#         target_col = 'title_r'
#         all_targets = raw_dataset['title_r'].unique().tolist()
#         pairs = []
#         for _, row in raw_dataset.iterrows():
#             pairs.append({
#                 'source_value': row[source_col],
#                 'gold_value': row[target_col],
#                 'target_values': all_targets,
#             })
#         formatted_datasets.append({'source_column': source_col,
#                                    'target_column': target_col,
#                                    'pairs': pairs})

#     return formatted_datasets

def read_all_csvs_in_data_directory(data_dir: str = "data") -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Recursively read all CSV files in the data directory structure.
    
    Args:
        data_dir: Path to the data directory (default: "data")
    
    Returns:
        Dictionary with structure: {folder_path: {csv_filename: DataFrame}}
        Example: {"data/benchmark/Hospital": {"gt.csv": DataFrame, "left.csv": DataFrame}}
    """
    all_data = {}
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: Data directory '{data_dir}' does not exist")
        return all_data
    
    # Iterate through all subdirectories
    for root, dirs, files in os.walk(data_path):
        csv_files = [f for f in files if f.endswith('.csv')]
        
        if csv_files:
            folder_path = str(Path(root).relative_to(data_path))
            all_data[folder_path] = {}
            
            for csv_file in csv_files:
                csv_path = os.path.join(root, csv_file)
                try:
                    df = pd.read_csv(csv_path)
                    all_data[folder_path][csv_file] = df
                    print(f"✓ Read {folder_path}/{csv_file} ({len(df)} rows, {len(df.columns)} columns)")
                except Exception as e:
                    print(f"✗ Error reading {folder_path}/{csv_file}: {e}")
                    all_data[folder_path][csv_file] = None
    
    return all_data

def read_all_csvs_flat(data_dir: str = "data") -> List[Dict]:
    """
    Recursively read all CSV files and return as a flat list.
    
    Args:
        data_dir: Path to the data directory (default: "data")
    
    Returns:
        List of dictionaries, each containing:
        {
            'folder': str,
            'filename': str,
            'path': str,
            'dataframe': pd.DataFrame
        }
    """
    all_csvs = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: Data directory '{data_dir}' does not exist")
        return all_csvs
    
    # Iterate through all subdirectories
    for root, dirs, files in os.walk(data_path):
        csv_files = [f for f in files if f.endswith('.csv')]
        
        for csv_file in csv_files:
            csv_path = os.path.join(root, csv_file)
            folder_path = str(Path(root).relative_to(data_path))
            
            try:
                df = pd.read_csv(csv_path)
                all_csvs.append({
                    'folder': folder_path,
                    'filename': csv_file,
                    'path': csv_path,
                    'dataframe': df
                })
                print(f"✓ Read {folder_path}/{csv_file} ({len(df)} rows, {len(df.columns)} columns)")
            except Exception as e:
                print(f"✗ Error reading {folder_path}/{csv_file}: {e}")
    
    return all_csvs

def read_datasets():
    formatted_datasets = []

    autofj_datasets_path = os.path.join("data", "benchmark")
    for root, dir, files in os.walk(autofj_datasets_path):
        gt_files = [f for f in files if f == 'gt.csv']
        if len(gt_files) > 1:
            print(f"Error: more than 1 ground truth file in this directory: {gt_files}")
            return []
        if gt_files:
            dataset_path = os.path.join(root, gt_files[0])
            print(f"Reading {dataset_path}...")
            raw_dataset = pd.read_csv(dataset_path)
            source_col = 'title_l'
            target_col = 'title_r'
            all_targets = raw_dataset[target_col].unique().tolist()

            for _, row in raw_dataset.iterrows():
                formatted_datasets.append({
                    'source_column': source_col,
                    'target_column': target_col,
                    'source_value': row[source_col],
                    'gold_value': row[target_col],
                    'target_values': all_targets,
                })
    
    # for name in dataset_names["dtt-dxf"]:
    #     dataset_path = os.path.join("data", "DXF", name, "ground truth.csv")
    #     print(f"Reading {dataset_path}...")
    #     raw_dataset = pd.read_csv(dataset_path)

    #     source_cols = [col for col in raw_dataset.columns if col.startswith("source")]
    #     target_cols = [col for col in raw_dataset.columns if col.startswith("target")]
    #     if len(source_cols) > 1 or len(target_cols) > 1:
    #         print(f"Error locating source-target columns: {raw_dataset.columns}")
    #         return []
    #     source_col = source_cols[0]
    #     target_col = target_cols[0]

    #     all_targets = raw_dataset[target_col].unique().tolist()

    #     for _, row in raw_dataset.iterrows():
    #         formatted_datasets.append({
    #             'source_column': source_col,
    #             'target_column': target_col,
    #             'source_value': row[source_col],
    #             'gold_value': row[target_col],
    #             'target_values': all_targets,
    #         })

    return formatted_datasets


def split_datasets(datasets, test_size=0.2, random_state=42):
    train_set, test_set = train_test_split(datasets, test_size=test_size, random_state=random_state)
    return train_set, test_set

def print_datasets_info(datasets):
    print("Previewing datasets...")
    for dataset in datasets:
        print(f"Source Column: {dataset['source_column']}")
        print(f"Target Column: {dataset['target_column']}")

        # for read_datasets_old
        print(f"Number of Pairs: {len(dataset['pairs'])}")
        for pair in dataset['pairs']:
            print(f"Source Value: {pair['source_value']}, Gold: {pair['gold_value']}, Targets: {pair['target_values'][:3]}...")  # Print first 3 targets for brevity