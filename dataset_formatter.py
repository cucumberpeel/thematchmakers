import json
import os
import pandas as pd
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

def read_datasets():
    formatted_datasets = []
    with open('datasets.json', 'r') as file:
        dataset_names = json.load(file)

    for name in dataset_names["autofj"]:
        dataset_path = os.path.join("data", "benchmark", name, "gt.csv")
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
    
    for name in dataset_names["dtt-dxf"]:
        dataset_path = os.path.join("data", "DXF", name, "ground truth.csv")
        print(f"Reading {dataset_path}...")
        raw_dataset = pd.read_csv(dataset_path)

        source_cols = [col for col in raw_dataset.columns if col.startswith("source")]
        target_cols = [col for col in raw_dataset.columns if col.startswith("target")]
        if len(source_cols) > 1 or len(target_cols) > 1:
            print(f"Error locating source-target columns: {raw_dataset.columns}")
            return []
        source_col = source_cols[0]
        target_col = target_cols[0]

        all_targets = raw_dataset[target_col].unique().tolist()

        for _, row in raw_dataset.iterrows():
            formatted_datasets.append({
                'source_column': source_col,
                'target_column': target_col,
                'source_value': row[source_col],
                'gold_value': row[target_col],
                'target_values': all_targets,
            })

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