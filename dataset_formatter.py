import pandas as pd
from sklearn.model_selection import train_test_split


def read_datasets_old():
    formatted_datasets = []
    dataset_paths = ['../data/Hospital/gt.csv', '../data/Country/gt.csv']

    for dataset_path in dataset_paths:
        raw_dataset = pd.read_csv(dataset_path)
        source_col = 'title_l'
        target_col = 'title_r'
        all_targets = raw_dataset['title_r'].unique().tolist()
        pairs = []
        for _, row in raw_dataset.iterrows():
            pairs.append({
                'source_value': row[source_col],
                'gold_value': row[target_col],
                'target_values': all_targets,
            })
        formatted_datasets.append({'source_column': source_col,
                                   'target_column': target_col,
                                   'pairs': pairs})

    return formatted_datasets

def read_datasets():
    formatted_datasets = []
    dataset_paths = ['../data/Hospital/gt.csv', '../data/Country/gt.csv']

    for dataset_path in dataset_paths:
        raw_dataset = pd.read_csv(dataset_path)
        source_col = 'title_l'
        target_col = 'title_r'
        all_targets = raw_dataset['title_r'].unique().tolist()

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
    for dataset in datasets:
        print(f"Source Column: {dataset['source_column']}")
        print(f"Target Column: {dataset['target_column']}")
        print(f"Number of Pairs: {len(dataset['pairs'])}")
        for pair in dataset['pairs']:
            print(f"Source Value: {pair['source_value']}, Gold: {pair['gold_value']}, Targets: {pair['target_values'][:3]}...")  # Print first 3 targets for brevity