import os
import pandas as pd
from sklearn.model_selection import train_test_split

def read_split_datasets(autofj=True, ss=True, wt=True, kbwt=True, test_size=0.2, random_state=42):
    train_datasets = []
    test_datasets = []

    if autofj:
        # autofj datasets: 1 source + 1 target column per file
        # source = title_l, target = title_r
        # expected: 50 datasets
        autofj_datasets_path = os.path.join("data", "autofj")
        autofj_count = 0
        for root, dir, files in os.walk(autofj_datasets_path):
            gt_files = [f for f in files if f == 'gt.csv']
            if len(gt_files) > 1:
                print(f"Error: more than 1 ground truth file in {root}: {gt_files}")
                return [], []
            if gt_files:
                autofj_count += 1
                dataset_path = os.path.join(root, gt_files[0])
                try:
                    raw_dataset = pd.read_csv(dataset_path)
                except Exception as e:
                    print(f"Error reading {dataset_path}: {e}")
                    return [], []
                source_col = 'title_l'
                target_col = 'title_r'
                all_targets = raw_dataset[target_col].unique().tolist()

                dataset_rows = []
                for _, row in raw_dataset.iterrows():
                    dataset_rows.append({
                        'source_column': source_col,
                        'target_column': target_col,
                        'source_value': str(row[source_col]),
                        'gold_value': str(row[target_col]),
                        'target_values': [str(t) for t in all_targets],
                    })
                
                if len(dataset_rows) > 0:
                    train_rows, test_rows = train_test_split(
                        dataset_rows, 
                        test_size=test_size, 
                        random_state=random_state
                    )
                    train_datasets.extend(train_rows)
                    test_datasets.extend(test_rows)
        print(f"Read and split {autofj_count} datasets from {autofj_datasets_path}")
    
    if ss:
        # spreadsheets datasets: 1 source + 1 target column per file
        # source = source-value, target = target-value
        # expected: 108 datasets
        ss_datasets_path = os.path.join("data", "ss")
        ss_count = 0
        for root, dir, files in os.walk(ss_datasets_path):
            gt_files = [f for f in files if f == 'ground truth.csv']
            if len(gt_files) > 1:
                print(f"Error: more than 1 ground truth file in {root}: {gt_files}")
                return [], []
            if gt_files:
                ss_count += 1
                dataset_path = os.path.join(root, gt_files[0])
                try:
                    raw_dataset = pd.read_csv(dataset_path)
                except Exception as e:
                    print(f"Error reading {dataset_path}: {e}")
                    return [], []
                source_col = 'source-value'
                target_col = 'target-value'
                all_targets = raw_dataset[target_col].unique().tolist()

                dataset_rows = []
                for _, row in raw_dataset.iterrows():
                    dataset_rows.append({
                        'source_column': source_col,
                        'target_column': target_col,
                        'source_value': str(row[source_col]),
                        'gold_value': str(row[target_col]),
                        'target_values': [str(t) for t in all_targets],
                    })
                
                if len(dataset_rows) > 0:
                    train_rows, test_rows = train_test_split(
                        dataset_rows, 
                        test_size=test_size, 
                        random_state=random_state
                    )
                    train_datasets.extend(train_rows)
                    test_datasets.extend(test_rows)
        print(f"Read and split {ss_count} datasets from {ss_datasets_path}")

    if wt:
        # wt datasets: multiple columns per file, check rows.txt to find source and target
        # expected: 32 datasets
        wt_datasets_path = os.path.join("data", "wt")
        wt_count = 0
        for root, dir, files in os.walk(wt_datasets_path):
            gt_files = [f for f in files if f == 'ground truth.csv']
            if len(gt_files) > 1:
                print(f"Error: more than 1 ground truth file in {root}: {gt_files}")
                return [], []
            if gt_files:
                # skip duplicate dataset
                if (root.endswith("original")): continue

                wt_count += 1
                dataset_path = os.path.join(root, gt_files[0])
                try:
                    raw_dataset = pd.read_csv(dataset_path)
                except Exception as e:
                    print(f"Error reading {dataset_path}: {e}")
                    return [], []

                try:
                    with open(os.path.join(root, 'rows.txt'), 'r') as file:
                        all_cols = raw_dataset.columns
                        matching_cols = file.readline().strip()
                        source_postfix, target_postfix = matching_cols.split(":")
                        source_col = "source-" + source_postfix
                        target_col = "target-" + target_postfix
                        if source_col not in all_cols:
                            print(f"Error in {dataset_path}: missing {source_col} from columns ({all_cols})")
                            return [], []
                        if target_col not in all_cols:
                            print(f"Error in {dataset_path}: missing {target_col} from columns ({all_cols})")
                            return [], []
                except Exception as e:
                    print(f"Error parsing columns from {dataset_path}: {e}")
                    return [], []

                all_targets = raw_dataset[target_col].unique().tolist()

                dataset_rows = []
                for _, row in raw_dataset.iterrows():
                    dataset_rows.append({
                        'source_column': source_col,
                        'target_column': target_col,
                        'source_value': str(row[source_col]),
                        'gold_value': str(row[target_col]),
                        'target_values': [str(t) for t in all_targets],
                    })
                
                if len(dataset_rows) > 0:
                    train_rows, test_rows = train_test_split(
                        dataset_rows, 
                        test_size=test_size, 
                        random_state=random_state
                    )
                    train_datasets.extend(train_rows)
                    test_datasets.extend(test_rows)
        print(f"Read and split {wt_count} datasets from {wt_datasets_path}")
    
    if kbwt:
        # kbwt datasets: 1 source + 1 target column per file
        # source column starts with 'source-', target column starts with 'target-'
        # e.g. source-Frogs, target-Frog
        # expected: 81 datasets
        kbwt_datasets_path = os.path.join("data", "kbwt")
        kbwt_count = 0
        for root, dir, files in os.walk(kbwt_datasets_path):
            gt_files = [f for f in files if f == 'ground truth.csv']
            if len(gt_files) > 1:
                print(f"Error: more than 1 ground truth file in {root}: {gt_files}")
                return [], []
            if gt_files:
                kbwt_count += 1
                dataset_path = os.path.join(root, gt_files[0])
                try:
                    raw_dataset = pd.read_csv(dataset_path)
                except Exception as e:
                    print(f"Error reading {dataset_path}: {e}")
                    return [], []

                source_cols = [col for col in raw_dataset.columns if col.startswith("source")]
                target_cols = [col for col in raw_dataset.columns if col.startswith("target")]
                if len(source_cols) > 1 or len(target_cols) > 1:
                    print(f"Error locating source-target columns: {raw_dataset.columns}")
                    return [], []
                source_col = source_cols[0]
                target_col = target_cols[0]
                all_targets = raw_dataset[target_col].unique().tolist()

                dataset_rows = []
                for _, row in raw_dataset.iterrows():
                    dataset_rows.append({
                        'source_column': source_col,
                        'target_column': target_col,
                        'source_value': str(row[source_col]),
                        'gold_value': str(row[target_col]),
                        'target_values': [str(t) for t in all_targets],
                    })
                
                if len(dataset_rows) > 0:
                    train_rows, test_rows = train_test_split(
                        dataset_rows, 
                        test_size=test_size, 
                        random_state=random_state
                    )
                    train_datasets.extend(train_rows)
                    test_datasets.extend(test_rows)
        print(f"Read and split {kbwt_count} datasets from {kbwt_datasets_path}")

    print(f"Training data size: {len(train_datasets)}")
    print(f"Test data size: {len(test_datasets)}")
    return train_datasets, test_datasets

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