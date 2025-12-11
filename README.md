# matchmaker

A semantic join operator which attempts to maximize the number of valid value matches captured between the values sets of two columns

## Project Overview

Matchmaker is a reinforcement learning-based semantic join operator designed to maximize the number of valid value matches between two column value sets. This project uses Reinforcement Learning (RL) to train an intelligent agent that can dynamically select and combine multiple matching algorithms based on different input features, achieving an optimal balance between accuracy and cost.

## Methodology

This project employs Reinforcement Learning (PPO algorithm) to train an agent that can dynamically select from the following matching algorithms:

- **lexical**: Lexical-based matching algorithm (lightweight)
- **semantic**: Semantic similarity-based matching algorithm (moderate cost)
- **llm**: Large Language Model-based reasoning matching algorithm (expensive)
- **shingles**: Character n-gram-based matching algorithm (lightweight)
- **regex**: Regular expression-based matching algorithm (moderate cost)
- **identity**: Exact matching algorithm (lightweight)
- **accent_fold**: Accent folding-based matching algorithm (lightweight)

The agent learns to select the most appropriate combination of matching algorithms within a given cost budget by observing input value features (such as edit distance, semantic similarity, etc.), maximizing matching accuracy.

## How to Run

### Requirements

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

### Training and Evaluation

The project supports four datasets: `autofj`, `ss`, `wt`, and `kbwt`.

To run:

```bash
python main.py <dataset_name>
```

Where `<dataset_name>` can be one of:
- `autofj`
- `ss`
- `wt`
- `kbwt`

Example:

```bash
# Train and evaluate on the autofj dataset
python main.py autofj

# Train and evaluate on the ss dataset
python main.py ss
```

The program will:
1. Automatically read and format the specified dataset
2. Train the reinforcement learning agent (if model checkpoint doesn't exist)
3. Evaluate the agent's performance on the test set
4. Output metrics including accuracy, precision, recall, and F1 score

Trained models are saved in the `model_checkpoints/<dataset_name>/` directory. Evaluation results and metrics are saved in `metrics_output.txt`.

## Baseline Results

We provide experimental results from two baseline methods, located at:

### 1. DTT Baseline

Results file: `baseline/dtt_result/dtt_result.txt`

This file contains performance metrics of the DTT (Deep Table Transformation) method on four datasets, including:
- Accuracy
- Precision
- Recall
- F1 Score
- Average edit distance
- Runtime

### 2. AutoFuzzy Baseline

Results file: `baseline/autofuzzy_result/autoFuzzy_result.txt`

This file contains performance metrics of the AutoFuzzy method on four datasets, including:
- Accuracy
- Precision
- Recall
- F1 Score
- Number of datasets and result sizes

Both baseline results cover the four datasets (`autofj`, `ss`, `wt`, and `kbwt`) and provide overall average performance metrics, which can be used for comparison with our reinforcement learning approach.
