from agent_environment import ValueMatchingEnv
import numpy as np


def compute_prf(golds, preds):
    """Compute precision/recall/F1 treating None as an abstention."""
    assert len(golds) == len(preds), "golds and preds must align"
    if not golds:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    tp = fp = fn = tn = 0

    for g, p in zip(golds, preds):
        if p is None:
            if g is None:
                tn += 1
            else:
                fn += 1
        else:
            if p == g:
                tp += 1
            else:
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


# Helper function to convert numeric state vector to tuple (hashable for Q-table)
def state_to_key(state):
        return tuple(state.round(3))  # round to reduce floating-point noise


def train_agent(primitives, primitive_names, dataset, feature_dim, max_steps):
    env = ValueMatchingEnv(primitives, feature_dim, max_steps)
    num_actions = env.action_space.n
    
    # Initialize Q-table
    Q = {}
    
    # Set hyperparameters
    alpha = 0.1    # learning rate
    gamma = 0.9    # discount factor
    epsilon = 0.5  # exploration rate
    epsilon_decay = 0.995  # decay epsilon over time
    min_epsilon = 0.01
    
    # Training loop
    num_episodes = 1000
    first_attempt_success = 0
    attempt_counts = []
    rewards_history = [] 
    
    for episode in range(num_episodes):
        # Sample a random value from dataset
        sample = dataset[episode % len(dataset)]  # Cycle through dataset

        # Reset environment with this sample
        state = env.reset(
            source=sample['source_value'],
            targets=sample['target_values'],
            gold=sample['gold_value']
        )
        
        done = False
        total_reward = 0
        
        while not done:
            state_key = state_to_key(state)
            
            valid_actions = env.get_valid_actions()

            if not valid_actions:  # No valid actions left
                break  # End episode early

            if np.random.rand() < epsilon:
                action = np.random.choice(valid_actions)  # Explore valid actions only
            else:
                # Exploit: choose best valid action from Q-table
                q_values = {a: Q.get((state_key, a), 0) for a in valid_actions}
                action = max(q_values, key=q_values.get)
            
            # Take action 
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Q-learning update
            next_state_key = state_to_key(next_state)
            
            if not done:
                # Get max Q-value for next state
                next_q_values = {a: Q.get((next_state_key, a), 0) for a in range(num_actions)}
                max_next_q_value = max(next_q_values.values()) if next_q_values else 0
            else:
                # Terminal state has no future value
                max_next_q_value = 0
            
            # Update Q-table
            current_q_value = Q.get((state_key, action), 0)
            updated_q_value = current_q_value + alpha * (reward + gamma * max_next_q_value - current_q_value)
            Q[(state_key, action)] = updated_q_value
            
            state = next_state

        rewards_history.append(total_reward)
        attempt_counts.append(info['attempts'])

        if info['correct'] and info['attempts'] == 1:
            first_attempt_success += 1
        
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        # Logging
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])  # Average over last 100
            avg_attempts = np.mean(attempt_counts[-100:])  # Average over last 100
            first_attempt_rate = first_attempt_success / 100
            
            print(f"Episode {episode + 1}/{num_episodes}, "
                f"Avg Reward: {avg_reward:.3f}, "
                f"Avg Attempts: {avg_attempts:.3f}, "
                #f"First-Attempt Success: {first_attempt_rate:.3f}, "
                f"Epsilon: {epsilon:.3f}"
            )
        first_attempt_success = 0  # Reset for next logging period
    
    return Q


def evaluate_agent(Q, primitives, primitive_names, test_dataset, feature_dim, max_steps):
    env = ValueMatchingEnv(primitives, feature_dim, max_steps)
    
    results = {
        'correct': 0,
        'total': 0,
        'total_attempts': 0,
        'algorithm_usage': {i: 0 for i in primitive_names}
    }
    golds = []
    preds = []
    
    for sample in test_dataset:
        state = env.reset(
            source=sample['source_value'],
            targets=sample['target_values'],
            gold=sample['gold_value']
        )
        
        done = False
        
        while not done:
            state_key = state_to_key(state)

            # Get valid actions
            valid_actions = env.get_valid_actions()
            
            if not valid_actions:  # No valid actions left
                break
            
            # Greedy action selection (no exploration)
            q_values = {a: Q.get((state_key, a), 0) for a in valid_actions}
            action = max(q_values, key=q_values.get)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            state = next_state
        
        # Record results
        results['total'] += 1
        results['total_attempts'] += info['attempts']
        if info['correct']:
            results['correct'] += 1
        golds.append(sample['gold_value'])
        preds.append(info['predicted'])

        for used_algorithm in info['history']:
            if used_algorithm != -1:
                results['algorithm_usage'][primitive_names[used_algorithm]] += 1

    # Print evaluation results
    prf = compute_prf(golds, preds)

    print(f"Accuracy: {results['correct'] / results['total']:.3f}")
    print(f"Precision: {prf['precision']:.3f}")
    print(f"Recall: {prf['recall']:.3f}")
    print(f"F1: {prf['f1']:.3f}")
    print(f"Average Attempts: {results['total_attempts'] / results['total']:.3f}")
    print(f"Algorithm Usage: {results['algorithm_usage']}")
    
    results.update(prf)
    return results


