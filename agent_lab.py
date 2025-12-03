from agent_environment import ValueMatchingEnv
import numpy as np


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
    success_count = 0
    
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
            
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                # Exploit: choose best action from Q-table
                q_values = {a: Q.get((state_key, a), 0) for a in range(num_actions)}
                action = max(q_values, key=q_values.get) # Best action
            
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
        
        # Track success
        if info['correct']:
            success_count += 1
        
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        # Logging
        if (episode + 1) % 100 == 0:
            avg_success = success_count / 100
            format_history = ", ".join([primitive_names[i] for i in info['history'] if i != -1])
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Total Reward: {total_reward:.3f}, "
                  f"Success Rate: {avg_success:.3f}, "
                  f"Epsilon: {epsilon:.3f}, "
                  f"Attempts: {info['attempts']}, "
                  f"History: [{format_history}]"
            )
            success_count = 0
    
    return Q


def evaluate_agent(Q, primitives, primitive_names, test_dataset, feature_dim, max_steps):
    env = ValueMatchingEnv(primitives, feature_dim, max_steps)
    num_actions = env.action_space.n
    
    results = {
        'correct': 0,
        'total': 0,
        'total_attempts': 0,
        'algorithm_usage': {i: 0 for i in primitive_names}
    }
    
    for sample in test_dataset:
        state = env.reset(
            source=sample['source_value'],
            targets=sample['target_values'],
            gold=sample['gold_value']
        )
        
        done = False
        
        while not done:
            state_key = state_to_key(state)
            
            # Greedy action selection (no exploration)
            q_values = {a: Q.get((state_key, a), 0) for a in range(num_actions)}
            action = max(q_values, key=q_values.get)
            #print("Selected action:", primitive_names[action])
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            state = next_state
        
        # Record results
        results['total'] += 1
        results['total_attempts'] += info['attempts']
        if info['correct']:
            results['correct'] += 1
        #print("history", info['history'])
        for used_algorithm in info['history']:
            if used_algorithm != -1:
                results['algorithm_usage'][primitive_names[used_algorithm]] += 1

    # Print evaluation results
    print(f"Accuracy: {results['correct'] / results['total']:.3f}")
    print(f"Average Attempts: {results['total_attempts'] / results['total']:.3f}")
    print(f"Algorithm Usage: {results['algorithm_usage']}")
    
    return results


