import ray
import json
import logging
from agent_environment import ValueMatchingEnv
from ray.rllib.algorithms.ppo import PPOConfig

def make_config(primitives, primitive_names, primitive_costs, dataset, feature_dim, max_steps):
    config = (
        PPOConfig()
        .environment(
            env=ValueMatchingEnv,
            env_config={
                "dataset": dataset,
                "primitives": primitives,
                "primitive_names": primitive_names,
                "costs": primitive_costs,
                "feature_dim": feature_dim,
                "max_steps": max_steps
            }
        )
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        .env_runners(
            num_env_runners=1,  # Number of parallel workers
            rollout_fragment_length="auto",  # Collect more samples per rollout
            sample_timeout_s=120,
        )
        .training(
            train_batch_size=2000,
            num_sgd_iter=10,
        )
        #.resources(
        #    num_gpus=1,
        #) 
    )
    return config


def train_agent(checkpoint_dir, primitives, primitive_names, primitive_costs, dataset, feature_dim, max_steps):
    ray.init(local_mode=True, logging_level=logging.DEBUG)
    config = make_config(primitives, primitive_names, primitive_costs, dataset, feature_dim, max_steps)
    algo = config.build()

    best_reward = float('-inf')
    best_iteration = 0

    for i in range(100):
        print(f" Starting iteration {i}")
        result = algo.train()
        
        if 'env_runners' in result:
            env_metrics = result['env_runners']
            reward_mean = env_metrics.get('episode_reward_mean', float('-inf'))
            episode_len_mean = env_metrics.get('episode_len_mean', 0)
            num_episodes = env_metrics.get('num_episodes', 0)
            episodes_timesteps_total = env_metrics.get('episodes_timesteps_total', 0)
            
            print(f"Iteration {i} results:")
            print(f"Reward Mean: {reward_mean:.3f}")
            print(f"Episode Length: {episode_len_mean:.2f} steps")
            print(f"Episodes: {num_episodes}")
            print(f"Total Timesteps: {episodes_timesteps_total}")

            # Check if this is the best performance so far
            if reward_mean > best_reward:
                best_reward = reward_mean
                best_iteration = i
                
                # Save the best checkpoint
                print(f"NEW BEST REWARD: {best_reward:.3f} at iteration {best_iteration}")
                print(f"Saving best checkpoint to: {checkpoint_dir}")
             
                algo.save(checkpoint_dir=checkpoint_dir)

            else:
                print(f"Current best: {best_reward:.3f} (iteration {best_iteration})")
        
        if 'info' in result and 'learner' in result['info']:
            learner_info = result['info']['learner']
            if 'default_policy' in learner_info:
                policy_stats = learner_info['default_policy']
                if 'learner_stats' in policy_stats:
                    stats = policy_stats['learner_stats']
                    print(f"Policy Loss: {stats.get('total_loss', 'N/A')}")
                    print(f"Policy Entropy: {stats.get('entropy', 'N/A')}")

        # Early stopping if reward is very good
        if reward_mean > 0.85:  # 85% success rate
            print(f"CONVERGED! Reward > 0.85")
            print(f"Stopping early at iteration {i}")
            break
    
    algo.restore(checkpoint_dir)

    return algo

def load_agent(checkpoint_dir, primitives, primitive_names, primitive_costs, dataset, feature_dim, max_steps):
    ray.init(local_mode=True, logging_level=logging.DEBUG)
    config = make_config(primitives, primitive_names, primitive_costs, dataset, feature_dim, max_steps)
    algo = config.build()
    algo.restore(checkpoint_dir)

    return algo

def evaluate_agent(algo, primitives, primitive_names, primitive_costs, test_dataset_path, feature_dim, max_steps):
    """Evaluate a trained PPO agent on the test dataset"""
    
    results = {
        'correct': 0,
        'total': 0,
        'total_attempts': 0,
        'algorithm_usage': {name: 0 for name in primitive_names}
    }

    with open(test_dataset_path, "r") as f:
        test_dataset = json.load(f)

    for i, sample in enumerate(test_dataset):
        # Create a temporary env for this episode
        env = ValueMatchingEnv({
            'primitives': primitives,
            'primitive_names': primitive_names,
            'costs': primitive_costs,
            'dataset': [sample],
            'feature_dim': feature_dim,
            'max_steps': max_steps
        })
        
        print(f"\nEvaluating sample {i+1}/{len(test_dataset)}: Source='{sample['source_value']}' Gold='{sample['gold_value']}'")
        state, info = env.reset()
        done = False
        
        while not done:
            # Use the trained policy to select action (greedy, no exploration)
            action = algo.compute_single_action(
                state, 
                explore=False  # Greedy policy for evaluation
            )
            print(f"Action taken: {primitive_names[action]}")
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            print(f"Predicted: {info['predicted']}, Correct: {info['correct']}, Attempts: {info['attempts']}")
        
        # Record results
        results['total'] += 1
        results['total_attempts'] += info['attempts']
        
        if info['correct']:
            results['correct'] += 1
        
        # Count algorithm usage
        for used_algorithm in info['history']:
            if used_algorithm != -1:
                results['algorithm_usage'][primitive_names[used_algorithm]] += 1
    
    # Print evaluation results
    print(f"\n=== Evaluation Results ===")
    print(f"Accuracy: {results['correct'] / results['total']:.3f}")
    print(f"Average Attempts: {results['total_attempts'] / results['total']:.3f}")
    print(f"Algorithm Usage: {results['algorithm_usage']}")
    
    return results
