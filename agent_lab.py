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

    for i in range(100):
        print(f"Iteration {i}")
        result = algo.train()
        # Print available keys on first iteration to debug
        if i == 0:
            print(f"Available result keys: {list(result.keys())}")

        if "episode_reward_mean" in result:
            reward = result["episode_reward_mean"]
            print(f"Reward = {reward}")

    algo.save(checkpoint_dir=checkpoint_dir)

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
