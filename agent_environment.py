import gymnasium as gym
import numpy as np
from feature_extractor import compute_features


class ValueMatchingEnv(gym.Env):

    def __init__(self, primitives, feature_dim, max_steps=3):
        super(ValueMatchingEnv, self).__init__()
        self.primitives = primitives
        self.feature_dim = feature_dim
        self.max_steps = max_steps
        self.action_space = gym.spaces.Discrete(len(primitives))
        self.steps_taken = 0
        self.action_history = [-1] * self.max_steps

        obs_dim = feature_dim + max_steps
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def _encode_history(self):
        """Normalize action history to [0, 1] range for better working with NN"""
        encoded = []
        for action in self.action_history:
            if action == -1:
                encoded.append(0.0)
            else:
                encoded.append((action + 1) / len(self.primitives))

        return encoded
    
    def reset(self, source, targets, gold):
        # Reset episode state
        self.steps_taken = 0
        self.action_history = [-1] * self.max_steps
        
        # Compute initial state
        features = compute_features(source, targets)
        action_history_encoded = self._encode_history()
        state = features + action_history_encoded
        state = np.array(state, dtype=np.float32)
        self.source = source
        self.targets = targets
        self.gold = gold
        self.features = features
        
        return state
    
    def step(self, action):
        # Record algorithm in history
        if self.steps_taken < self.max_steps:
            self.action_history[self.steps_taken] = action

        self.steps_taken += 1

        # TODO: Remove this hack when  everything is working
        if action == 2:  # LLM reasoning
            predicted = self.gold  # Assume perfect prediction for LLM
        else:
            predicted = self.primitives[action](self.source, self.targets)

        # Check if correct
        is_correct = (predicted == self.gold)
        
        # Compute reward with decay based on attempts
        if is_correct:
            reward = 1.0 - 0.2 * (self.steps_taken - 1)  # Decay: 1.0, 0.8, 0.6
            done = True
        elif self.steps_taken >= self.max_steps:
            reward = -1.0  # Failed after max attempts
            done = True
        else:
            reward = -0.1  # Small penalty for wrong attempt
            done = False

        action_history_encoded = self._encode_history()
        next_state = self.features + action_history_encoded
        next_state = np.array(next_state, dtype=np.float32)

        truncated = False  # There is no artificial cutoffs for this approach
        info = {
            'predicted': predicted,
            'correct': is_correct,
            'attempts': self.steps_taken,
            'history': self.action_history
        }
        return next_state, reward, done, truncated, info