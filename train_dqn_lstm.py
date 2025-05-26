"""Train DQN with LSTM on real sepsis patient data
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from tqdm import tqdm

# Import custom data loader
from custom_data_loader import load_real_data

# Import additional modules for attention mechanism
import torch.nn.functional as F

# Custom feature extractor with Bidirectional LSTM and Attention
class BidirectionalLSTMWithAttention(BaseFeaturesExtractor):
    """
    Feature extractor that uses Bidirectional LSTM with Attention to process sequential data
    """
    def __init__(self, observation_space, features_dim=128):
        super(BidirectionalLSTMWithAttention, self).__init__(observation_space, features_dim)
        
        n_input_features = observation_space.shape[0]
        hidden_size = 256  # Hidden size for each direction
        
        # Bidirectional LSTM layer definition
        self.unflatten = nn.Unflatten(1, (1, n_input_features))
        self.lstm = nn.LSTM(
            input_size=n_input_features,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=2,
            dropout=0.2,
            bidirectional=True  # Make it bidirectional
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)  # 2x hidden size because bidirectional
        
        # Feature extraction layers
        self.linear1 = nn.Linear(hidden_size * 2, 128)
        self.linear2 = nn.Linear(128, features_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, observations):
        # Reshape input to add sequence dimension [batch_size, 1, features]
        x = self.unflatten(observations)
        
        # Process through Bidirectional LSTM
        # lstm_out shape: [batch_size, seq_len, hidden_size*2]
        lstm_out, _ = self.lstm(x)
        
        # Apply attention mechanism
        # Calculate attention weights
        attention_weights = self.attention(lstm_out)  # Shape: [batch_size, seq_len, 1]
        attention_weights = F.softmax(attention_weights, dim=1)  # Normalize along sequence dimension
        
        # Apply attention weights to get context vector
        context = torch.sum(attention_weights * lstm_out, dim=1)  # Shape: [batch_size, hidden_size*2]
        
        # Apply feature extraction layers
        x = self.relu(self.linear1(context))
        x = self.dropout(x)
        x = self.relu(self.linear2(x))
        
        return x

# Legacy LSTM Feature Extractor (kept for comparison)
class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    Original LSTM feature extractor (kept for comparison)
    """
    def __init__(self, observation_space, features_dim=128):
        super(LSTMFeatureExtractor, self).__init__(observation_space, features_dim)
        
        n_input_features = observation_space.shape[0]
        
        # LSTM layer definition
        self.unflatten = nn.Unflatten(1, (1, n_input_features))
        self.lstm = nn.LSTM(
            input_size=n_input_features,
            hidden_size=256,
            batch_first=True,
            num_layers=2,
            dropout=0.2
        )
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, features_dim)
        self.relu = nn.ReLU()
    
    def forward(self, observations):
        # Reshape input to add sequence dimension [batch_size, 1, features]
        x = self.unflatten(observations)
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        x = self.relu(self.linear1(last_output))
        x = self.relu(self.linear2(x))
        return x

# Custom Gym environment for Sepsis prediction
class SepsisEnv(gym.Env):
    """Sepsis environment for Reinforcement Learning"""
    
    def __init__(self, df, balance_sampling=True, sepsis_sample_prob=0.5):
        super(SepsisEnv, self).__init__()
        
        self.df = df
        self.reward_range = (-10.0, 5.0)  # Adjusted based on new reward structure
        self.balance_sampling = balance_sampling
        self.sepsis_sample_prob = sepsis_sample_prob
        
        # Separate indices for sepsis and non-sepsis patients for balanced sampling
        self.sepsis_patient_indices = []
        self.non_sepsis_patient_indices = []
        
        # Find all patients with and without sepsis
        for patient in df['patient'].unique():
            patient_df = df[df['patient'] == patient]
            start_idx = patient_df.index[0]
            
            if patient_df['SepsisLabel'].sum() > 0:
                self.sepsis_patient_indices.append(start_idx)
            else:
                self.non_sepsis_patient_indices.append(start_idx)
        
        print(f"Environment initialized with {len(self.sepsis_patient_indices)} sepsis patients and {len(self.non_sepsis_patient_indices)} non-sepsis patients")
        
        # Use all patient indices for normal iteration
        self.patient_start_index = self.sepsis_patient_indices + self.non_sepsis_patient_indices
        self.index = 0
        
        # Binary action space: 0 for non-sepsis, 1 for sepsis
        self.action_space = spaces.Discrete(2)
        
        # Get feature columns (excluding non-feature columns)
        self.feature_cols = [col for col in df.columns if col not in [
            'SepsisLabel', 'patient', 'zeros_reward', 'ones_reward', 
            'end_episode', 'index'
        ]]
        
        print(f"Using {len(self.feature_cols)} features for observations")
        
        # Observation space: features from the patient data
        low = np.full(len(self.feature_cols), -np.inf)
        high = np.full(len(self.feature_cols), np.inf)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32)
    
    def _next_observation(self):
        """Get the next observation"""
        obs = self.df.loc[self.current_step, self.feature_cols].values.astype(np.float32)
        return obs
    
    def step(self, action):
        """Execute one time step"""
        self.current_step += 1
        
        # Get reward based on action
        if action == 0:
            reward = float(self.df.loc[self.current_step, 'zeros_reward'])
        else:
            reward = float(self.df.loc[self.current_step, 'ones_reward'])
        
        # Check if episode is done
        done = bool(self.df.loc[self.current_step, 'end_episode'])
        if done:
            self.index += 1
            if self.index >= len(self.patient_start_index):
                self.index = 0  # Reset to beginning if we've gone through all patients
        
        # Get next observation
        obs = self._next_observation()
        
        return obs, reward, done, False, {}  # Last False is for truncated flag in gym
    
    def reset(self, seed=None, options=None):
        """Reset the environment with optional balanced sampling"""
        super().reset(seed=seed)
        
        # Balanced sampling between sepsis and non-sepsis patients
        if self.balance_sampling and (len(self.sepsis_patient_indices) > 0 and len(self.non_sepsis_patient_indices) > 0):
            # With probability sepsis_sample_prob, choose a sepsis patient
            if np.random.random() < self.sepsis_sample_prob:
                idx = np.random.randint(0, len(self.sepsis_patient_indices))
                self.current_step = self.sepsis_patient_indices[idx]
            else:
                idx = np.random.randint(0, len(self.non_sepsis_patient_indices))
                self.current_step = self.non_sepsis_patient_indices[idx]
        else:
            # Set the current step to the start of a patient's data (sequential)
            if self.index < len(self.patient_start_index):
                self.current_step = self.patient_start_index[self.index]
            else:
                self.index = 0
                self.current_step = self.patient_start_index[0]
        
        return self._next_observation(), {}

def train_and_evaluate(data_path='data/training_setA', max_files=None, total_timesteps=100000, eval_episodes=100):
    """
    Train and evaluate the DQN+LSTM model on real patient data
    
    Args:
        data_path: Path to the directory with patient data files
        max_files: Maximum number of files to load (None to load all)
        total_timesteps: Total timesteps to train for
        eval_episodes: Number of episodes for evaluation
    """
    # Load real patient data with balanced sampling
    df = load_real_data(data_path, max_files, balance_ratio=0.4)
    
    # Create environment with balanced sampling
    env = SepsisEnv(df, balance_sampling=True, sepsis_sample_prob=0.5)
    
    # Create policy kwargs with bidirectional LSTM and attention feature extractor
    policy_kwargs = {
        "features_extractor_class": BidirectionalLSTMWithAttention,
        "features_extractor_kwargs": {"features_dim": 128},
        "net_arch": [128, 64]  # Define architecture for the policy network
    }
    
    # Learning rate scheduler callback
    class LRSchedulerCallback(BaseCallback):
        def __init__(self, initial_lr=5e-4, min_lr=5e-5, decay_factor=0.9, decay_steps=10000, verbose=1):
            super(LRSchedulerCallback, self).__init__(verbose)
            self.initial_lr = initial_lr
            self.min_lr = min_lr
            self.decay_factor = decay_factor
            self.decay_steps = decay_steps
            self.current_lr = initial_lr
        
        def _on_step(self):
            # Decay learning rate at specified intervals
            if self.n_calls % self.decay_steps == 0 and self.n_calls > 0:
                # Calculate new learning rate
                self.current_lr = max(self.current_lr * self.decay_factor, self.min_lr)
                # Update model's learning rate
                self.model.learning_rate = self.current_lr
                if self.verbose > 0:
                    print(f"Step {self.n_calls}: Learning rate adjusted to {self.current_lr:.6f}")
            return True
    
    # Checkpoint callback to save models during training
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/",
        name_prefix="dqn_lstm_sepsis"
    )
    
    # Create learning rate scheduler
    lr_scheduler = LRSchedulerCallback(
        initial_lr=5e-4,
        min_lr=5e-5,
        decay_factor=0.9,
        decay_steps=10000,
        verbose=1
    )
    
    # Create the DQN model with LSTM and improved parameters
    model = DQN(
        "MlpPolicy", 
        env, 
        learning_rate=5e-4,              # Lower learning rate for stability
        buffer_size=50000,               # Larger replay buffer
        learning_starts=1000,            # Start learning after 1000 steps
        batch_size=128,                  # Larger batch size
        gamma=0.99,                      # Discount factor
        exploration_fraction=0.2,        # Longer exploration
        exploration_final_eps=0.05,      # Higher final exploration rate
        policy_kwargs=policy_kwargs,
        tau=0.005,                       # Soft update coefficient
        train_freq=4,                    # Update the model every 4 steps
        gradient_steps=1,                # How many gradient steps to do after each rollout
        target_update_interval=1000,     # Update the target network every 1000 steps
        verbose=1
    )
    
    # Train the model
    print("\nTraining DQN with LSTM on real patient data...")
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, lr_scheduler])
    
    # Evaluate the model
    print("\nEvaluating the model...")
    
    # Run episodes for evaluation with detailed metrics
    total_reward = 0
    total_steps = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    # Create a separate environment for evaluation with sequential sampling
    eval_env = SepsisEnv(df, balance_sampling=False)
    
    for _ in tqdm(range(eval_episodes), desc="Evaluating"):
        obs, _ = eval_env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = eval_env.step(action)
            
            # Update metrics
            sepsis_label = eval_env.df.loc[eval_env.current_step, 'SepsisLabel']
            if action == 1 and sepsis_label == 1:
                true_positives += 1
            elif action == 1 and sepsis_label == 0:
                false_positives += 1
            elif action == 0 and sepsis_label == 0:
                true_negatives += 1
            elif action == 0 and sepsis_label == 1:
                false_negatives += 1
                
            episode_reward += reward
            episode_steps += 1
            
        total_reward += episode_reward
        total_steps += episode_steps
    
    # Calculate metrics
    total_predictions = true_positives + true_negatives + false_positives + false_negatives
    accuracy = (true_positives + true_negatives) / total_predictions if total_predictions > 0 else 0
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Specificity (true negative rate)
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    
    # Print detailed results
    print("\nEvaluation Results:")
    print(f"  - Total predictions: {total_predictions}")
    print(f"  - True Positives: {true_positives} ({true_positives/total_predictions:.2%} of all predictions)")
    print(f"  - False Positives: {false_positives} ({false_positives/total_predictions:.2%} of all predictions)")
    print(f"  - True Negatives: {true_negatives} ({true_negatives/total_predictions:.2%} of all predictions)")
    print(f"  - False Negatives: {false_negatives} ({false_negatives/total_predictions:.2%} of all predictions)")
    print(f"\nPerformance Metrics:")
    print(f"  - Average reward per episode: {total_reward / eval_episodes:.4f}")
    print(f"  - Average steps per episode: {total_steps / eval_episodes:.1f}")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall/Sensitivity: {recall:.4f}")
    print(f"  - Specificity: {specificity:.4f}")
    print(f"  - F1 Score: {f1_score:.4f}")
    
    return model

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs("./models", exist_ok=True)
    
    # Use all files for training
    max_files = None  # Set to None to use all files
    
    # Train for longer time
    total_timesteps = 100000  # Increase training time for better learning
    
    # Train and evaluate model on real patient data
    model = train_and_evaluate(max_files=max_files, total_timesteps=total_timesteps)
    
    # Save the final model
    model.save("dqn_lstm_sepsis_model_final")
    print("\nFinal model saved as dqn_lstm_sepsis_model_final")
    print("\nIntermediate models were saved in the './models/' directory")
