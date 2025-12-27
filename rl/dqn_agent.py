"""
DQN Agent for Clash Royale RL

Deep Q-Network agent that learns to play Clash Royale through experience.
Trains locally on Mac (CPU or MPS for Apple Silicon).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Tuple, List
import os


class DQNNetwork(nn.Module):
    """
    Neural network for Q-value approximation

    Architecture:
    - Input: State vector (1171 values)
    - Hidden layers: 512 -> 256 -> 128
    - Output: Q-values for each action
    """

    def __init__(self, state_size: int, action_size: int):
        super(DQNNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions
    """

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store a transition"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of transitions"""
        batch = random.sample(self.buffer, batch_size)

        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN agent with experience replay and target network
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        device: str = None
    ):
        """
        Initialize DQN agent

        Args:
            state_size: Size of state vector
            action_size: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: How often to update target network
            device: 'cuda', 'mps', 'cpu', or None (auto-detect)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print("Using CUDA GPU for training")
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
                print("Using Apple Silicon MPS for training")
            else:
                self.device = torch.device('cpu')
                print("Using CPU for training")
        else:
            self.device = torch.device(device)

        # Q-networks
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Training stats
        self.steps = 0
        self.episodes = 0
        self.total_reward = 0
        self.episode_rewards = []
        self.losses = []

    def select_action(self, state: np.ndarray, valid_actions: List[int] = None) -> int:
        """
        Select action using epsilon-greedy policy

        Args:
            state: Current state vector
            valid_actions: List of valid action indices (None = all actions valid)

        Returns:
            Selected action index
        """
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            # Random action
            if valid_actions is not None:
                return random.choice(valid_actions)
            return random.randint(0, self.action_size - 1)

        # Greedy action (exploitation)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).cpu().numpy()[0]

            # Mask invalid actions
            if valid_actions is not None:
                mask = np.full(self.action_size, -np.inf)
                mask[valid_actions] = 0
                q_values = q_values + mask

            return np.argmax(q_values)

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_reward += reward

    def train_step(self) -> float:
        """
        Perform one training step

        Returns:
            Loss value
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Update steps
        self.steps += 1

        # Update target network
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        loss_value = loss.item()
        self.losses.append(loss_value)

        return loss_value

    def end_episode(self, episode_reward: float = None):
        """Mark end of episode and update stats"""
        self.episodes += 1

        if episode_reward is not None:
            self.episode_rewards.append(episode_reward)
        else:
            self.episode_rewards.append(self.total_reward)

        self.total_reward = 0

    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes,
            'episode_rewards': self.episode_rewards,
            'state_size': self.state_size,
            'action_size': self.action_size,
        }

        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        if not os.path.exists(filepath):
            print(f"Checkpoint not found: {filepath}")
            return False

        checkpoint = torch.load(filepath, map_location=self.device)

        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        self.episode_rewards = checkpoint['episode_rewards']

        print(f"Checkpoint loaded: {filepath}")
        print(f"  Episodes: {self.episodes}")
        print(f"  Steps: {self.steps}")
        print(f"  Epsilon: {self.epsilon:.4f}")

        return True

    def get_stats(self) -> dict:
        """Get training statistics"""
        recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) > 0 else [0]
        recent_losses = self.losses[-100:] if len(self.losses) > 0 else [0]

        return {
            'episodes': self.episodes,
            'steps': self.steps,
            'epsilon': self.epsilon,
            'avg_reward_100': np.mean(recent_rewards),
            'avg_loss_100': np.mean(recent_losses),
            'buffer_size': len(self.replay_buffer),
        }
