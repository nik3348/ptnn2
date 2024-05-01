import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter


# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super(QNetwork, self).__init__()
        self.fc = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        x = self.fc2(x)
        return x


# Define the Deep Q-Network Agent
class DQNAgent:
    def __init__(
        self,
        state_size,
        hidden_size,
        action_size,
        lr=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
    ):
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.q_network = QNetwork(state_size, hidden_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.writer = SummaryWriter()

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state)
                return torch.argmax(q_values).item()

    def train(self, state, action, reward, next_state, done, episode):
        state = torch.from_numpy(state).view(-1).to(torch.float32)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        q_values = self.q_network(state)
        next_q_values = self.q_network(next_state)
        target = q_values.clone()
        clipped_reward = np.clip(reward, -1.0, 1.0)

        with torch.no_grad():
            if done:
                target[action] = clipped_reward
            else:
                target[action] = (
                    clipped_reward + self.gamma * next_q_values[0][action]
                )

        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.writer.add_scalar("Loss/train", loss, episode)

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def load(self):
        self.q_network = torch.load("model.pt")

    def save(self):
        torch.save(self.q_network, "model.pt")


# Main training loop
def train_dqn(env_name, num_episodes, max_steps):
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    hidden_size = 256
    action_size = env.action_space.n
    agent = DQNAgent(state_size, hidden_size, action_size)
    agent.load()

    for episode in range(num_episodes):
        state = env.reset()
        state = state[0]
        total_reward = 0
        for step in range(max_steps):
            action = agent.select_action(state)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.train(state, action, reward, observation, done, episode)
            state = observation
            total_reward += reward
            if done:
                break
        print("Episode {}: Total Reward = {}".format(episode + 1, total_reward))

        if episode % 100 == 0:
            agent.save()
            print("Model saved")
    env.close()


if __name__ == "__main__":
    env_name = "CartPole-v1"
    num_episodes = 1000
    max_steps = 500
    train_dqn(env_name, num_episodes, max_steps)
