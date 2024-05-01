import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from modules.DenseNet import DenseNet


episodes = 1000
seq_length = 64
batch_size = 32
decay_factor = 0.99
learning_rate = 0.0001
isLoad = True
name = "Breakout1-3"
model_path = f"models/{name}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
writer = SummaryWriter()


class CNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=4)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=hidden_size, kernel_size=4, stride=2
        )
        self.fc1 = nn.Linear(hidden_size * 6 * 4, output_size, device=device)

    def forward(self, input):
        token = input.permute(2, 0, 1)
        token = self.conv1(token)
        token = torch.relu(token)
        token = self.pool(token)
        token = self.conv2(token)
        token = torch.relu(token)
        token = self.pool(token)
        token = token.view(-1)
        token = self.fc1(token)
        token = torch.relu(token)

        return token


class PrioritizedReplayBuffer:
    def __init__(
        self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-5
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.buffer = np.empty(capacity, dtype=object)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, transition, priority=1.0):
        self.buffer[self.pos] = transition
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        priorities = self.priorities[: self.size]
        probs = priorities**self.alpha
        probs /= probs.sum()
        indices = np.random.choice(self.size, batch_size, p=probs)
        samples = self.buffer[indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        self.priorities[indices] = priorities + self.epsilon

    def __len__(self):
        return self.size


class LSTM_DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_DQN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, device=device)
        self.fc = nn.Linear(hidden_size, hidden_size, device=device)
        self.vn = nn.Linear(hidden_size, 1, device=device)
        self.an = nn.Linear(hidden_size, output_size, device=device)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        output = output[:, -1, :]
        output = torch.relu(self.fc(output))

        advantage = self.an(output)
        state_value = self.vn(output)
        q_values = state_value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values, hidden

    def init_hidden(self, batch_size):
        return (
            torch.zeros(1, batch_size, self.hidden_size, device=device).to(device),
            torch.zeros(1, batch_size, self.hidden_size, device=device).to(device),
        )


class DQNAgent:
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.model1 = LSTM_DQN(input_size, hidden_size, output_size).to(device)
        self.model2 = LSTM_DQN(input_size, hidden_size, output_size).to(device)
        self.optimizer = optim.Adam(
            list(self.model1.parameters()) + list(self.model2.parameters()),
            lr=learning_rate,
        )
        self.criterion = nn.MSELoss()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.output_size - 1)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(np.array(state))

                hidden1 = self.model1.init_hidden(state.unsqueeze(0).shape[0])
                q_values1, _ = self.model1(state.unsqueeze(0), hidden1)

                hidden2 = self.model2.init_hidden(state.unsqueeze(0).shape[0])
                q_values2, _ = self.model2(state.unsqueeze(0), hidden2)

                q_values = q_values1 + q_values2
                return q_values.argmax().item()

    def train(self, sample):
        batch, indices, weights = sample
        weights = torch.FloatTensor(weights).to(device)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.BoolTensor(dones).to(device)

        next_q_values1, _ = self.model1(
            next_states, self.model1.init_hidden(next_states.shape[0])
        )
        next_q_values2, _ = self.model2(
            next_states, self.model2.init_hidden(next_states.shape[0])
        )

        q_values_next = next_q_values1 + next_q_values2
        best_actions = torch.argmax(q_values_next, dim=1)

        q_values_target1 = rewards + (1 - dones.float()) * self.gamma * torch.min(
            next_q_values1[range(next_states.shape[0]), best_actions]
        )
        q_values_target2 = rewards + (1 - dones.float()) * self.gamma * torch.min(
            next_q_values2[range(next_states.shape[0]), best_actions]
        )

        q_values, _ = self.model1(states, self.model1.init_hidden(states.shape[0]))
        q_values = torch.argmax(q_values, dim=1)

        td_errors = torch.abs(q_values_target1 - q_values)
        loss1 = torch.mean(weights * (q_values - q_values_target1) ** 2)
        loss2 = torch.mean(weights * (q_values - q_values_target2) ** 2)

        self.optimizer.zero_grad()
        (loss1 + loss2).backward()
        self.optimizer.step()
        writer.add_scalar(f"{name}/loss", (loss1 + loss2), episode)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        buffer.update_priorities(
            indices, td_errors.cpu().detach().numpy() + self.epsilon
        )

    def load(self):
        self.model1 = torch.load(model_path + "1" + ".pt")
        self.model2 = torch.load(model_path + "2" + ".pt")
        print("Model loaded")

    def save(self):
        torch.save(self.model1, model_path + "1" + ".pt")
        torch.save(self.model2, model_path + "2" + ".pt")
        print("Model saved")


if __name__ == "__main__":
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    buffer = PrioritizedReplayBuffer(capacity=10000)
    cnn = CNN(hidden_size=64, output_size=125).to(device)
    agent = DQNAgent(128, hidden_size=128, output_size=output_size)

    if isLoad:
        agent.load()

    for episode in range(episodes):
        state = env.reset()[0]
        state = torch.FloatTensor(state).to(device)
        state = cnn(state)
        state = torch.cat((state, torch.tensor([0.0, 0, 0]).to(device)), dim=0)

        state = state.cpu().detach().numpy()

        pad = [np.zeros_like(state) for _ in range(seq_length - 1)]
        pad.append(state)
        state = pad

        done = False
        time_step = 0
        total_reward = 0

        while not done:
            time_step += 1

            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            # if action == 1:
            #     total_reward = 0
            total_reward += reward
            # reward = total_reward

            next_state = torch.FloatTensor(next_state).to(device)
            next_state = cnn(next_state)
            next_state = torch.cat(
                (next_state, torch.tensor([reward, time_step, action]).to(device)), dim=0
            )

            next = state.copy()
            next.append(next_state.cpu().detach().numpy())
            if len(next) > seq_length:
                next.pop(0)
            next_state = next

            done = terminated or truncated

            buffer.add((state, action, reward, next_state, done))
            agent.train(buffer.sample(batch_size))
            state = next_state

        writer.add_scalar(f"{name}/time_step", total_reward, episode)

        if episode % 10 == 0:
            agent.save()
