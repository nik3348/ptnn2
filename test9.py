import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from torchvision import models
from torch.utils.tensorboard import SummaryWriter


episodes = 1000
seq_length = 64
batch_size = 128
decay_factor = 0.99
learning_rate = 1e-4
epsilon = 1.0
isLoad = True
human = True
name = "Breakout1-4"
model_path = f"models/{name}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.in_channels = in_channels
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                self._make_dense_layer(in_channels + i * growth_rate, growth_rate)
            )

    def _make_dense_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.transition(x)


class DenseNet(nn.Module):
    def __init__(self, num_blocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()

        self.num_blocks = num_blocks
        self.growth_rate = growth_rate

        num_init_features = 2 * growth_rate
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
        )

        num_features = num_init_features
        for i, num_layers in enumerate(num_blocks):
            block = DenseBlock(num_features, growth_rate, num_layers)
            self.features.add_module(f"denseblock_{i}", block)
            num_features += num_layers * growth_rate
            if i != len(num_blocks) - 1:
                out_channels = int(num_features * reduction)
                transition = TransitionLayer(num_features, out_channels)
                self.features.add_module(f"transition_{i}", transition)
                num_features = out_channels

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = x.unsqueeze(0)
        features = self.features(x)
        out = self.avgpool(features)
        out = out.view(features.size(0), -1)
        out = self.classifier(out)
        return out.view(-1)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = models.resnet50().to(device)

        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = x.unsqueeze(0)
        x = self.resnet(x)
        return x.view(-1)


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
        self.fc1 = nn.Linear(1003, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, device=device)
        self.fc2 = nn.Linear(hidden_size, hidden_size, device=device)
        self.vn = nn.Linear(hidden_size, 1, device=device)
        self.an = nn.Linear(hidden_size, output_size, device=device)

    def forward(self, input, hidden):
        output = self.fc1(input)
        output, hidden = self.lstm(output, hidden)
        output = output[:, -1, :]
        output = torch.relu(self.fc2(output))

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
        temperature=1.0,
        temperature_decay=0.995,
        temperature_min=0.01,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.temperature_min = temperature_min
        self.tau = 0
        self.tau_prime = 10

        self.policy_net = LSTM_DQN(input_size, hidden_size, output_size).to(device)
        self.target_net = LSTM_DQN(input_size, hidden_size, output_size).to(device)
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + list(self.target_net.parameters()),
            lr=learning_rate,
        )
        self.criterion = nn.MSELoss()

    def select_action(self, state):
        # try:
        #     val = input()
        #     val = int(val)
        #     if val < 0 or val >= self.output_size:
        #         val = 0
        # except Exception:
        #     val = 0
        # return val

        if random.random() < self.epsilon:
            return random.randint(0, self.output_size - 1)

        with torch.no_grad():
            state = torch.FloatTensor(np.array(state)).to(device)

            hidden = self.policy_net.init_hidden(state.unsqueeze(0).shape[0])
            q_values, _ = self.policy_net(state.unsqueeze(0), hidden)

            if self.temperature > self.temperature_min:
                self.temperature *= self.temperature_decay

            probabilities = torch.softmax(q_values / self.temperature, dim=-1)
            return torch.multinomial(probabilities, 1).squeeze(-1).item()

    def train(self, replay_buffer):
        self.tau += 1
        batch, indices, weights = replay_buffer.sample(batch_size)
        weights = torch.FloatTensor(weights).to(device)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.BoolTensor(dones).to(device)

        q_values, _ = self.policy_net(
            states, self.policy_net.init_hidden(states.shape[0])
        )
        policy_qvalues, _ = self.policy_net(
            next_states, self.target_net.init_hidden(next_states.shape[0])
        )
        target_qvalues, _ = self.target_net(
            next_states, self.target_net.init_hidden(next_states.shape[0])
        )

        q_values_action = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        best_action_indices = torch.argmax(policy_qvalues, dim=1)
        best_qvalues = target_qvalues.gather(
            1, best_action_indices.unsqueeze(1)
        ).squeeze(1)
        expected_q = rewards + (self.gamma * dones * best_qvalues)

        loss = nn.functional.mse_loss(q_values_action, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        td_errors = torch.abs(q_values_action - expected_q)
        replay_buffer.update_priorities(
            indices, td_errors.cpu().detach().numpy() + self.epsilon
        )

        self.replace_target_net()
        return loss.item()

    def replace_target_net(self):
        if self.tau % self.tau_prime == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def load(self):
        self.policy_net = torch.load(model_path + "1" + ".pt")
        self.target_net = torch.load(model_path + "2" + ".pt")
        print("Model loaded")

    def save(self):
        torch.save(self.policy_net, model_path + "1" + ".pt")
        torch.save(self.target_net, model_path + "2" + ".pt")

def normalize(state):
    max_pixel_value = torch.max(state)
    normalized_image_tensor = state / max_pixel_value

    mean_pixel_value = torch.mean(normalized_image_tensor)
    std_pixel_value = torch.std(normalized_image_tensor)
    return (normalized_image_tensor - mean_pixel_value) / std_pixel_value

if __name__ == "__main__":
    env = gym.make("ALE/Breakout-v5", render_mode="human" if human else "rgb_array")
    writer = SummaryWriter()
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    resnet = ResNet().to(device)
    agent = DQNAgent(128, hidden_size=128, output_size=output_size, epsilon=epsilon)

    if isLoad:
        agent.load()
        supreme_buffer = torch.load("buffer_alpha.pt")
        # supreme_buffer = PrioritizedReplayBuffer(capacity=10000)

    for episode in range(episodes):
        buffer = PrioritizedReplayBuffer(capacity=10000)

        state = env.reset()[0]
        state = torch.FloatTensor(state).to(device)
        state = normalize(state)
        state = resnet(state)
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

            total_reward += reward
            if reward == 0:
                reward -= 1

            next_state = torch.FloatTensor(next_state).to(device)
            next_state = normalize(next_state)
            next_state = resnet(next_state)
            next_state = torch.cat(
                (next_state, torch.tensor([reward, time_step, action]).to(device)),
                dim=0,
            )

            next = state.copy()
            next.append(next_state.cpu().detach().numpy())
            if len(next) > seq_length:
                next.pop(0)
            next_state = next

            done = terminated or truncated

            buffer.add((state, action, reward, next_state, done))
            state = next_state

        loss = agent.train(buffer)
        writer.add_scalar(f"{name}/loss", loss, episode)
        writer.add_scalar(f"{name}/total_reward", total_reward, episode)

        if episode % 10 == 0:
            print(f"Progress: {episode * 100/episodes}%, Total Reward: {total_reward}")
            agent.save()

        if total_reward >= 11:
            for i in range(buffer.size):
                supreme_buffer.add(buffer.buffer[i])
            print(supreme_buffer.size)
            torch.save(supreme_buffer, "buffer_alpha.pt")
