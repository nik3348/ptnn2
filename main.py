import torch
from torch import nn
import torch.optim as optim

import gymnasium as gym


class PPOPolicyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PPOPolicyLSTM, self).__init__()
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.fc_actor = nn.Linear(hidden_size, output_size)
        self.fc_critic = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden):
        hx, cx = self.lstm(x, hidden)
        actor_output = self.fc_actor(hx)
        critic_output = self.fc_critic(hx)
        return actor_output, critic_output, (hx, cx)


env = gym.make("ALE/Breakout-v5", render_mode="human")
PATH = "model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 0.001
gamma = 0.99
epsilon = 0.2
ep_length = 1000
num_episodes = 100
input_size = (
    env.observation_space.shape[0]
    * env.observation_space.shape[1]
    * env.observation_space.shape[2]
)
hidden_size = 256
output_size = env.action_space.n
model = PPOPolicyLSTM(input_size, hidden_size, output_size).to(device)
model = torch.load(PATH)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for episode in range(num_episodes):
    states, actions, values, rewards, old_probs = [], [], [], [], []
    observation, info = env.reset()

    for t in range(ep_length):
        observation = torch.from_numpy(observation).view(-1).to(torch.float32)
        states.append(observation)

        hidden = (torch.zeros(hidden_size), torch.zeros(hidden_size))

        with torch.no_grad():
            action_prob, value, hidden = model(observation, hidden)

        values.append(value)
        action_prob = torch.softmax(action_prob, dim=-1)
        action = torch.multinomial(action_prob, num_samples=1).item()

        old_probs.append(action_prob[action])
        actions.append(action)

        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        if terminated or truncated:
            break

    states = torch.stack(states).to(torch.float32).detach()
    actions = torch.tensor(actions).to(torch.int64).detach()
    values = torch.tensor(values).to(torch.float32).detach()
    old_probs = torch.tensor(old_probs).to(torch.float32).detach()

    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns).to(torch.float32)
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    hidden_state = torch.zeros((t + 1, hidden_size), dtype=torch.float32)
    cell_state = torch.zeros((t + 1, hidden_size), dtype=torch.float32)

    logits, values, _ = model(states, (hidden_state, cell_state))

    probs = torch.softmax(logits, dim=-1)

    new_probs = torch.gather(probs, 1, actions.view(-1, 1))
    ratio = torch.squeeze(new_probs) / old_probs

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()

    critic_loss = nn.functional.mse_loss(torch.squeeze(values), returns)

    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()

    loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if episode % 10 == 0:
        torch.save(model, PATH)

env.close()
torch.save(model, PATH)
