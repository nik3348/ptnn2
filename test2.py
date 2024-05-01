import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym


class PPOPolicyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PPOPolicyLSTM, self).__init__()
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.fc_actor = nn.Linear(hidden_size, output_size)
        self.fc_critic = nn.Linear(hidden_size, 1)

        self.fc = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x = torch.relu(self.fc(x))
        x = self.fc2(x)
        return x, x, x

        hx, cx = self.lstm(x, hidden)
        actor_output = self.fc_actor(hx)
        critic_output = self.fc_critic(hx)
        return actor_output, critic_output, (hx, cx)


env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
PATH = "model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 0.001
gamma = 0.99
epsilon = 0.2
ep_length = 10
num_episodes = 100
input_size = (
    env.observation_space.shape[0]
    * env.observation_space.shape[1]
    * env.observation_space.shape[2]
)
hidden_size = 256
output_size = env.action_space.n
model = PPOPolicyLSTM(input_size, hidden_size, output_size).to(device)
# model = torch.load(PATH)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
writer = SummaryWriter()

for episode in range(num_episodes):
    states, next_state, actions, values, rewards, old_probs, dones = [], [], [], [], [], [], []
    observation, info = env.reset()

    for t in range(ep_length):
        observation = torch.from_numpy(observation).view(-1).to(torch.float32).to(device)
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
        next_state.append(torch.from_numpy(observation).view(-1).to(torch.float32).to(device))
        rewards.append(reward)

        if terminated or truncated:
            dones.append(1)
            break

        dones.append(0)

    states = torch.stack(states).to(torch.float32).detach()
    next_state = torch.stack(next_state).to(torch.float32).detach()
    actions = torch.tensor(actions).to(torch.int64).detach()
    # values = torch.tensor(values).to(torch.float32).detach()
    rewards = torch.tensor(rewards).to(torch.float32).detach()
    old_probs = torch.tensor(old_probs).to(torch.float32).detach()
    dones = torch.tensor(dones).to(torch.float32).detach()  

    hidden_state = torch.zeros((t + 1, hidden_size), dtype=torch.float32)
    cell_state = torch.zeros((t + 1, hidden_size), dtype=torch.float32)
    logits, _, _ = model(states, (hidden_state, cell_state))
    values, _, _ = model(next_state, (hidden_state, cell_state))
    print(reward + gamma * torch.max(values).item())
    print(logits.shape, values.shape, rewards.shape, dones.shape)
    target = rewards + (gamma * torch.squeeze(values) * (1 - dones))
    loss = nn.functional.mse_loss(old_probs, target)
    writer.add_scalar("Loss/train", loss, episode)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if episode % 10 == 0:
        torch.save(model, PATH)
        print(f"Episode: {episode}, Loss: {loss.item()}")

writer.flush()
env.close()
torch.save(model, PATH)
