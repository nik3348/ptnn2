import ray
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.algorithms import ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.logger import pretty_print
import gymnasium as gym

ray.init()

torch, nn = try_import_torch()


class CustomTorchModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.fc1 = nn.Linear(obs_space.shape[0], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_outputs)

        self.fc4 = nn.Linear(64, 1)

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        self.value = self.fc4(x)
        x = self.fc3(x)

        return x, state

    def value_function(self):
        return self.value.squeeze(1)


ModelCatalog.register_custom_model("my_torch_model", CustomTorchModel)

env_name = "ALE/Breakout-v5"

algo = ppo.PPO(
    env=env_name,
    config={
        "framework": "torch",
        # "model": {
        #     "custom_model": "my_torch_model",
        # },
    },
)


checkpoint_dir = './models'
# algo.load_checkpoint(checkpoint_dir)

for i in range(1):
    result = algo.train()

    if i % 5 == 0:
        algo.save_checkpoint(checkpoint_dir)
        print(f"Checkpoint saved in directory {checkpoint_dir}")

env = gym.make(env_name)
obs, info = env.reset()
terminated = truncated = False
episode_reward = 0

while not terminated and not truncated:
    obs = torch.from_numpy(obs)
    print(obs.shape)
    action = algo.compute_single_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
