import sys
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from torch import nn

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import (
    RewardShapingInterface,
    TrainingInfoInterface,
    register_env,
)
from sample_factory.model.encoder import Encoder
from sample_factory.model.model_utils import nonlinearity
from sample_factory.train import run_rl
from sample_factory.enjoy import enjoy
from sample_factory.utils.typing import Config, ObsSpace, ActionSpace
from sf_examples.atari.atari_utils import ATARI_ENVS, make_atari_env


from sample_factory.model.encoder import Encoder
from sample_factory.model.decoder import Decoder
from sample_factory.model.core import ModelCore
from sample_factory.model.actor_critic import ActorCritic
from sample_factory.algo.utils.context import global_model_factory


class CustomEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)
        # build custom encoder architecture
        ...

    def forward(self, obs_dict):
        # custom forward logic
        ...


class CustomCore(ModelCore):
    def __init__(self, cfg: Config, input_size: int):
        super().__init__(cfg)
        # build custom core architecture
        ...

    def forward(self, head_output, rnn_states):
        # custom forward logic
        ...


class CustomDecoder(Decoder):
    def __init__(self, cfg: Config, decoder_input_size: int):
        super().__init__(cfg)
        # build custom decoder architecture
        ...

    def forward(self, core_output):
        # custom forward logic
        ...


class CustomActorCritic(ActorCritic):
    def __init__(
        self,
        model_factory,
        obs_space: ObsSpace,
        action_space: ActionSpace,
        cfg: Config,
    ):
        super().__init__(obs_space, action_space, cfg)

        self.encoder = CustomEncoder(cfg, obs_space)
        self.core = CustomCore(cfg, self.encoder.get_out_size())
        self.decoder = CustomDecoder(cfg, self.core.get_out_size())
        self.critic_linear = nn.Linear(self.decoder.get_out_size())
        self.action_parameterization = self.get_action_parameterization(
            self.decoder.get_out_size()
        )

    def forward(self, normalized_obs_dict, rnn_states, values_only=False):
        # forward logic
        ...


def register_model_components():
    global_model_factory().register_actor_critic_factory(CustomActorCritic)


def main():
    for env in ATARI_ENVS:
        register_env(env.name, make_atari_env)

    # register_model_components()
    parser, partial_cfg = parse_sf_args(argv=None, evaluation=False)
    cfg = parse_full_cfg(parser, None)
    cfg

    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
