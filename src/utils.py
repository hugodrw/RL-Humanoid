import numpy as np
#import scipy.signal
from gym.spaces import Box

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

def combined_shape(length, shape=None):
    pass


# on laisse la possibilité de moduler le NN ou on en code un immutable dans l'actor et le critic?
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    pass

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    pass


class MLPGaussianActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        std = 0.5 * np.ones(act_dim, dtype=np.float32)  # might change: it's possible to have a dynamic std to better explore (https://kae1506.medium.com/actor-critic-methods-with-continous-action-spaces-having-too-many-things-to-do-e4ff69cd537d)
        self.std = torch.nn.Parameter(torch.as_tensor(std)) 
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        return Normal(mu, self.std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act) # OK for TRPO but is it used in PPO ? (cf Loss TRPO vs PPO)
        return pi, logp_a

class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        pass
    def forward(self, obs):
        pass


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()
        pass

    def step(self, obs):
        with torch.no_grad():
            pass

    def act(self, obs):
        pass