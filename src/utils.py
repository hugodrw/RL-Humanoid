import numpy as np
#import scipy.signal
from gym.spaces import Box

import torch
import torch.nn as nn
from torch.distributions.normal import Normal


# on laisse la possibilité de moduler le NN ou on en code un immutable dans l'actor et le critic?
def mlp(sizes, activation, output_activation=nn.Identity):
    '''
        Spinup a variable size MLP
    '''
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def discount_cumsum(x, discount):
    """
    Computes discounted cumulative sums of vectors.

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
    discount_list = np.array([discount**i for i in range(len(x)+1)])
    return np.array([sum(discount_list[:-(1+i)]*x[i:]) for i in range(len(x))])


class MLPGaussianActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        # we initialize log_std and make it learnable
        # the std is used in the normal distribution used by the policy to control exploration
        # we learn log_std instead of std because 
        #   1. std must be positive whereas log_std is unbounded as the exponential will ensure it is positive (that simplifies the optimization process)
        #   2. log space is generally more numerically stable
        #   3. log standard deviation parameter can be adjusted more finely than the standard deviation to have a better control over exploration behavior of the policy
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        # mapping from state to best action (batch of states to batch of actions ?)
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        # Normal distribution used by the policy to control exploration
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        # log prob, not sure when it is used (in TRPO Loss-> OK, in PPO loss -> ??)
        return pi.log_prob(act).sum(axis=-1)

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act) 

        # NB: pi is a vector normal distribution of size act_dim,
        # logp_a is a scalar for a given action
        return pi, logp_a 


class MLPCritic(nn.Module):
    
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)
    
    def forward(self, obs):
        return self.v_net(obs).squeeze(-1) # squeeze to remove last dim -> critical to ensure v has right shape


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad(): # why no grad here ? check the computational graph in ppo.py
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(a)
            return a.numpy(), logp_a.numpy(), v.numpy() # why numpy instead of tensor? 

    def act(self, obs):
        return self.step(obs)[0]
