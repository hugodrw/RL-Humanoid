import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import argparse
import src.utils as utils

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, buffer_size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((buffer_size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(buffer_size, dtype=np.float32)
        self.rew_buf = np.zeros(buffer_size, dtype=np.float32)
        self.ret_buf = np.zeros(buffer_size, dtype=np.float32)
        self.val_buf = np.zeros(buffer_size, dtype=np.float32)
        self.logp_buf = np.zeros(buffer_size, dtype=np.float32)

        self.gamma, self.lam = gamma, lam
        self.pos = 0  # pos is the index of a given timestep of agent-environment interaction
        self.path_start_idx = 0
        self.max_size = buffer_size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.pos] = obs
        self.act_buf[self.pos] = act
        self.rew_buf[self.pos] = rew
        self.val_buf[self.pos] = val
        self.logp_buf[self.pos] = logp
        self.pos += 1


    def finish_path(self, last_val=0):
        # https://arxiv.org/pdf/1506.02438.pdf
        pass

    def get(self):
        pass



def ppo(env_fn, actor_critic=utils.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, save_freq=10):
    """
    Proximal Policy Optimization (by clipping), 
    
    with early stopping based on approximate KL
    
    Args:
        env_fn : a function which creates a copy of the environment. 
        ----> Normalization of Observation, Observation clipping, Reward scaling, Reward clipping here ? 
        ----> to check https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/vec_env/vec_normalize.py#L39

        ...
    """
    # Random seed
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment 
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor critic module
    ac = actor_critic(env.observation_space, env.action_space)

    # Set up experience buffer
    buf = PPOBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)






    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        pass

    # Set up function for computing value loss
    def compute_loss_v(data):
        pass

    # Set up optimizers for policy and value function


    # Set up fonction to perform PPO update
    def update():
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Humanoid-v4')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    ppo(lambda : gym.make(args.env), actor_critic=utils.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs)
    