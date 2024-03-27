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
        self.obs_buf = np.zeros((buffer_size, obs_dim), dtype=np.float32) # observation
        self.act_buf = np.zeros((buffer_size, act_dim), dtype=np.float32) # action
        self.adv_buf = np.zeros(buffer_size, dtype=np.float32) # advantage
        self.rew_buf = np.zeros(buffer_size, dtype=np.float32) # reward
        self.ret_buf = np.zeros(buffer_size, dtype=np.float32) # return
        self.val_buf = np.zeros(buffer_size, dtype=np.float32) # value
        self.logp_buf = np.zeros(buffer_size, dtype=np.float32) # log probability

        self.gamma, self.lam = gamma, lam
        self.pos = 0  # pos is the index of a given timestep of agent-environment interaction in the buffer
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
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda (https://arxiv.org/pdf/1506.02438.pdf ),
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        path_slice = slice(self.path_start_idx, self.pos)
        # append value at the end of reward list:
        #   if timeout: enable boostraping 
        #   if terminal state: value is reward
        rewards = np.append(self.rew_buf[path_slice], last_val)
        values = np.append(self.val_buf[path_slice], last_val)

        # GAE-Lambda advantage calculation
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1] # TD error
        self.adv_buf[path_slice] = utils.discount_cumsum(deltas, self.gamma * self.lam) 

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = utils.discount_cumsum(rewards, self.gamma)[:-1]         # why [:-1] ?


    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get   ------ if not full, what happens ? 

        # advantage normalization tricks
        mean_adv, mean_std = self.adv_buf.mean(), self.adv_buf.std()
        normalized_adv = (self.adv_buf - mean_adv) / mean_std

        # we create a dict with all the data from the buffer
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)

        # we transform dict values in torch tensor
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}



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
    