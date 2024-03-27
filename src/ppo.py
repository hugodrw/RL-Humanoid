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
        self.obs_buf = np.zeros((buffer_size, *obs_dim), dtype=np.float32) # transform (x, ) to x 
        self.act_buf = np.zeros((buffer_size, *act_dim), dtype=np.float32) # action
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
        assert self.pos < self.max_size     # buffer has to have room so you can store
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
        self.ret_buf[path_slice] = utils.discount_cumsum(rewards, self.gamma)[:-1]  
        
        self.path_start_idx = self.pos


    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.pos == self.max_size    # buffer has to be full before you can get
        self.pos, self.path_start_idx = 0, 0 # reset pointers
        # advantage normalization tricks
        mean_adv, mean_std = self.adv_buf.mean(), self.adv_buf.std()
        normalized_adv = (self.adv_buf - mean_adv) / mean_std

        # we create a dict with all the data from the buffer
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)

        # we transform dict values in torch tensor
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


def normalised_env_function(env, seed):
    """
    Environment preprocessing to enhance performance
    Tricks used : Normalization of Observation, Observation Clipping, Reward Scaling, reward Clipping
    """
    def thunk():
        env = gym.make(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        return env
    
    # return a function that will be instantiated in the ppo function
    return thunk



def ppo(env_fn, actor_critic=utils.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, save_freq=10):
    """
    Proximal Policy Optimization (by clipping), 
    
    with early stopping based on approximate KL
    
    Args:
        env_fn : a function that apply preprocessing.
        
        actor_critic: 
        ...
    """
    #  =========== Helper functions ===========

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
        '''
            Policy loss, as described in the "Proximal Policy Optimization Algorithms" paper
            https://arxiv.org/abs/1707.06347
        '''
        # Seperate the values of the data
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Compute the loss using the equation from the paper
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        normal_av = ratio * adv
        loss_pi = -(torch.min(normal_av, clip_adv)).mean()

        # Compute the approx KL. Optimisation trick described here: 
        # https://spinningup.openai.com/en/latest/algorithms/ppo.html#key-equations
        mean_approx_kl = (logp_old - logp).mean().item()

        return loss_pi, mean_approx_kl

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        # Manual implementation of L2 loss, could replace with torch implem
        # Label for the network is explicit return (sum of rewards)
        return ((ac.v(obs) - ret)**2).mean()

    # Set up optimizers for policy and value function used below
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Set up fonction to perform PPO update
    def update():
        '''
            Update the value and policy networks, called every epoch
        '''
        # Get the whole dataset from buffer
        data = buf.get()

        # Maximise the relative return based on the estimated values by updating the policy weights
        # Stops when the max iterations are hit or AVG KL divergence threshold is hit
        for i in range(train_pi_iters):
            # No batches, loss computed on entire RL epoch
            pi_optimizer.zero_grad()
            loss_pi, mean_approx_kl = compute_loss_pi(data) # TODO: check mpi_avg(pi_info['kl'])
            if mean_approx_kl > 1.5 * target_kl:
                print('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            # mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

        # Learn the new value function based on the latest policy
        for i in range(train_v_iters):
            # No batches, loss computed on entire RL epoch
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            # mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

        

    # =========== Main Script ===========
    # Reset the env, ep length and get the first observation
    (obs, _), episode_return, ep_len = env.reset(), 0, 0 # handle new version of gym


    # Main RL Loop
    for epoch in range(epochs):
        # Generate experience by acting in the environment, using the current policy
        # This runs until the buffer is full, and resets the environment when an episode ends
        # Many episodes are run sequentially, and the buffer is filled with the experience
        for step_count in range(steps_per_epoch):
            # Get action, value and logp used for policy backprop
            a, v, logp = ac.step(torch.as_tensor(obs, dtype=torch.float32))

            # Take a step in the env using the selected action
            next_o, r, term, _, _ = env.step(a)
            episode_return += r
            ep_len += 1
            
            # save to the buffer with the previous state, action and reward
            buf.store(obs, a, r, v, logp)

            # Update the observation
            obs = next_o

            # Check for special cases
            episode_timeout = ep_len == max_ep_len
            terminal = term or episode_timeout
            epoch_ended = step_count==steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # bootstrap value target if not terminal
                if episode_timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(obs, dtype=torch.float32))
                else:
                    v = 0
                if term and use_wandb:
                    # Log wandb
                    # wand logging
                    wandb.log({
                            "episode reward": episode_return,
                            "mean episode reward": episode_return/ep_len,
                            "episode length": ep_len,
                        })
                # Finish the path and restart the env for a new episode
                buf.finish_path(v)
                (obs, _), episode_return, ep_len = env.reset(), 0, 0
        
        # Perform PPO update for this epoch
        update()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Humanoid-v4')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--normalize_env', type=bool, default='False')

    args = parser.parse_args()

    print(args.cpu)
    print(args.normalize_env)

    use_wandb = True
    if use_wandb:
        import wandb
        wandb.init(project='RL-Humanoid')
    
    ppo(lambda: gym.make(args.env), actor_critic=utils.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs)
    
    # else:
    #     ppo(env_fn(args.env, args.seed), actor_critic=uls.MLPActorCritic,
    #         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
    #         seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs)
    
    