import gym
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.distributions import Categorical

class PPO_local(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_size):
        super().__init__()
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.hidden_size = hidden_size
        self.logits_net = nn.Sequential(
            nn.Linear(self.obs_shape, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.action_shape)
        )
        self.critic = nn.Sequential(
            nn.Linear(self.obs_shape, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1)
        )
        self.optimizer = Adam([
            {'params': self.logits_net.parameters(), 'lr': 1e-2},
            {'params': self.critic.parameters(), 'lr': 1e-2}
        ])
        self.MSELoss = nn.MSELoss()
        self.clip_param = 0.2
        self.gamma = 0.99
        self.batch_obs = []
        self.batch_acts = []
        self.batch_logp = []
        self.batch_rews = []

    def get_policy(self, obs_tensor):
        logits = self.logits_net(obs_tensor)
        policy = Categorical(logits=logits)
        return policy

    def action_select(self, obs):
        obs_tensor = torch.from_numpy(obs).type(torch.float32)
        dist = self.get_policy(obs_tensor)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action.item(), logp.item()

    def rewards_to_go(self):
        batch_rews = self.batch_rews.copy()
        rews_to_go = [sum(batch_rews[i:]) for i in range(len(batch_rews))]
        torch_rews_to_go = torch.tensor(np.array(rews_to_go), dtype=torch.float32)
        return torch_rews_to_go

    def get_batch(self):
        batch_obs = torch.tensor(np.array(self.batch_obs), dtype=torch.float32).detach()
        batch_acts = torch.tensor(np.array(self.batch_acts)).detach()
        batch_logp = torch.tensor(np.array(self.batch_logp)).detach()
        batch_rews = torch.tensor(np.array(self.batch_rews)).detach()
        return batch_obs, batch_acts, batch_logp, batch_rews
    def update(self):
        batch_obs, batch_acts, batch_old_logp, batch_rews = self.get_batch()
        batch_rets = self.rewards_to_go()
        batch_values = self.critic(batch_obs).squeeze()
        batch_advantages = batch_rets.detach() - batch_values.detach()

        logp = self.get_policy(batch_obs).log_prob(batch_acts)
        ratio = torch.exp(logp - batch_old_logp.detach())
        surr1 = ratio * batch_advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * batch_advantages
        loss = -torch.min(surr1, surr2) + 0.5 * self.MSELoss(batch_values, batch_rets)

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        return loss.mean()

    def add_to_buffer(self, obs, act, logp, rew):
        self.batch_obs.append(obs)
        self.batch_acts.append(act)
        self.batch_logp.append(logp)
        self.batch_rews.append(rew)

    def clear_buffer(self):
        self.batch_obs = []
        self.batch_acts = []
        self.batch_logp = []
        self.batch_rews = []


def train(env_id="CartPole-v1", hidden_size=32):
    env = gym.make(env_id)
    agent = PPO_local(env.observation_space.shape[0], env.action_space.n, hidden_size)
    # [1] add to buffer each timestep, [2] update agent each episode, [3] clear buffer each episode
    num_episodes = 300
    total_ep_rews = []
    for episode in range(num_episodes):
        obs, done, t = env.reset(), False, 0
        agent.clear_buffer()
        while not done:
            action, logp = agent.action_select(obs)
            new_obs, rew, done, _ = env.step(action)
            agent.add_to_buffer(obs, action, logp, rew)
            t += 1
            obs = new_obs
        loss = agent.update()
        ep_rew = sum(agent.batch_rews)
        total_ep_rews.append(ep_rew)
        print(f"| Trained Episode {episode} | Rewards: {ep_rew:<5} | Ep Len: {t:<3} | Loss: {loss:<5.1f} |")
    return total_ep_rews
def plot(ep_rews):
    import matplotlib.pyplot as plt
    plt.plot(ep_rews)
    plt.title("CartPole-v1, PPO_local, hidden_size=32, episodes=100")
    plt.xlabel("Episode")
    plt.ylabel("Total Rewards")
    plt.show()

if __name__ == "__main__":
    ep_rews = train()
    plot(ep_rews)