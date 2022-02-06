import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import random

import config
from normalizer import Normalizer
q_norm = Normalizer(1)
bc_norm = Normalizer(3)
l2_norm = Normalizer(3)

from state import RunningNorm
adv_norm = RunningNorm(1)

goal_norm = RunningNorm(config.GOAL1_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        goal_size = config.GOAL1_SIZE if config.D2RL else 0

        if config.BLIND:
            state_dim = config.LL_STATE_SIZE
            self.l1 = nn.Linear(state_dim + config.GOAL1_SIZE - goal_size, 256)
        else:
            self.l1 = nn.Linear(state_dim - config.GOAL0_SIZE * 2 + config.GOAL1_SIZE - goal_size, 256)

        self.l2 = nn.Linear(256 + goal_size, 256)
        self.l22 = nn.Linear(256 + goal_size, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action

    def forward(self, state):
        state = state[:, config.GOAL0_SIZE:]
        goal = state[:, -config.GOAL1_SIZE:]
        state = state[:, :-config.GOAL1_SIZE]
        if config.BLIND:
            state = state[:, :config.LL_STATE_SIZE]

        if not config.PPO_NORM_IN:
            goal = goal_norm(goal, update=False)#torch.tanh(goal)
        if not config.D2RL:
            state = torch.cat([state, goal], 1)
        a = F.relu(self.l1(state))
        if config.D2RL:
            a = F.relu(self.l2(torch.cat([a, goal], 1)))
            a = F.relu(self.l22(torch.cat([a, goal], 1)))
        else:
            a = F.relu(self.l2(a))
            a = F.relu(self.l22(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        goal_size = config.GOAL1_SIZE if config.D2RL else 0
        # Q1 architecture
        #self.l1 = nn.Linear(state_dim + action_dim + config.GOAL1_SIZE, 256)
        self.l1 = nn.Linear(state_dim - config.GOAL0_SIZE + action_dim + config.GOAL1_SIZE - goal_size, 256)
        self.l2 = nn.Linear(256+goal_size, 256)
        self.l22 = nn.Linear(256+goal_size, 256)
        self.l3 = nn.Linear(256, 2)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim -config.GOAL0_SIZE + action_dim + config.GOAL1_SIZE - goal_size, 256)
        self.l5 = nn.Linear(256+goal_size, 256)
        self.l55 = nn.Linear(256+goal_size, 256)
        self.l6 = nn.Linear(256, 2)


    def forward(self, state, action, goal):
        state = state[:, config.GOAL0_SIZE:]
        if not config.D2RL:
            state = torch.cat([state, goal], 1)
        if not config.PPO_NORM_IN:
#            goal = torch.tanh(goal)
            goal = goal_norm(goal, update=1 != len(state))#torch.tanh(goal)

        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        if config.D2RL:
            q1 = F.relu(self.l2(torch.cat([q1, goal], 1)))
            q1 = F.relu(self.l22(torch.cat([q1, goal], 1)))
        else:
            q1 = F.relu(self.l2(q1))
            q1 = F.relu(self.l22(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        if config.D2RL:
            q2 = F.relu(self.l5(torch.cat([q2, goal], 1)))
            q2 = F.relu(self.l55(torch.cat([q2, goal], 1)))
        else:
            q2 = F.relu(self.l5(q2))
            q2 = F.relu(self.l55(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action, goal):
        state = state[:, config.GOAL0_SIZE:]

        if not config.D2RL:
            state = torch.cat([state, goal], 1)
        if not config.PPO_NORM_IN:
            #goal = torch.tanh(goal)
            goal = goal_norm(goal, update=1 != len(state))#torch.tanh(goal)

        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        if config.D2RL:
            q1 = F.relu(self.l2(torch.cat([q1, goal], 1)))
            q1 = F.relu(self.l22(torch.cat([q1, goal], 1)))
        else:
            q1 = F.relu(self.l2(q1))
            q1 = F.relu(self.l22(q1))
        q1 = self.l3(q1)
        return q1

class TD3_BC(object):
    def __init__(
        self, 
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        alpha=2.5,
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        self.total_it = 0

    def set_opt(self, model):
        def acp():
            for p in self.actor.parameters():
                yield p
            if not config.TD3_PPO_GRADS:
                return
            for p in model.actor.parameters():
                yield p

        self.actor_optimizer = torch.optim.Adam(acp(), lr=3e-4)

    def select_action(self, state, train_mode=True):
        assert 1 == len(state) or not train_mode
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()

        if train_mode:
            action = action.flatten()
            action += 0.2 * np.random.randn(len(action))
            action = action.clip(-1, 1)

            random_actions = np.random.uniform(low=-1, high=1, size=len(action))
            action += np.random.binomial(1, 0.3, 1)[0] * (random_actions - action)

        return torch.from_numpy(action.clip(-1, 1)).view(1, -1)

    def train_critic(self, replay_buffer, get_goal, batch_size, use_norm, do_update):
        self.total_it += 1

        # Sample replay buffer 
        state, action, next_state, reward, discount, goals_, oldp_, return_, oracle = replay_buffer.sample(batch_size)

        d1 = get_goal(state)
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
#            assert all((a-b).abs().sum() < 1e-5 for a,b in zip(
#                    next_state[:, -config.GOAL0_SIZE:], get_goal(os)[:, -config.GOAL0_SIZE:])), "nope {} vs {}".format(
#                    next_state[0, -config.GOAL0_SIZE:], get_goal(os)[0, -config.GOAL0_SIZE:]
#                )
            next_goal = get_goal(next_state).sample()
            next_action = (
#                self.actor_target(next_state) + noise
                self.actor_target(torch.cat([next_state[:, :-config.GOAL0_SIZE], next_goal], 1)) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action, next_goal)
            target_Q = torch.min(target_Q1, target_Q2)

            kl = torch.distributions.kl_divergence(
                d1, get_goal(oracle)).mean(1)
            if random.random() < .01: print("KL--->", kl.mean())
            reward[:, 0] = config.KL_SCALE * (config.KL_DELTA + (-1. + (kl < config.KL_MIN)))

            target_Q = reward + discount * target_Q
            #print("\n".join(["%.2f -> %.2f"%(a, b) for a, b in zip(reward, discount)]))
            if config.CLIP_Q: assert False
            if config.CLIP_Q: target_Q = target_Q.clamp(-1. / (1.-self.discount), 0)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action, d1.sample())
#        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        l1 = current_Q1 - target_Q
        l2 = current_Q2 - target_Q

        if use_norm:
            l1 = adv_norm(l1, update=do_update)
            l2 = adv_norm(l2, update=do_update)

        if config.SEPERATE_CRITICS:
            critic_loss = l1.pow(2).mean(0)[0] + l2.pow(2).mean(0)[0]
        else:
            critic_loss = l1.pow(2).mean() + l2.pow(2).mean()
        # Compute critic loss

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss.item()

    def train_actor(self, replay_buffer, use_bc, get_goal, batch_size=256):
        # Delayed policy updates
        if self.total_it % self.policy_freq:
            return 0.

        # Sample replay buffer 
        state, action, next_state, reward, discount, goals, log_probs, return_, oracle = replay_buffer.sample(batch_size)

        d1 = get_goal(state)

        if config.ADVANTAGE:
            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = (
                    torch.randn_like(action) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)

                next_goal = get_goal(next_state).sample()
                next_action = (
    #                self.actor_target(next_state) + noise
                    self.actor_target(torch.cat([next_state[:, :-config.GOAL0_SIZE], next_goal], 1)) + noise
                ).clamp(-self.max_action, self.max_action)

                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_state, next_action, next_goal)
                target_Q = torch.min(target_Q1, target_Q2)

                kl = torch.distributions.kl_divergence(
                    d1, get_goal(oracle)).mean(1)
                reward[:, 0] = config.KL_SCALE * (config.KL_DELTA + (-1. + (kl < config.KL_MIN)))

                target_Q = reward + discount * target_Q

        # Compute actor loss
        goal = d1.sample()
        pi = self.actor(torch.cat([ state[:, :-config.GOAL0_SIZE], goal ], 1))
        #pi = self.actor(state)
        Q = self.critic.Q1(state, pi, goal)
        
        adv = Q if not config.ADVANTAGE else (Q - target_Q)
        if config.ADV_NORM:
            adv = adv_norm(adv, update=True)
        actor_loss = -adv.mean(0)[0] + pi.pow(2).mean()

# PPO STYLE CLIP - will weight older experiences more carefully

        if config.TD3_PPO_CLIP:

            ratio = (
                d1.log_prob(goals) - log_probs
                ).exp().mean(1, keepdim=True)

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - 2e-1, 1.0 + 2e-1) * adv
            actor_loss  = -torch.min(surr1, surr2).mean()# + pi.pow(2).mean()

# PPO STYLE CLIP - as for new ones, no particular effect should be made
            
        # Optimize the actor 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss.item()

    def polyak(self, tau=None):
        tau = tau if tau is not None else self.tau
# Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def rpolyak(self, tau=None):
        tau = tau if tau is not None else self.tau
# Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def polyak_actor(self, tau=None):
        tau = tau if tau is not None else self.tau
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def polyak_critic(self, tau=None):
        tau = tau if tau is not None else self.tau
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

