# -*- coding: utf-8 -*-
"""duh5_search.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cENuoHf5TCa62a3x3F73x9XEmGlfyTIQ

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rezer0dai/TD3_BC/blob/her/td3_bc_her.ipynb)
"""

import sys

import numpy as np
import torch
import gym
import argparse
import os

import utils
import TD3_BC
#import TD3
#import OurDDPG
import config

print(open("config.py").read())


import state
state_norm = state.GlobalNormalizerWithTime(config.STATE_SIZE)
state_norm.share_memory()

def eval_policy(n_epochs, policy, eval_env, seed, normalize_state, model, seed_offset=100, eval_episodes=10, pv=False):
    load_state = lambda obs: obs["observation"].reshape(1,-1)

    eval_env.seed(seed + seed_offset)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        ei = 0
        d1 = None
        dn = None
        kls = []
        while not done:
            ei += 1
            state = load_state(state)

            action, g, d, v = model(normalize_state(state), exploit=True)

            if d1 is None:
                d1 = d
            elif dn is not None:
                kls.append(torch.distributions.kl_divergence(d, dn).mean())
            dn = d

            action = action[0].view(-1)
            action = action.cpu().detach().numpy()

#            if pv: print(f"[{ei}] -> {v}")

            #action = policy.select_action(normalize_state(state))
            state, reward, done, _ = eval_env.step(action)
            eval_env.render()
            avg_reward += reward
#        print(f"KLs (x) : { torch.distributions.kl_divergence(d1, dn).mean() } - [\n {' -> '.join(str(kl) for kl in kls)} \n] - (0) : { torch.distributions.kl_divergence(d1, dn).mean(0) }")

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"[{n_epochs}] Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")

    print(f"KLs (0) : { torch.distributions.kl_divergence(d1, dn).mean(0) }, (1) : { torch.distributions.kl_divergence(d1, dn).mean(1)}, (x) : { torch.distributions.kl_divergence(d1, dn).mean() }")
    return avg_reward

import random
from copy import deepcopy

def her(replay_buffer, achieved_goals, her_ratio, n_future, fail_filter, dummy_filter, kl, kl_filter=.8):
    if not len(achieved_goals):
        return False

    total = 0
    replay_buffer.ptr = replay_buffer.ptr - len(achieved_goals)
    replay_buffer.size = replay_buffer.size - len(achieved_goals)

    if all(np.linalg.norm(achieved_goals[0] - g) < .05 for g in achieved_goals):
        if "usher" in config.ENV:
            #return False
            if random.random() > .1:
                return False
        #print("actor in place")# "\n".join([str(replay_buffer.action[replay_buffer.ptr + i]) for i in range(len(achieved_goals))]))

#        return False

    norm_ind = replay_buffer.ptr

    ep = deepcopy([ (
            replay_buffer.state[replay_buffer.ptr + i],
            replay_buffer.action[replay_buffer.ptr + i],
            replay_buffer.next_state[replay_buffer.ptr + i],
            replay_buffer.reward[replay_buffer.ptr + i],
            replay_buffer.discount[replay_buffer.ptr + i],
            replay_buffer.goals[replay_buffer.ptr + i],
            replay_buffer.lp[replay_buffer.ptr + i],
            replay_buffer.ret[replay_buffer.ptr + i],
            replay_buffer.oracle[replay_buffer.ptr + i],
            ) for i in range(len(achieved_goals)) ])

    for _ in range(config.HER_PER_EP):
        ep_ = []
        for j, e in enumerate(ep[:-1]):
            s, a, n, r, d, g, p, q, o = deepcopy(e)
            gi = j + random.randint(0, len(achieved_goals[j:][:n_future])-2)

            if random.random() < (dummy_filter if (np.linalg.norm(achieved_goals[0] - achieved_goals[gi]) < .05) else 0.):
                continue

            #assert all(achieved_goals[j] == n[:config.GOAL_SIZE]), "A)failed with {} + {} [{}][{}] <{}>".format(
            #    j, len(ep), achieved_goals[j], n[:config.GOAL_SIZE], np.linalg.norm(achieved_goals[j] - n[:config.GOAL_SIZE])
            #    )

            r = -1. + (np.linalg.norm(achieved_goals[j] - achieved_goals[gi]) < .05)# kl recalculated in replay buffer learning
            if -1 == r and random.random() < fail_filter:
                continue

            s[-config.GOAL_SIZE:] = deepcopy(achieved_goals[gi])
            n[-config.GOAL_SIZE:] = deepcopy(achieved_goals[gi])#+1])

#            if kl(e[-1], s) > .05 and random.random() < kl_filter:
#                continue

            #g = ep[gi+1][-4] #mu
            #p = ep[gi+1][-3] #scale
            total += (-1 != r)
            replay_buffer.add(s, a, n, r, d, g, p, q, o)


        for e in ep:
            if random.random() < her_ratio:
                continue
            replay_buffer.add(*deepcopy(e))

        #print("--> HT", replay_buffer.ptr, total)

#        for e in ep:
#            if random.random() < her_ratio:
#                continue
#            replay_buffer.add(*e)

    #assert all(replay_buffer.not_done[:replay_buffer.size])
    #print("\n diff", counter, norm_ind, replay_buffer.ptr, replay_buffer.ptr-norm_ind, sum(0. == replay_buffer.reward[norm_ind:replay_buffer.ptr]))

    #print("HER TOTAL", replay_buffer.ptr)
    if len(replay_buffer.state[norm_ind:replay_buffer.ptr]) > 1:# edge of buffer
        replay_buffer.normalize_state(replay_buffer.state[norm_ind:replay_buffer.ptr], update=True)

    return True

import random
from open_gym import make_env

file_name = f"{config.ENV}_{config.SEED}"
print("---------------------------------------")
print(f"Policy: , Env: {config.ENV}, Seed: {config.SEED}")
print("---------------------------------------")

env = make_env(config.ENV, render=False, colab=True)
eval_env = make_env(config.ENV, render=True, colab=True)

# Set seeds
env.seed(config.SEED)
env.action_space.seed(config.SEED)
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)

state_dim = env.state_size()
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

kwargs = { # let it default for td3 and td3+bc
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": config.DISCOUNT,
        "tau": config.TAU,
}
policy = TD3_BC.TD3_BC(**kwargs)
    # Initialize policy
#    policy = TD3.TD3(**kwargs)
#    policy = OurDDPG.DDPG(**kwargs)

import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

import math

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=1.):#math.e):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            layer_init(nn.Linear(num_inputs - config.GOAL0_SIZE, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh() if not config.ELU else nn.ELU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh() if not config.ELU else nn.ELU(),
            layer_init(nn.Linear(hidden_size, num_outputs), std=0.001),
        )

        #self.apply(init_weights)

        self.log_std = nn.Parameter(std + torch.zeros(1, num_outputs))
        self.sigma = lambda x: (-(1. + torch.relu(x)))# if config.PPO_NORM_IN else x

        self.mu = lambda x: torch.tanh(x) if config.PPO_NORM_IN else x

# adding gradients from control here
        self.add_module("critic", policy.critic)
        self.add_module("control", policy.actor)

    @torch.no_grad()
    def get_goal(self, x, detach=True, target=True):
        with torch.no_grad():
            mu    = self.mu(self.actor(x[:, config.GOAL0_SIZE:]))
            std   = self.sigma(self.log_std).exp().clamp(-.1, .1).expand_as(mu)
            dist  = Normal(mu, std)
            return dist.sample()

    @torch.no_grad()
    def get_goal_dist(self, x, detach=True, target=True):
        with torch.no_grad():
            mu    = self.mu(self.actor(x[:, config.GOAL0_SIZE:]))
            std   = self.sigma(self.log_std).exp().clamp(-.1, .1).expand_as(mu)
            dist  = Normal(mu, std)
            return dist

    @torch.no_grad()
    def values(self, states, actions, goals):
        with torch.no_grad():
            return torch.min(*policy.critic_target(states, actions, goals))

    def forward(self, x, detach=True, target=True, exploit=True):
        if False:#target:
            with torch.no_grad():
                mu    = self.mu(self.actor(x[:, config.GOAL0_SIZE:]))
                std   = self.sigma(self.log_std).exp().clamp(-.1, .1).expand_as(mu)
                dist  = Normal(mu, std)
                goal = dist.sample()
        else:
            mu    = self.mu(self.actor(x[:, config.GOAL0_SIZE:]))
            std   = self.sigma(self.log_std).exp().clamp(-.1, .1).expand_as(mu)
            dist  = Normal(mu, std)
            goal = dist.sample()

        if detach:
            with torch.no_grad():
#                #action = policy.actor(x)#torch.cat([x[:, :-config.GOAL0_SIZE], goal], 1))
#                action = policy.actor_target(torch.cat([x[:, :-config.GOAL0_SIZE], goal], 1))
#                #action = policy.actor(torch.cat([x[:, :-config.GOAL0_SIZE], goal], 1))
                #action = policy.actor_target(x)
                action = policy.select_action(torch.cat([x[:, :-config.GOAL0_SIZE], goal], 1), train_mode=not exploit).view(len(x), -1)
        else:
#            #action = policy.actor(x)#torch.cat([x[:, :-config.GOAL0_SIZE], goal], 1))
#            action = policy.actor_target(torch.cat([x[:, :-config.GOAL0_SIZE], goal], 1))
#            #action = policy.actor(torch.cat([x[:, :-config.GOAL0_SIZE], goal], 1))
                action = policy.actor_target(torch.cat([x[:, :-config.GOAL0_SIZE], goal], 1))

        if target:
            target_Q1, target_Q2 = policy.critic_target(x, action, goal)
            value = torch.min(target_Q1, target_Q2)
        else:
            #value = policy.critic.Q2(x, action)
            value = policy.critic.Q1(x, action, goal)

        return action, goal, dist, value

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = list(values) + [next_value]

    gae = 0
    returns = []
    discount = [gamma]
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        discount.insert(0, discount[0] * gamma * tau)
        returns.insert(0, gae + values[step])
    return returns, discount[1:]

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, gae, discount, open_state):
    yield -1, states, actions, log_probs, returns, gae, discount, open_state

    #print("PPO ITER", batch_size, mini_batch_size, torch.cat([p.view(-1) for p in policy.actor.parameters()]).sum())

    batch_size = states.size(0)
    for i in reversed(range(batch_size // mini_batch_size)):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield i, states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], gae[rand_ids, :].detach(), discount[rand_ids, :], open_state[rand_ids, :]

def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, gaes, discounts, open_states, callback):
    #return
    clip_param=0.2
    for epoch in range(ppo_epochs):
        #with torch.no_grad():#if True:#
        grads_pos = False#True#
        value = model(states, target=True, detach=grads_pos)[-1]
        open_value = model(open_states, target=True, detach=not grads_pos)[-1]
        advantages = gaes + discounts * open_value - value # we using target critic not part of optimizer reach

        for ei, state, action, old_log_prob, return_, adv, discount, open_state in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages, discounts, open_states):
#            if -1 != ei: adv = adv.detach()
            #advantages = TD3_BC.adv_norm(advantages, update=False)
            _, _, dist, value = model(state, target=False, detach=True)
            new_log_prob = dist.log_prob(action)

            ratio = (new_log_prob - old_log_prob).exp().mean(1, keepdim=True)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv

            if random.random() < .01: print(new_log_prob.mean(), old_log_prob.mean(), ratio.mean())

            if -1 == ei:
                actor_loss  = -torch.min(surr1, surr2).mean()
            else:
                actor_loss  = -torch.min(surr1, surr2).mean(0)[1]

            if config.SEPERATE_CRITICS:
                critic_loss = (return_ - value).pow(2).mean(0)[1]
            else:
                critic_loss = (return_ - value).pow(2).mean()

            entropy = dist.entropy().mean()

            #print(f"retain_graph={0==ei}, with {ei}")
            optimizer.zero_grad()
            (.5 * critic_loss + actor_loss - .001 * entropy).backward()#retain_graph=states.size(0)//mini_batch_size-1!=ei)
            #(.5 * critic_loss + actor_loss + .9 * (new_log_prob - discounts * old_log_prob).clip(-1, 0)).mean().backward()#retain_graph=states.size(0)//mini_batch_size-1!=ei)
            optimizer.step()

        callback(epoch)
#Hyper params:
hidden_size      = 256
lr               = config.LRPPO#3e-5
mini_batch_size  = config.MINI_BATCH_SIZE
ppo_epochs       = config.PPO_EPOCHS_PER_UPDATE

def ap():
    yield model.log_std
    for p in model.actor.parameters():
        yield p
    for p in policy.actor.parameters():
        yield p
    if not config.PPO_TRAIN_ACTOR:
        return
    for p in policy.actor_target.parameters():
        yield p
def total():
    for p in ap():
        yield p
    if not config.PPO_TRAIN_CRITIC:
        return
    for p in policy.critic.parameters():
        yield p

model = ActorCritic(state_dim, config.GOAL1_SIZE, hidden_size).to(device)
q_optimizer = optim.Adam(policy.critic.parameters(), lr=lr)
pi_optimizer = optim.Adam(ap(), lr=lr)
optimizer = optim.Adam(total(), lr=lr, eps=1e-5)

policy.set_opt(model)

replay_buffer_critic = utils.ReplayBuffer(state_norm, state_dim, action_dim)
replay_buffer_bc = utils.ReplayBuffer(state_norm, state_dim, action_dim)
replay_buffer_td3 = utils.ReplayBuffer(state_norm, state_dim, action_dim)

replay_buffer_ppo = utils.ReplayBuffer(state_norm, state_dim, action_dim)#, max_size = (config.STEPS_PER_EPOCH - config.PPO_DELAY_LEARN) // (1 if not config.HRL else ((1 if config.PPO_GAE_N != 10 else 2) * 50 // config.PPO_GAE_N)))

replay_buffer_fgae = utils.ReplayBuffer(state_norm, state_dim, action_dim)#, max_size = 50 * config.HER_PER_EP * 40)#5 * config.UPDATE_EVERY * config.HER_PER_EP * 100 // 50)

rew = lambda r: (r + config.REW_DELTA) * config.REW_SCALE
kls = lambda r: (r + config.KL_DELTA) * config.KL_SCALE

@torch.no_grad()
def floating_gae(k_step, exp,
                sar_buf, achieved_goals, oracles,
                policy, model, norm_state,
                next_state,
                dummy_filter=.97, her_ratio=.8, fail_filter=.8,
                do_her=True,
                ):

    if all(np.linalg.norm(achieved_goals[0] - g) < .05 for g in achieved_goals):
        if "usher" in config.ENV:
            #return False
            if random.random() > .33:
                return 0

    is_ppo = not do_her and 0. == her_ratio#PPO

    count = 0
    for jj in range(k_step, len(achieved_goals))[::(k_step if config.HRL else 1)]:

        target_i = jj - (k_step if is_ppo else random.randint(1, k_step))
        j = jj if is_ppo else random.randint(target_i + 1, jj)

        states = [ deepcopy(sar[0].reshape(-1)) for sar in (sar_buf + [next_state])[target_i:j+1] ]
        actions = np.vstack([sar[1] for sar in sar_buf[target_i:j]])

        for _ in range(config.ACHIEVED_PUSH_N):
            if do_her and random.random() < her_ratio:#(her_ratio if 0. != her_ratio else config.PPO_HER_RATIO):#max(.3, her_ratio):
                end = len(achieved_goals)
                if random.random() < config.LOOKAHEAD_1:
                    end = target_i + 1

                elif random.random() < config.LOOKAHEAD_K:
                    end = target_i + k_step

                goal = random.choice(achieved_goals[max(0, target_i-2):end])
#                goal = achieved_goals[target_i+1]
            else: goal = states[0][-config.GOAL0_SIZE:]

#            if 1 != k_step: goal = states[0][-3:]
#            else: goal = random.choice(achieved_goals[target_i:])

            direct_rewards = [-1. * (np.linalg.norm(goal - achieved_goals[k]) > .05) for k in range(target_i, j)]

            local_sum = sum(np.hstack(direct_rewards) == 0.)

            if local_sum > 0:
                break

        if not is_ppo and random.random() < (dummy_filter if (np.linalg.norm(achieved_goals[0] - goal) < .05) else 0.):
            continue

        count += local_sum > 0

        states = np.vstack([
            np.concatenate(
                [s[:-3], deepcopy(goal)]
            ) for s in states])

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            norms = norm_state(states)
            dist = model.get_goal_dist(norms)
            goals = dist.sample()
            next_action = policy.actor_target(torch.cat([
                                                         norms[-1].view(1, -1)[:, :-config.GOAL0_SIZE],
                                                         goals[-1].view(1, -1)
                                                ], 1)).numpy()
            actions = np.concatenate([actions, next_action])
            q_min = torch.min(*policy.critic_target(
                norms,  torch.from_numpy(actions), goals)
                ).cpu().numpy()

        oracle_states = np.concatenate(oracles[target_i:][:len(q_min)])
        do = model.get_goal_dist(norm_state(oracle_states))
        kl = torch.distributions.kl_divergence(dist, do)

        gae, discounts = compute_gae(
            q_min[-1],
            [np.array([kls(-1. + (k.numpy().mean() < config.KL_MIN)), rew(-1. + (0. == r))]) for k, r in zip(kl, direct_rewards)],
#            direct_rewards,
            [0 != i for i in reversed(range(len(direct_rewards)))],
            q_min[:-1],
            gamma=config.DISCOUNT)

        returns = np.vstack(discounts) * q_min[-1] + np.vstack(gae)
        for i, (s, a, r, d, o, ret) in enumerate(zip(states[:-1], actions[:-1], gae, discounts, oracle_states, returns)):
            if 0. != direct_rewards[i] and random.random() < fail_filter:
                continue

            assert 1 != k_step or all(sar_buf[target_i+i][3].reshape(-1)[:-3] == states[-1].reshape(-1)[:-3]), f"!!inconsistency from her {sar_buf[target_i+i][3]} vs {states[-1]}"

            if config.HRL and i and 0. == her_ratio and k_step * 9 <= len(achieved_goals):
                break

            exp.add(s, a, states[-1], r, d, sar_buf[target_i+i][-4], sar_buf[target_i+i][-3], ret, o)


    if is_ppo:
        return count # not for PPO!

    for i, (sarn, o) in enumerate(zip(sar_buf[:-1], oracles[:-1])):
        if random.random() < her_ratio:
            continue
        s, a, r, n = sarn[:4]
        exp.add(s, a, n, np.array([kls(-1. + (sarn[-1].mean() < config.KL_MIN)), rew(-1. + (0. == r))]), config.DISCOUNT, sarn[-4], sarn[-3], np.zeros(2), o)

    #print("TOTAL FLOATS ", exp.ptr, count)

    return count

#%%time

def normalize(x, update=False):
    if not config.NORMALIZE:
        return torch.from_numpy(x).float()
    return state_norm(torch.from_numpy(x).float(), update)
load_state = lambda obs: obs["observation"].reshape(1,-1)


print("---------------------------------------")
print(f"Policy DUH: , Env: {config.ENV}, Seed: {config.SEED}, Observation shape: {state_dim}")
print("---------------------------------------")

counter = 0
score = -100
hindsight = 0

done = True
achieved_goals = []

import time
tm = time.time()
ft = time.time()

t = 0
norm_ind = 0
add_prev_exp = 0
total_steps = config.STEPS_PER_EPOCH * config.EPOCHS

n_ep_steps = 100#48 if not config.HRL or 10 != config.PPO_GAE_N else 100

normalized_once = False

update_every = config.UPDATE_EVERY
steps_per_epoch = config.STEPS_PER_EPOCH
eval_freq = config.EVAL_FREQ

n_eps = 0
n_epochs = 0

#eval_policy(n_epochs, policy, eval_env, config.SEED, normalize, model)

while n_epochs < config.EPOCHS:
#while t < total_steps:
    counter += 1

    if done:
        n_eps += 1
        add_prev_exp *= 0
        if len(achieved_goals):
            if config.NORMALIZE:
                normalize(np.stack(s[0] for s in sarsgpv).reshape(len(sarsgpv), -1), True)
            #out = """

            norm_ind = replay_buffer_fgae.ptr if config.TD3_GAE else replay_buffer_critic.ptr
            for z in range(config.HER_PER_EP):
                if config.TD3_GAE:
                    floating_gae(config.TD3_GAE_N, replay_buffer_fgae,
                            sarsgpv, achieved_goals, oracles,
                            policy, model, normalize,
                            next_state)
                    if 0 == norm_ind and replay_buffer_fgae.ptr != 0 and not z:
                        replay_buffer_fgae.max_size = 40 * config.HER_PER_EP * replay_buffer_fgae.size
                        print(f"[FGAE BUF STAT] {replay_buffer_fgae.ptr} : # of exp added per ep => { replay_buffer_fgae.max_size // (replay_buffer_fgae.size * config.HER_PER_EP) } # of total episodes active in buffer")
                floating_gae(1, replay_buffer_critic,
                            sarsgpv, achieved_goals, oracles,
                            policy, model, normalize,
                            next_state)
                for z in range(2): floating_gae(config.TD3_GAE_N, replay_buffer_critic,
                            sarsgpv, achieved_goals, oracles,
                            policy, model, normalize,
                            next_state, do_her=False, her_ratio=1., fail_filter=.0, dummy_filter=.33)

            out = """if norm_ind < replay_buffer_critic.ptr:
                if config.TD3_GAE:
                    normalize(replay_buffer_fgae.state[norm_ind:replay_buffer_fgae.ptr], update=True)
                else:
                    normalize(replay_buffer_critic.state[norm_ind:replay_buffer_critic.ptr], update=True)
                    """
            #assert False
            #"""

            if norm_ind != replay_buffer_critic.ptr:
                ni2 = replay_buffer_ppo.ptr

                ok = floating_gae(config.PPO_GAE_N, replay_buffer_ppo,
                        sarsgpv, achieved_goals, oracles,
                        policy, model, normalize,
                        next_state, do_her=config.PPO_HER,
                        dummy_filter=0., her_ratio=0., fail_filter=.0)

                ni2 = replay_buffer_ppo.ptr-ni2
                if ok:
                    print("??", n_eps, ni2, ok, replay_buffer_ppo.ptr, t, steps_per_epoch)

                if 0 == norm_ind:
                    replay_buffer_ppo.max_size = 3 * config.STEPS_PER_EPOCH // 2#((config.STEPS_PER_EPOCH - config.PPO_DELAY_LEARN) // n_ep_steps) * replay_buffer_ppo.size
                    print(f"[PPO BUF STAT] {replay_buffer_ppo.ptr} : # of exp added per ep => { replay_buffer_ppo.max_size // replay_buffer_ppo.size } # of total episodes active in buffer")

                if ok > 0:
                    add_prev_exp += 1

                if ni2 >= 0 and ni2 <= len(achieved_goals):
                    t += ni2

        oracles = []
        sarsgpv = []
#        add_prev_exp = ok > 0 and (norm_ind != (replay_buffer_fgae.ptr if config.TD3_GAE else replay_buffer_critic.ptr))
        achieved_goals = []

        state = load_state(env.reset())

    #t += add_prev_exp

    with torch.no_grad():
#        action, goal, dist, value = model(
#            normalize(state),
#            detach=True, exploit=False)

        norms = normalize(state)
        dist = model.get_goal_dist(norms)
        if not config.HRL or 0 == len(achieved_goals) % config.PPO_GAE_N:
            goal = dist.sample()
        action = policy.select_action(torch.cat([norms[:, :-config.GOAL0_SIZE], goal], 1), train_mode=True).view(1, -1)
        value = model.values(norms, action, goal)

    action = action.view(-1).cpu().detach().numpy()
    if t < config.START_STEPS:# or (-1. == reward and random.random() < mc_w / 10.):
        action = env.action_space.sample()

    observation, reward, done, _ = env.step(action)
    if len(achieved_goals) == n_ep_steps:
        done = True

    next_state = load_state(observation)
    achieved_goals.append(observation["achieved_goal"])

    #print("*", dist.log_prob(goal))
    oracle_state = np.hstack([state.reshape(-1)[:-config.GOAL0_SIZE], achieved_goals[-1]]).reshape(1, -1)
    ns = normalize(oracle_state)
    gd = model.get_goal_dist(ns)
    kl = torch.distributions.kl_divergence(dist, gd).numpy()
    if kl.mean() < config.KL_MIN and random.random() < config.HINDSIGHT_ACTION:
        goal = gd.sample()
        hindsight += 1
        #dist = gd

    oracles.append(oracle_state)

    sarsgpv.append([
        state, action, reward, next_state,
        goal.numpy(), dist.log_prob(goal).numpy(), value.numpy(),
        kl])
#        gd.log_prob(goal).numpy()])

    state = next_state

    dbg_step = 3000
    if 0 == counter % dbg_step:
        print(f"[{(time.time()-tm)//1}s / {(time.time()-ft)//60}m] stats : total:{counter}; good:{t}; hindsight:{hindsight} [{'%.2f'%(hindsight * 100 / dbg_step)}%] PPO buffer:{replay_buffer_ppo.ptr}")
        tm = time.time()
        hindsight = 0

    def train_control(epoch):
        get_goal = lambda s: model.get_goal_dist(s)

        class MIX:
            def sample(self, bs):
                #return replay_buffer_fgae.sample(bs)
                #return replay_buffer_critic.sample(bs)

                b1 = replay_buffer_critic.sample(bs * 3//8)
                b2 = replay_buffer_fgae.sample(bs * 5//8)
                return [ torch.cat([a, b]) for a, b in zip(b1, b2) ]

        exp = MIX()

        if config.TD3_GAE:
            policy.train_critic(exp, get_goal=get_goal, batch_size=config.BATCH_SIZE, use_norm=False, do_update=False)#
        else:
            policy.train_critic(replay_buffer_critic, get_goal=get_goal, batch_size=config.BATCH_SIZE, use_norm=False, do_update=False)#

        if config.CRITIC_EMPHATIZE_RECENT and config.TD3_GAE:
            policy.train_actor(exp, use_bc=False, get_goal=get_goal, batch_size=config.BATCH_SIZE)
        else:
            policy.train_actor(replay_buffer_critic, use_bc=False, get_goal=get_goal, batch_size=config.BATCH_SIZE)

# works GooD!!
#        policy.train_critic(replay_buffer_critic, get_goal=get_goal, batch_size=config.BATCH_SIZE, use_norm=False, do_update=False)
#        policy.train_actor(replay_buffer_critic, use_bc=False, get_goal=get_goal, batch_size=config.BATCH_SIZE)

    if t > config.UPDATE_AFTER and t > update_every:
        update_every = t+config.UPDATE_EVERY
        for e in range(config.UPDATE_COUNT):
            train_control(e)
            policy.polyak_actor(.005)
        policy.polyak_critic(.05)


    if t > steps_per_epoch:
        steps_per_epoch = t+config.STEPS_PER_EPOCH
#        if not normalized_once:
#            normalize(replay_buffer_ppo.state, update=True)# TODO TEST, last test does not cover this!!!
#        normalized_once = True
        
        def callback(_e):
            if config.RPOLYAK:
                policy.rpolyak(.05)
            else:
                policy.polyak(.05)

        print("DO PPO", replay_buffer_ppo.ptr, t)
        states, _, open_state, gae, discount, actions, log_probs, returns, _ = replay_buffer_ppo.sample_full()
        ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, gae, discount, open_state, callback)

    sys.stdout.flush()

    # Evaluate episode
    if t > eval_freq:
        eval_freq = t+config.EVAL_FREQ
        print(f"Epochs : {(t + 1) / config.EVAL_FREQ} [#timestamps : {counter}]")
        score = eval_policy(n_epochs, policy, eval_env, config.SEED, normalize, model)

        n_epochs += 1
        
    if score > (-40. if "usher" in config.ENV else -10.):
        break

print("FINALE", score)
sys.stdout.flush()
