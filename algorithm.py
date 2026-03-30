import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from env import (
    NUM_UAV, MAX_AGV_SELECT, CONTINUOUS_ACTION_DIM, DISCRETE_ACTION_DIM
)
import os

# 设置随机种子
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

GUMBEL_TAU_INIT = 1.0
GUMBEL_TAU_MIN = 0.01
GUMBEL_TAU_DECAY = 0.995

def flatten_obs(obs):
    uav_info = [obs['uav_id']] + obs['uav_position'].tolist() + [obs['compute_available']]
    agv_info = []
    for agv in obs['observed_agvs']:
        agv_info += [agv['id']] + agv['position'].tolist() + [agv['task_size']]
    max_agvs = 10
    agv_info = agv_info[:max_agvs * 4]
    agv_info += [0] * (max_agvs * 4 - len(agv_info))
    return uav_info + agv_info

class ActorNetwork(nn.Module):
    def __init__(self, input_dim, continuous_action_dim, discrete_action_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.continuous_head = nn.Linear(128, continuous_action_dim)
        self.continuous_head.weight.data.uniform_(-3e-3, 3e-3)

        self.discrete_head = nn.Linear(128, discrete_action_dim)
        self.discrete_head.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x, tau=1.0, training=True):
        batch_size = x.size(0)
        if batch_size == 1:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        else:
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))

        continuous_action = torch.tanh(self.continuous_head(x))
        discrete_logits = self.discrete_head(x)

        if training:
            gumbel_noise = self._sample_gumbel(discrete_logits.shape, discrete_logits.device)
            y = discrete_logits + gumbel_noise
            discrete_action = F.softmax(y / tau, dim=-1)
        else:
            discrete_action = torch.argmax(discrete_logits, dim=-1)
            discrete_action = F.one_hot(discrete_action, num_classes=discrete_logits.shape[-1]).float()

        return continuous_action, discrete_action, discrete_logits

    def _sample_gumbel(self, shape, device, eps=1e-20):
        u = torch.rand(shape, device=device)
        return -torch.log(-torch.log(u + eps) + eps)

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, num_agents):
        super(CriticNetwork, self).__init__()
        total_continuous_action = CONTINUOUS_ACTION_DIM * num_agents
        total_discrete_action = DISCRETE_ACTION_DIM * num_agents

        self.state_fc1 = nn.Linear(state_dim, 256)
        self.state_bn1 = nn.BatchNorm1d(256)
        self.state_fc2 = nn.Linear(256, 128)

        self.continuous_action_fc = nn.Linear(total_continuous_action, 128)
        self.discrete_action_fc = nn.Linear(total_discrete_action, 128)

        self.combine_fc1 = nn.Linear(128 + 128 + 128, 128)
        self.combine_fc2 = nn.Linear(128, 64)
        self.layer_norm = nn.LayerNorm(64)
        self.out = nn.Linear(64, 1)
        self.out.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, continuous_actions, discrete_actions):
        batch_size = state.size(0)
        if batch_size == 1:
            s = F.relu(self.state_fc1(state))
            s = F.relu(self.state_fc2(s))
        else:
            s = F.relu(self.state_bn1(self.state_fc1(state)))
            s = F.relu(self.state_fc2(s))

        a_cont = F.relu(self.continuous_action_fc(continuous_actions))
        a_disc = F.relu(self.discrete_action_fc(discrete_actions))

        x = torch.cat([s, a_cont, a_disc], dim=1)
        x = F.relu(self.combine_fc1(x))
        x = F.relu(self.combine_fc2(x))
        x = self.layer_norm(x) if batch_size > 1 else x
        return self.out(x)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def push(self, state, actions, rewards, next_state, done):
        max_p = self.max_priority ** self.alpha
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, actions, rewards, next_state, done)
        self.priorities[self.position] = max_p
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

        probs = priorities / (np.sum(priorities) + 1e-6)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights)

        self.beta = min(1.0, self.beta + self.beta_increment)
        samples = [self.buffer[idx] for idx in indices]
        state, actions, rewards, next_state, done = zip(*samples)
        return state, actions, rewards, next_state, done, weights, indices

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            clipped_error = np.clip(error, 1e-4, 10.0)
            self.priorities[idx] = (clipped_error + 1e-5) ** self.alpha
            self.max_priority = max(self.max_priority, error + 1e-5)

    def __len__(self):
        return len(self.buffer)

class MADDPG:
    def __init__(self, num_agents, obs_dims):
        self.num_agents = num_agents
        self.obs_dims = obs_dims

        self.actors = []
        self.critics = []
        self.target_actors = []
        self.target_critics = []
        self.actor_optimizers = []
        self.critic_optimizers = []

        self.gumbel_tau = GUMBEL_TAU_INIT
        self.exploration_noise = 2.0
        self.noise_decay = 0.97
        self.min_noise = 0.02

        self.curriculum_phase = 0
        self.curriculum_thresholds = [100, 200, 300, 400]
        self.update_iterations = 1
        self.grad_clip_value = 0.5
        self.initial_reward_factor = 0.2
        self.reward_factor = self.initial_reward_factor
        self.reward_factor_growth = 0.005

        for i in range(num_agents):
            actor = ActorNetwork(obs_dims[i], CONTINUOUS_ACTION_DIM, DISCRETE_ACTION_DIM)
            target_actor = ActorNetwork(obs_dims[i], CONTINUOUS_ACTION_DIM, DISCRETE_ACTION_DIM)
            target_actor.load_state_dict(actor.state_dict())

            critic = CriticNetwork(obs_dims[i], num_agents)
            target_critic = CriticNetwork(obs_dims[i], num_agents)
            target_critic.load_state_dict(critic.state_dict())

            self.actors.append(actor)
            self.critics.append(critic)
            self.target_actors.append(target_actor)
            self.target_critics.append(target_critic)

            self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=1e-5, weight_decay=1e-5))
            self.critic_optimizers.append(optim.Adam(critic.parameters(), lr=5e-5, weight_decay=1e-5))

        self.memory = PrioritizedReplayBuffer(capacity=10000, alpha=0.6, beta=0.4)
        self.batch_size = 32
        self.gamma = 0.80
        self.tau = 0.001

        self.lr_scheduler_actors = []
        self.lr_scheduler_critics = []
        for i in range(num_agents):
            self.lr_scheduler_actors.append(
                optim.lr_scheduler.CosineAnnealingWarmRestarts(self.actor_optimizers[i], T_0=100, T_mult=1,
                                                               eta_min=5e-6))
            self.lr_scheduler_critics.append(
                optim.lr_scheduler.CosineAnnealingWarmRestarts(self.critic_optimizers[i], T_0=100, T_mult=1,
                                                               eta_min=2e-5))

        self.q_value_history = deque(maxlen=100)

    def select_action(self, observations, training=True):
        actions = []
        cont_actions_list = []
        disc_actions_list = []

        for i, obs in enumerate(observations):
            obs_tensor = torch.FloatTensor(flatten_obs(obs)).unsqueeze(0)

            if self.curriculum_phase == 0 and random.random() < 0.5 and training:
                cont_action = np.random.uniform(-1, 1, CONTINUOUS_ACTION_DIM)
                disc_action = np.random.uniform(0, 1, DISCRETE_ACTION_DIM)
                disc_action = disc_action / np.sum(disc_action)
            else:
                cont_action_t, disc_action_t, _ = self.actors[i](
                    obs_tensor, tau=self.gumbel_tau, training=training
                )
                cont_action = cont_action_t.detach().numpy()[0]
                disc_action = disc_action_t.detach().numpy()[0]

                if training:
                    noise_scale = self.exploration_noise * (1.5 if self.curriculum_phase == 0 else 1.0)
                    noise = np.random.normal(0, noise_scale, size=cont_action.shape)
                    cont_action = cont_action + noise
                    cont_action = np.clip(cont_action, -1, 1)

            cont_actions_list.append(cont_action)
            disc_actions_list.append(disc_action)
            actions.append((cont_action, disc_action))

        return actions, cont_actions_list, disc_actions_list

    def store_transition(self, state, actions, rewards, next_state, done):
        self.memory.push(state, actions, rewards, next_state, done)

    def update(self, batch):
        states, actions, rewards, next_states, dones, weights, indices = batch

        batch_states = []
        batch_cont_actions = []
        batch_disc_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []

        for _ in range(self.num_agents):
            batch_states.append([])
            batch_cont_actions.append([])
            batch_disc_actions.append([])
            batch_rewards.append([])
            batch_next_states.append([])
            batch_dones.append([])

        for i in range(len(states)):
            for agent_id in range(self.num_agents):
                batch_states[agent_id].append(flatten_obs(states[i][agent_id]))
                batch_cont_actions[agent_id].append(actions[i][agent_id][0])
                batch_disc_actions[agent_id].append(actions[i][agent_id][1])
                batch_rewards[agent_id].append(rewards[i][agent_id])
                batch_next_states[agent_id].append(flatten_obs(next_states[i][agent_id]))
                batch_dones[agent_id].append(dones[i])

        for agent_id in range(self.num_agents):
            batch_states[agent_id] = torch.FloatTensor(np.array(batch_states[agent_id]))
            batch_cont_actions[agent_id] = torch.FloatTensor(np.array(batch_cont_actions[agent_id]))
            batch_disc_actions[agent_id] = torch.FloatTensor(np.array(batch_disc_actions[agent_id]))
            batch_rewards[agent_id] = torch.FloatTensor(np.array(batch_rewards[agent_id])).unsqueeze(1)
            batch_next_states[agent_id] = torch.FloatTensor(np.array(batch_next_states[agent_id]))
            batch_dones[agent_id] = torch.FloatTensor(np.array(batch_dones[agent_id])).unsqueeze(1)

        all_cont_actions = torch.cat(batch_cont_actions, dim=1)
        all_disc_actions = torch.cat(batch_disc_actions, dim=1)

        errors = []
        q_values = []

        for agent_id in range(self.num_agents):
            with torch.no_grad():
                next_cont_list = []
                next_disc_list = []
                for i in range(self.num_agents):
                    c, d, _ = self.target_actors[i](batch_next_states[i], tau=self.gumbel_tau, training=True)
                    next_cont_list.append(c)
                    next_disc_list.append(d)

                next_all_cont = torch.cat(next_cont_list, dim=1)
                next_all_disc = torch.cat(next_disc_list, dim=1)

                target_q = batch_rewards[agent_id] + \
                           (1 - batch_dones[agent_id]) * self.gamma * \
                           self.target_critics[agent_id](batch_next_states[agent_id], next_all_cont, next_all_disc)

            current_q = self.critics[agent_id](batch_states[agent_id], all_cont_actions, all_disc_actions)
            q_values.append(current_q.mean().item())

            td_error = torch.abs(current_q - target_q).detach().numpy()
            errors.append(td_error)

            weights_tensor = torch.FloatTensor(weights).unsqueeze(1)
            critic_loss = (weights_tensor * F.mse_loss(current_q, target_q, reduction='none')).mean()

            self.critic_optimizers[agent_id].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[agent_id].parameters(), max_norm=self.grad_clip_value)
            self.critic_optimizers[agent_id].step()

            curr_cont_list = []
            curr_disc_list = []
            for i in range(self.num_agents):
                if i == agent_id:
                    c, d, _ = self.actors[i](batch_states[i], tau=self.gumbel_tau, training=True)
                    curr_cont_list.append(c)
                    curr_disc_list.append(d)
                else:
                    curr_cont_list.append(batch_cont_actions[i].detach())
                    curr_disc_list.append(batch_disc_actions[i].detach())

            curr_all_cont = torch.cat(curr_cont_list, dim=1)
            curr_all_disc = torch.cat(curr_disc_list, dim=1)

            actor_loss = -self.critics[agent_id](batch_states[agent_id], curr_all_cont, curr_all_disc).mean()

            self.actor_optimizers[agent_id].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), max_norm=self.grad_clip_value)
            self.actor_optimizers[agent_id].step()

            self.soft_update(self.target_actors[agent_id], self.actors[agent_id])
            self.soft_update(self.target_critics[agent_id], self.critics[agent_id])

        mean_errors = np.mean(np.concatenate(errors), axis=0)
        self.memory.update_priorities(indices, mean_errors)

        for i in range(self.num_agents):
            self.lr_scheduler_actors[i].step()
            self.lr_scheduler_critics[i].step()

        return q_values

    def soft_update(self, target_net, eval_net):
        for target_param, param in zip(target_net.parameters(), eval_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update_curriculum(self, episode):
        old_phase = self.curriculum_phase
        if episode > self.curriculum_thresholds[min(self.curriculum_phase, len(self.curriculum_thresholds) - 1)]:
            if self.curriculum_phase < len(self.curriculum_thresholds):
                self.curriculum_phase += 1

        if old_phase != self.curriculum_phase:
            pass # 静默更新

        if self.curriculum_phase == 1:
            self.batch_size = 64
            self.gamma = 0.85
            self.tau = 0.002
            self.update_iterations = 2
            self.grad_clip_value = 0.75
        elif self.curriculum_phase == 2:
            self.batch_size = 96
            self.gamma = 0.90
            self.tau = 0.005
            self.update_iterations = 3
            self.grad_clip_value = 1.0
        elif self.curriculum_phase == 3:
            self.batch_size = 128
            self.gamma = 0.95
            self.tau = 0.005
            self.update_iterations = 4
            self.grad_clip_value = 1.5

    def update_exploration_noise(self, episode):
        if episode < 50:
            self.exploration_noise = 2.0
        else:
            self.exploration_noise = max(self.min_noise, self.exploration_noise * self.noise_decay)
        if episode < 350:
            self.reward_factor = min(1.0, self.initial_reward_factor + episode * self.reward_factor_growth)

    def update_gumbel_tau(self):
        self.gumbel_tau = max(GUMBEL_TAU_MIN, self.gumbel_tau * GUMBEL_TAU_DECAY)

    def save_models(self, episode):
        save_path = f'models/episode_{episode}'
        os.makedirs(save_path, exist_ok=True)
        for i in range(self.num_agents):
            torch.save(self.actors[i].state_dict(), f'{save_path}/actor_{i}.pth')
            torch.save(self.critics[i].state_dict(), f'{save_path}/critic_{i}.pth')