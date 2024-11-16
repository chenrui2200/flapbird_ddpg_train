import asyncio
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from gym import spaces

from flap_bird_env import FlappyEnv
from src.flappy import Flappy


# 定義 Actor 網絡
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


# 定義 Critic 網絡
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.relu(self.fc1(torch.cat([state, action], 1)))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# DDPG 演算法
class DDPG:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        # 複製權重
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.replay_buffer = []
        self.max_buffer_size = 100000
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.005

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        return self.actor(state).detach().numpy()[0]

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # 隨機選擇一批樣本
        batch = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        state, action, reward, next_state, done = zip(*[self.replay_buffer[i] for i in batch])

        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)

        # 更新 Critic
        target_action = self.actor_target(next_state)
        target_q = self.critic_target(next_state, target_action)
        expected_q = reward + (1 - done) * self.gamma * target_q
        critic_loss = nn.MSELoss()(self.critic(state, action), expected_q.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新 Actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新 Target 網絡
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def add_experience(self, experience):
        if len(self.replay_buffer) >= self.max_buffer_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append(experience)

    def save(self, filename):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])


# 訓練循環
def train(load_model=False, save_model_path='ddpg_model.pth'):
    env = FlappyEnv(flappy_game=Flappy())
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    ddpg = DDPG(state_dim, action_dim)

    # 加载模型
    if load_model and os.path.exists(save_model_path):
        ddpg.load(save_model_path)
        print("Loaded model from", save_model_path)

    num_episodes = 1000
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = ddpg.select_action(state)
            next_state, reward, done, _ = env.step(action)
            ddpg.add_experience((state, action, reward, next_state, float(done)))
            ddpg.update()
            state = next_state
            total_reward += reward

            if done:
                break

        print(f'Episode {episode}, Total Reward: {total_reward}')

        # 每隔一定回合保存一次模型
        if episode % 100 == 0:
            ddpg.save(save_model_path)
            print("Saved model to", save_model_path)

if __name__ == '__main__':
    # 启动 asyncio 事件循环
    asyncio.run(train(load_model=True))


