import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.relu(self.fc1(torch.cat([state, action], 1)))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DDPG:
    def __init__(self, state_dim, action_dim):
        # 检查是否有可用的 GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print('device', self.device)

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        # 复制权重
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.replay_buffer = []
        self.max_buffer_size = 100000
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.005
        self.action_dim = action_dim
        self.noise = self.ou_noise()

    def ou_noise(self):
        # 初始化 Ornstein-Uhlenbeck 噪声
        self.mu = np.zeros(self.action_dim)
        self.theta = 0.15
        self.sigma = 0.2
        self.dt = 1e-2
        self.x_prev = np.zeros(self.action_dim)
        return self

    def get_ou_noise(self):
        # 生成 Ornstein-Uhlenbeck 噪声
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(
            self.dt) * np.random.randn(self.action_dim)
        self.x_prev = x
        return x

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy()[0]

        # 添加噪声
        noise = self.get_ou_noise()
        action = action + noise

        # 限制动作范围，例如在 [0, 1] 之间
        action = np.clip(action, 0, 1)
        return action

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # 随机选择一批样本
        batch = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        state, action, reward, next_state, done = zip(*[self.replay_buffer[i] for i in batch])

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

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

        # 更新 Target 网络
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