import asyncio
import os
import time

import numpy as np

from flap_bird_env import FlappyEnv
from model import DDPG
from src.flappy import Flappy

# 訓練循環
def train(load_model=False):
    env = FlappyEnv(flappy_game=Flappy())
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    # 输入状态空间和动作空间维度
    ddpg = DDPG(state_dim, action_dim)

    # 加载模型
    save_model_path = 'ddpg_model.pth'
    if load_model and os.path.exists(save_model_path):
        ddpg.load(save_model_path)
        print("Loaded model from", save_model_path)

    num_episodes = 10000
    state = env.reset()

    for episode in range(num_episodes):
        # 计算总的奖励值
        total_reward = 0

        for num_step in range(1000):
            action_values = ddpg.select_action(state)
            probabilities = np.exp(action_values) / np.sum(np.exp(action_values))
            action = np.random.choice(len(action_values), p=probabilities)

            next_state, reward, done, _ = env.step(action, num_step)
            # 增加到经验区
            ddpg.add_experience((state, action_values, reward, next_state, float(done)))
            ddpg.update()
            state = next_state
            total_reward += reward

            print(f'Episode {episode}, num_step {num_step} , '
                  f'action_values: {action_values}, '
                  f'action: {action}, '
                  f'state: {next_state}, '
                  f'current Reward: {reward}')

            if done:
                break

        # 每epoch保存一次模型
        if episode % 100 == 0:
            ddpg.save(save_model_path)
            print(f"Episode {episode}, total_step: {num_step}, saved model to", save_model_path)

if __name__ == '__main__':
    # 启动 asyncio 事件循环
    asyncio.run(train(load_model=True))


