import asyncio
import os
import time

import numpy as np
from matplotlib import pyplot as plt

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
    state, _, _ = env.reset()
    rewards = []  # 用于记录每个回合的总奖励

    for episode in range(num_episodes):

        for num_step in range(1000):
            action_values = ddpg.select_action(state)
            next_state, reward, done, _ = env.step(action_values[0])
            # 增加到经验区
            ddpg.add_experience((state, action_values, reward, next_state, float(done)))
            ddpg.update()
            state = next_state

            if len(rewards) > 10000:
                rewards.pop(0)

            print(f'Episode {episode}, num_step {num_step}, '
                  f'action: {action_values}, '
                  f'state: {next_state}, '
                  f'current Reward: {reward}')

            if done:
                break

        # 每epoch保存一次模型
        if episode % 100 == 0:
            ddpg.save(save_model_path)
            print(f"Episode {episode}, total_step: {num_step}, saved model to", save_model_path)

    # 绘制奖励线图
    plt.plot(rewards)
    plt.title("Training Rewards Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig("training_rewards.png", format='png')

if __name__ == '__main__':
    # 启动 asyncio 事件循环
    asyncio.run(train(load_model=True))


