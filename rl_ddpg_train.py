import asyncio
import os
import time

from flap_bird_env import FlappyEnv
from model import DDPG
from src.flappy import Flappy


# 定義 Actor 網絡

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
    num_steps = 3000
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for num_step in range(num_steps):
            action = ddpg.select_action(state)
            next_state, reward, done, _ = env.step(action)
            ddpg.add_experience((state, action, reward, next_state, float(done)))
            ddpg.update()
            state = next_state
            total_reward += reward
            time.sleep(0.1)
            # if done:
            #     # 开始新的博弈
            #     env.flappy_game.play()
            print(f'Episode {episode}, num_step {num_step} ,Total Reward: {total_reward}')

        # 每隔一定回合保存一次模型
        if episode % 100 == 0:
            ddpg.save(save_model_path)
            print("Saved model to", save_model_path)

if __name__ == '__main__':
    # 启动 asyncio 事件循环
    asyncio.run(train(load_model=True))


