import gym
import numpy as np
from gym import spaces


class FlappyEnv(gym.Env):
    def __init__(self, flappy_game):
        super(FlappyEnv, self).__init__()
        self.state_dim = 10
        self.flappy_game = flappy_game

        # 定义动作空间（0: 不跳，1: 跳）
        self.action_space = spaces.Discrete(2)

        # 定义状态空间，可以根据你的游戏状态定义
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

    def reset(self):
        # 重置游戏状态
        self.flappy_game.start()
        return self.get_state()

    def step(self, action):
        # 执行动作
        if action == 1:  # 跳
            self.flappy_game.player.flap()

        # 更新游戏状态
        self.flappy_game.play()

        # 获取新的状态、奖励和是否结束
        state = self.get_state()
        reward = self.get_reward()
        done = self.is_game_over()

        return state, reward, done, {}

    def get_state(self):
        # 返回当前状态的表示
        return np.array([self.flappy_game.player.y, self.flappy_game.score.current])

    def get_reward(self):
        # 定义奖励机制
        if self.flappy_game.player.collided(self.flappy_game.pipes, self.flappy_game.floor):
            return -1  # 碰撞时惩罚
        return 1  # 成功穿过管道时奖励

    def is_game_over(self):
        return self.flappy_game.player.collided(self.flappy_game.pipes, self.flappy_game.floor)
