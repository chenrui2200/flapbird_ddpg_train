import gym
import numpy as np
from gym import spaces


class FlappyEnv(gym.Env):
    def __init__(self, flappy_game):
        super(FlappyEnv, self).__init__()
        # 我只关注笨鸟和它前面的管道口坐标的位置
        self.state_dim = 6
        self.flappy_game = flappy_game

        # 定义动作空间（0: 不跳，1: 跳）
        self.action_space = spaces.Discrete(1)

        # 定义状态空间，可以根据你的游戏状态定义
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

    def reset(self):
        # 重置游戏状态
        self.flappy_game.start()

        return self.get_state()

    def step(self, action):
        # 执行动作
        threshold = 0.5
        if action[0] > threshold and hasattr(self.flappy_game, 'player'):  # 跳
            self.flappy_game.player.flap()

        # 更新游戏状态
        self.flappy_game.play()

        # 获取新的状态、奖励和是否结束
        state = self.get_state()
        reward = self.get_reward()
        done = self.is_game_over()

        if done:
            reward -= 10

        return state, reward, done, {}

    def get_state(self):
        # 找到笨鸟前面的管道up和low各一个
        up_pipe, low_pipe = self.find_closest_pipes()
        # 返回当前状态的表示
        play_x, player_y = self.get_play_xy()
        return np.array([
            play_x,
            player_y,
            up_pipe.x if up_pipe else 0,
            up_pipe.y if up_pipe else 0,
            low_pipe.x if low_pipe else 0,
            low_pipe.y if low_pipe else 0
        ])

    def get_play_xy(self):
        if hasattr(self.flappy_game, 'player'):
            return self.flappy_game.player.x, self.flappy_game.player.y
        else:
            return 0, 0
    def find_closest_pipes(self):
        closest_pipes = None
        min_diff = float('inf')  # 初始化最低差值为正无穷

        # 遍历上下管道
        if hasattr(self.flappy_game, 'pipes'):
            for up_pipe, low_pipe in zip(self.flappy_game.pipes.upper, self.flappy_game.pipes.upper.lower):
                if up_pipe.x > self.flappy_game.player.x and low_pipe.x > self.flappy_game.player.x:
                    # 计算上管道和下管道的 x 坐标差值
                    diff = abs(up_pipe.x - low_pipe.x)
                    if diff < min_diff:
                        min_diff = diff
                        closest_pipes = (up_pipe, low_pipe)

            return closest_pipes  # 返回找到的最接近的管道组
        else:
            return None, None

    def get_reward(self):
        # 定义奖励机制
        if hasattr(self.flappy_game, 'player') and hasattr(self.flappy_game, 'pipes'):
            if self.flappy_game.player.collided(self.flappy_game.pipes, self.flappy_game.floor):
                return -1  # 碰撞时惩罚
            return 1  # 成功穿过管道时奖励
        else:
            return 0

    def is_game_over(self):
        if hasattr(self.flappy_game, 'player') and hasattr(self.flappy_game, 'pipes'):
            return self.flappy_game.player.collided(self.flappy_game.pipes, self.flappy_game.floor)
        else:
            return False