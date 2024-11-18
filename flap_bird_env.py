import asyncio
import time

import gym
import numpy as np
from gym import spaces

class FlappyEnv(gym.Env):
    def __init__(self, flappy_game):
        super(FlappyEnv, self).__init__()
        # 我只关注笨鸟和它前面的管道口坐标的位置
        self.state_dim = 9
        self.flappy_game = flappy_game

        # 定义动作空间（0: 不跳，1: 跳）
        self.action_space = spaces.Discrete(1)

        # 定义状态空间，可以根据你的游戏状态定义
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

        # 定义局编号
        self.current_game_num = 0
        # 定义需要跳过的pip
        self.up_pipe, self.low_pipe = None, None

    def reset(self):
        # 重置游戏状态
        import threading
        thread = threading.Thread(target=self.start_flappy_game)
        thread.start()
        self.current_game_num == self.flappy_game.game_num
        return self.get_state()

    def start_flappy_game(self):
        asyncio.run(self.flappy_game.start())

    def step(self, action, num_step):
        # 执行动作
        threshold = 0.5
        if action > threshold and hasattr(self.flappy_game, 'player'):  # 跳
            self.flappy_game.player.flap()

        # 获取新的状态、奖励和是否结束
        time.sleep(0.1)
        state, up_pipe, low_pipe = self.get_state()
        reward = self.get_reward()
        done = False
        '''
            0:  (player.x + player.w) if player else 0,
            1:  player.y if player else 0,
            2:  (player.y + player.h)if player else 0,
            3:  up_pipe.x if up_pipe else 0,
            4:  (up_pipe.x + up_pipe.w) if up_pipe else 0,
            5:  (up_pipe.y + up_pipe.h) if up_pipe else 0,
            6:  low_pipe.x if low_pipe else 0,
            7:  (low_pipe.x + low_pipe.w)if low_pipe else 0,
            8:  low_pipe.y if low_pipe else 0
        '''
        if state[5] < state[1] < state[8]:
            reward += 0.1

        if self.current_game_num < self.flappy_game.game_num:
            self.current_game_num = self.flappy_game.game_num
            reward -= 1
            done = True

        # if self.up_pipe and self.low_pipe:
        #     if self.up_pipe.number != up_pipe.number \
        #             and self.up_pipe in self.flappy_game.pipes.upper \
        #             and self.low_pipe in self.flappy_game.pipes.lower:
        #         # 此处需要修改bug，重新开始博弈后也会掉进这个逻辑里
        #         reward += 1
        #         self.up_pipe, self.low_pipe = up_pipe, low_pipe
        # else:
        #     self.up_pipe, self.low_pipe = up_pipe, low_pipe
        reward += 0.1 * (self.flappy_game.score.score - 1) + 0.1 * num_step

        return state, reward, done, {}

    def get_state(self):
        # 找到笨鸟前面的管道up和low各一个
        up_pipe, low_pipe = self.find_closest_pipes()
        # 返回当前状态的表示
        player = self.get_play()
        return np.array([
            (player.x + player.w) if player else 0,
            player.y if player else 0,
            (player.y + player.h)if player else 0,
            up_pipe.x if up_pipe else 0,
            (up_pipe.x + up_pipe.w) if up_pipe else 0,
            (up_pipe.y + up_pipe.h) if up_pipe else 0,
            low_pipe.x if low_pipe else 0,
            (low_pipe.x + low_pipe.w)if low_pipe else 0,
            low_pipe.y if low_pipe else 0
        ]), up_pipe, low_pipe

    def get_play(self):
        if hasattr(self.flappy_game, 'player'):
            return self.flappy_game.player
        else:
            return None
    def find_closest_pipes(self):
        closest_pipes = None
        min_diff = float('inf')  # 初始化最低差值为正无穷

        # 遍历上下管道
        if hasattr(self.flappy_game, 'pipes'):
            upper_pipes = self.flappy_game.pipes.upper
            lower_pipes = self.flappy_game.pipes.lower

            for i in range(len(upper_pipes)):
                up_pipe = upper_pipes[i]
                low_pipe = lower_pipes[i]
                if (up_pipe.x + up_pipe.w) > self.flappy_game.player.x and \
                        (low_pipe.x + low_pipe.w) > self.flappy_game.player.x:
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
        if hasattr(self.flappy_game, 'player') and hasattr(self.flappy_game, 'pipes') \
                and hasattr(self.flappy_game, 'floor'):
            # 碰撞时惩罚
            if self.flappy_game.player.collided(self.flappy_game.pipes, self.flappy_game.floor):
                return -0.5
            return 0.1
        else:
            return 0

    def is_game_over(self):
        if hasattr(self.flappy_game, 'player') and hasattr(self.flappy_game, 'pipes'):
            return self.flappy_game.player.collided(self.flappy_game.pipes, self.flappy_game.floor)
        else:
            return False