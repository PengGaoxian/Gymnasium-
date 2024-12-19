import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class GridWorldEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4
    }

    def __init__(self, render_mode=None, size=5):
        self.size = size # 正方形网格的大小
        self.window_size = 512 # PyGame窗口的大小
		
        self.observation_space = spaces.Dict({
			"agent": spaces.Box(0, size - 1, shape=(2,), dtype=int), # 智能体的位置，是一个2维的Box空间，取值范围为0到size-1
			"target": spaces.Box(0, size - 1, shape=(2,), dtype=int), # 目标的位置，是一个2维的Box空间，取值范围为0到size-1
		})
		
        self.action_space = spaces.Discrete(4) # 动作空间，是一个离散空间，包含4个动作
        self._action_to_direction = {
			0: np.array([1, 0]), # 向右
			1: np.array([0, 1]), # 向上
			2: np.array([-1, 0]), # 向左
			3: np.array([0, -1]), # 向下
		}

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
		# 我们需要这一行来为self.np_random设置随机种子
        super().reset(seed=seed)
		# 随机均匀地选择智能体的位置
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
		# 随机选择目标位置，直到目标位置不与智能体位置相同
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
				0, 
				self.size,
				size=2,
				dtype=int
			)

        observation = self._get_obs() # 获取观测值
        info = self._get_info() # 获取额外信息

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
		# 将动作(取值为{0,1,2,3})映射为行走方向
        direction = self._action_to_direction[action]
		# 使用np.clip确保智能体不会离开网格
        self._agent_location = np.clip(
			self._agent_location + direction, 0, self.size - 1
		)
		# 如果智能体到达目标位置,则当前回合结束
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # 布尔值稀疏奖励
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _get_obs(self): # 返回观测值
        return {
			"agent": self._agent_location,
			"target": self._target_location
		}
	
    def _get_info(self): # 收集除observation和reward之外的额外信息
        return {
			"distance": np.linalg.norm(
				self._agent_location - self._target_location, ord=1
			)
		}
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init() # 初始化pygame
            pygame.display.init() # 初始化pygame的显示模块
            self.window = pygame.display.set_mode(
				(self.window_size, self.window_size)
			) # 设置窗口大小为window_size x window_size

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock() # 创建一个时钟对象，用于控制渲染的帧率

        canvas = pygame.Surface(
			(self.window_size, self.window_size)
		) # 创建一个大小为window_size x window_size的画布，用于绘制环境
        canvas.fill(
			(255, 255, 255)
		) # 将画布填充为白色
        pix_square_size = (
			self.window_size / self.size
		) # 设置单个网格方块的像素大小

		# 首先绘制目标位置
        pygame.draw.rect(
			canvas,
			(255, 0, 0),
			pygame.Rect(
				pix_square_size * self._target_location,
				(pix_square_size, pix_square_size),
			),
		) # 绘制目标位置，使用红色填充
		# 然后绘制智能体
        pygame.draw.circle(
			canvas,
			(0, 0, 255),
			(self._agent_location + 0.5) * pix_square_size,
			pix_square_size / 3,
		) # 绘制智能体，使用蓝色填充
		# 最后添加一些网格线
        for x in range(self.size + 1):
            pygame.draw.line(
				canvas,
				0,
				(0, pix_square_size * x),
				(self.window_size, pix_square_size * x),
				width=3,
			)
            pygame.draw.line(
				canvas,
				0,
				(pix_square_size * x, 0),
				(pix_square_size * x, self.window_size),
				width=3,
			)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect()) # 将画布上的绘图复制到窗口中
            pygame.event.pump() # 处理事件，作用是处理事件队列中的事件，确保事件被处理
            pygame.display.update() # 更新窗口，作用是更新窗口的显示，确保窗口内容被更新
            self.clock.tick(self.metadata["render_fps"]) # 控制渲染的帧率
        else: # 渲染模式为rgb_array
            return np.transpose(
				np.array(pygame.surfarray.pixels3d(canvas)), # 将画布转换为3维数组
				axes=(1, 0, 2), # 将数组的轴顺序从(0, 1, 2)改为(1, 0, 2)，即从(height, width, channels)改为(width, height, channels)
			)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()