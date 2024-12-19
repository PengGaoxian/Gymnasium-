## 1 gymnasium简介
- 所有Gym的开发工作已经转移到Gymnasium
- Gymnasium是一个较新的库，它视图解决Gym中的一些限制和问题，并提供更现代化的接口
- Gymnasium设计时考虑了Gym的兼容性。他提供了一个兼容层，使得大多数Gym环境可以直接在Gymnasium中使用，无需或只需少量修改

## 2 gymnasium的基本使用方法
### 2.1 基本使用流程
1. 导入gymnasium库
2. 创建环境
3. 重置环境
4. 执行动作
5. 关闭环境

代码示例
```python
import gymnasium as gym
env = gym.make('gym_examples/GridWorld-v0') # 创建环境

observation, info = env.reset(seed=42) # 重置环境

for _ in range(1000):
    action = env.action_space.sample() # 随机选择动作
    observation, reward, terminated, truncated, info = env.step(action) # 执行动作

    if terminated or truncated:
        break

env.close() # 关闭环境
```
### 2.2 封装器的使用
#### 2.2.1 预定义封装器
Gymnasium提供了多种封装器(Wrapper)来修改环境的行为。以下是一些常用的封装器:
- `TimeLimit`：限制每个回合的最大步数
- `RecordVideo`：记录环境渲染的视频
- `RecordEpisodeStatistics`：记录每个回合的统计信息
- `FrameStack`：将多个连续帧堆叠起来，形成一个状态表示
- `ResizeObservation`：将环境观测值的形状调整为指定的大小
- `ResizeAction`：将环境动作的形状调整为指定的大小
- `NormalizeObservation`：将环境观测值归一化到0-1之间
- `NormalizeReward`：将环境奖励归一化到0-1之间

使用示例
```python
import gymnasium as gym
env = gym.make('gym_examples/GridWorld-v0')
env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
env = gym.wrappers.RecordVideo(env, video_folder='videos')
env = gym.wrappers.RecordEpisodeStatistics(env)
env = gym.wrappers.FrameStack(env, num_stack=4)
env = gym.wrappers.ResizeObservation(env, shape=(84, 84))
env = gym.wrappers.ResizeAction(env, max_action=1.0)
env = gym.wrappers.NormalizeObservation(env)
env = gym.wrappers.NormalizeReward(env)
```

#### 2.2.2 自定义封装器
自定义封装器需要继承`gymnasium.Wrapper`类，并实现`__init__()`、`reset()`、`step()`、`render()`、`close()`等方法。

自定义封装器示例
```python
import gymnasium as gym

class CustomWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
```
使用自定义封装器
```python
import gymnasium as gym
env = CustomWrapper(gym.make('gym_examples/GridWorld-v0'))
```



## 3 使用gymnasium封装自定义环境
### 3.1 官方示例代码结构
示例文件结构
```
gym-examples/
├── gym_examples/ # 主包目录
│   ├── __init__.py # 环境注册
│   ├── envs/ # 环境实现目录
│   │   ├── __init__.py # 声明envs是一个包，用于导出grid_world.py中的环境类
│   │   └── grid_world.py # 具体环境实现
│   └── wrappers/ # 包装器目录（可选）
│       ├── __init__.py
│       ├── relative_position.py # 在grid_world.py基础上扩展环境
│       ├── reacher_weighted_reward.py
│       ├── discrete_action.py
│       └── clip_reward.py
├── setup.py # 包安装和依赖配置
└── README.md
```
wrapper是指包装器，用于修改或增强现有环境的行为，而不需要直接修改环境的源代码。
使用wrappers的一个关键优势是他们提供了一种灵活的方式来修改和扩展环境的功能，而不需要改变环境本身的实现。这使得研究人员可以专注于算法的开发，同时利用wrappers来适应不同的实验条件和研究目标。

### 3.2 编写环境文件（gym-examples/gym_examples/envs/grid_world.py）
所有自定义环境必须继承抽象类`gymnasium.Env`。同时需要定义metadata，在Gym环境中，metadata字典包含了环境的元数据，这些数据提供了关于环境行为和特性的额外信息
- `render_modes`：这个键的值是一个列表，指明了环境支持的渲染模式。在这个例子中，环境支持两种渲染模式：
	1. `human`：这种模式通常指在屏幕上以图形界面的形式渲染环境，适合人类观察者观看。
	2. `rgb_array`：这种模式下，环境的渲染结果会以RGB数组的形式返回，这可以用于机器学习算法的输入，或者进行进一步处理和分析。
- `render_fps`：这个键表示环境渲染的帧率，即每秒可以渲染的帧数。在这个例子中，4表示环境每秒4帧的速率进行渲染。这通常用于控制渲染速度，使动画的播放更加平滑或符合特定的显示需要。

在环境文件中需要实现`__init__()`、`reset()`、`step()`、`render()`、`close()`等方法，确保环境能够按照强化学习的标准工作流程运行
```python
import gymnasium as gym
from gymnasium import spaces

class GridWorldEnv(gym.Env):
	metadata = {
		"render_modes": ["human", "rgb_array"], # 环境支持的渲染模式
		"render_fps": 4 # 渲染帧率，每秒渲染4帧
	}
```

#### 3.2.1 `__init__()`方法
初始化方法，用于设置环境的初始状态。这里可以定义=环境参数=、初始化=状态空间=和=动作空间=等
	- 定义`observation_space`和`action_space`时，智能体可以执行的状态和动作的类型和范围，需要从Gymnasium导入spaces模块
	- spaces模块提供了多种空间类型，用于表示强化学习环境中可能的动作和观测的类型和结构
		- `Box`：用于表示连续空间，可以指定低值和高值，以及形状和数据类型
		- `Discrete`：用于表示离散空间，可以指定可能的数量
		- `Dict`：用于表示字典空间，可以指定多个键和对应的值空间
		- `Tuple`：用于表示元组空间，可以指定多个值空间
		- `MultiBinary`：用于表示多重二进制空间，可以指定每个二进制值的取值范围
		- `MultiDiscrete`：用于表示多重离散空间，可以指定每个离散值的取值范围
```python
class GridWorldEnv(gym.Env):
	# metadata

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
```
#### 3.2.2 `reset()`方法
用于重置环境状态，在每个训练周期（episode）开始时，`reset()`方法被调用以重置环境到一个初始状态
- 每次训练周期结束并且接收到结束信号（`done`标志）时，会调用`reset()`方法来重置环境状态
- 用户可以通过`reset()`方法传递一个`seed`参数，用于初始化环境使用的任何随机数生成器，确保环境行为的确定性和可复现性
```python
class GridWorldEnv(gym.Env):
	# metadata
	# __init__()

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
```
#### 3.2.3 `step()`方法
`step()` 方法是环境与智能体交互的核心，包含了环境逻辑的核心部分。方法处理动作，更新环境状态，并返回五个值组成的元组`(observation, reward, terminated, truncated, info)`：
- 观察（observation）：这是环境状态的表示，智能体根据这个观察来选择动作。观察可以是状态的一部分或全部，也可以是经过加工的信息，如图像、向量等。观察是智能体与环境交互的直接输入。
- 奖励（reward）：这是一个标量值，表示智能体执行动作后从环境中获得的即时反馈。奖励用于指导智能体学习哪些行为是好的，哪些不是好的。在许多任务中，智能体的目标是最大化其获得的总奖励。
- 是否结束（terminated）：这是一个布尔值，表示当前周期（episode）是否结束。如果是True，则表示智能体已经完成了任务，或者环境已经达到了一个终止状态，智能体需要重新开始新的周期。
- 是否截断（truncated）：这也是一个布尔值，与`terminated`类似，但表示周期结束的原因可能不是任务完成，而是其他原因，如超时、达到某个特定的中间状态或违反了某些规则。在某些实现中，truncated可能与terminated同时为True。
- 额外信息（info）：这是一个字典，包含除观察、奖励、终止和节点之外的额外信息。这些信息可以包括关于状态转换的元数据，如是否处于探索阶段、环境的内部计数器、额外的性能评估指标等。

```python
class GridWorldEnv(gym.Env):
	# metadata
	# __init__()
	# reset()

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
```
#### 3.2.4 `render()`方法
`render()` 方法用于将环境的状态可视化。使用Gymnasium创建自定义环境时，PyGame是一种流行的库，用于渲染环境的视觉表示。PyGame允许创建图形窗口，并将环境的状态绘制到屏幕上，这对于需要视觉反馈的强化学习任务非常有用。
- `human`：以图形界面的形式渲染，适用于人类观察者
- `rgb_array`：返回一个RGB图像数组，可以用于机器学习模型或进一步处理。

```python
class GridWorldEnv(gym.Env):
	# metadata
	# __init__()
	# reset()
	# step()

	def render(self):
		if self.render_mode == "rgb_array":
			return self._render_frame()

	def _render_frame(self):
		if self.window is None and self.render_mode == "human":
			pygame.init()
			pygame.display.init()
			self.window = pygame.display.set_mode(
				(self.window_size, self.window_size)
			)

		if self.clock is None and self.render_mode == "human":
			self.clock = pygame.time.Clock()

		canvas = pygame.Surface(
			(self.window_size, self.window_size)
		)
		canvas.fill(
			(255, 255, 255)
		)
		pix_square_size = (
			self.window_size / self.size
		) # 单个网格方块的像素大小

		# 首先绘制目标位置
		pygame.draw.rect(
			canvas,
			(255, 0, 0),
			pygame.Rect(
				pix_square_size * self._target_location,
				(pix_square_size, pix_square_size),
			),
		)
		# 现在绘制智能体
		pygame.draw.circle(
			canvas,
			(0, 0, 255),
			(self._agent_location + 0.5) * pix_square_size,
			pix_square_size / 3,
		)
		# 最后,添加一些网格线
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
			# 下面这行代码将我们在`canvas`上的绘图复制到可见窗口中
			self.window.blit(canvas, canvas.get_rect())
			pygame.event.pump()
			pygame.display.update()
			# 我们需要确保人类渲染在预定义的帧率下发生。
			# 下面的行将自动添加延迟以保持帧率稳定。
			self.clock.tick(self.metadata["render_fps"])
		else: # 渲染模式为rgb_array
			return np.transpose(
				np.array(pygame.surfarray.pixels3d(canvas)),
				axes=(1, 0, 2),
			)
```
#### 3.2.5 `close()`方法
`close()` 方法用于在环境不再使用时进行清理操作，例如关闭图形界面窗口、释放资源或执行其他必要的清理任务，是一个没有参数也没有返回值的方法。如果环境使用了PyGame或其他图形库创建了渲染窗口，`close()`方法会关闭这些窗口，释放资源。
```python
class GridWorldEnv(gym.Env):
	# metadata
	# __init__()
	# reset()
	# step()
	# render()

	def close(self):
		if self.window is not None:
			pygame.display.quit()
			pygame.quit()
```
## 4 注册环境
### 4.1 注册环境（`gym-examples/gym_examples/__init__.py`）
编写完上述与环境相关的代码后，需要注册自定义环境，注册自定义环境时为了使gymnasium检测到该环境
```python
from gymnasium.envs.registration import register

register(
	id="gym_examples/GridWorld-v0",
	entry_point="gym_examples.envs:GridWorldEnv",
	max_episode_steps=300,
)
```
environment ID由三部分组成:
- 命名空间`gym_examples`（可选）
- 环境名称`GridWorld`
- 版本号`v0`（可选）
`entry_point`参数在注册自定义环境时使用，它制定了如何导入这个环境类，格式通常是`module:classname`。`module`是包含环境类的Python模块的路径，`classname`是环境中具体的类的名称。其他可指定的参数如下：
- `reward_threshold`：指定智能体在环境中获得奖励的阈值，用于评估智能体的性能。
- `nondeterministic`：指定环境是否具有随机性，如果为True，则环境的行为可能会有所不同，即使输入相同。
- `max_episode_steps`：指定每个训练周期的最大步数，用于限制训练的持续时间。
- `order_enforce`：指定是否强制执行环境注册的顺序，如果为True，则环境必须按照注册的顺序加载。
- `autoreset`：指定是否在每个训练周期结束时自动重置环境，如果为True，则环境会在每个训练周期结束时自动重置。
- `kwargs`：指定其他可选参数，用于传递给环境类。

### 4.2 导入环境（`gym-examples/gym_examples/envs/__init__.py`）
文件中需要包含以下的内容，将envs目录视为一个包，可以从该包中导出GridWorldEnv，即`import envs.GridWorldEnv`
```python
from gym_examples.envs.grid_world import GridWorldEnv
```
### 4.3 测试环境（`gym_examples/test/test.py`)
经过注册的自定义环境`GridWorldEnv`可由以下命令创建
```python
import 
env = gym.make("gym_examples/GridWorld-v0")
```
## 5 打包&安装
### 5.1 创建包Package（`gym-examples/setup.py`）
将代码构建为python的包，方便在不同项目中重用自定义的环境代码，在`gym-examples/setup.py`中写入以下内容
```python
from setuptools import setup

setup(
	name="gym_examples",
	version="0.0.1",
	install_requires=["gymnasium>=0.26.0", "pygame>=2.1.0"],
)
```
此处可以将"=="改为">="，以确保安装的版本符合要求。如果不打算做图形化可以删除pygame的依赖。

### 5.2 安装环境包
安装自定义环境（在包含`setup.py`的目录下运行），作用是以“可编辑”（editable）或“开发”模式安装当前目录下的Python包。
```bash
pip install -e .
```
安装成功后会生成`gym_examples.egg-info`文件夹，执行命令后的底层逻辑流程：
1. 读取 `setup.py` 中的配置
2. 在 `site-packages` 创建一个 `.egg-link` 文件
3. 将项目路径添加到 `easy-install.pth`
优势：
- 开发时可以直接修改源代码，修改立即生效
- 不需要每次修改代码后都重新安装包
- 便于调试和开发
### 5.3 自定义环境示例
可以在任何目录位置，使用以下命令来导入自定义的环境
```python
import gym_examples
env = gymnasium.make('gym_examples/GridWorld-v0')
```
传参版本
```python
import gym_examples
env = gymnasium.make('gym_examples/GridWorld-v0', size=10)
