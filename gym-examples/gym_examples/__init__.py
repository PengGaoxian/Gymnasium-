from gymnasium.envs.registration import register

register(
	id="gym_examples/GridWorld-v0", # 注册环境
	entry_point="gym_examples.envs:GridWorldEnv", # 指定环境实现类(依赖于gym_examples/envs/__init__.py)
	max_episode_steps=300, # 设置最大步数
)