import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO

# 定義斜拋運動環境
class ProjectileEnv(gym.Env):
    def __init__(self):
        super(ProjectileEnv, self).__init__()
        self.max_angle = 0.5 * np.pi
        self.min_angle = 0
        self.action_space = spaces.Discrete(2)  # 增加或減少角度
        self.observation_space = spaces.Box(low=np.array([self.min_angle]), high=np.array([self.max_angle]), dtype=np.float32)
        self.angle = 0 * np.pi # 初始角度

    def reset(self):
        self.angle = 0 * np.pi
        return np.array([self.angle], dtype=np.float32)

    def step(self, action):
        if action == 0:
            self.angle -= 0.01 * np.pi
        elif action == 1:
            self.angle += 0.01 * np.pi

        self.angle = np.clip(self.angle, self.min_angle, self.max_angle)

        distance = self.calculate_distance(self.angle)
        reward = distance

        done = True  # 每次只計算一次拋射

        return np.array([self.angle], dtype=np.float32), reward, done, {}

    def calculate_distance(self, angle):
        # 物理計算斜拋運動的水平距離
        g = 9.81  # 重力加速度
        v0 = 10  # 初速度
        theta = angle
        distance = (v0**2 * np.sin(2 * theta)) / g
        return distance

# 創建環境
env = ProjectileEnv()

# 使用PPO算法進行訓練
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000)

# 評估模型
obs = env.reset()
for i in range(300):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    print(f"Angle: {obs[0]}, Distance: {rewards}")
