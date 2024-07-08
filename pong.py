import numpy as np
import gym
from gym import spaces

# 定義斜拋運動環境
class ProjectileEnv(gym.Env):
    def __init__(self):
        super(ProjectileEnv, self).__init__()
        self.max_angle = 0.5 * np.pi  # 最大角度
        self.min_angle = 0  # 最小角度
        self.action_space = spaces.Discrete(2)  # 動作空間：增加或減少角度
        self.observation_space = spaces.Box(low=np.array([self.min_angle]), high=np.array([self.max_angle]), dtype=np.float32)
        self.angle = 2 / 7 * np.pi  # 初始角度

    def reset(self):
        self.angle = 2 / 7 * np.pi
        return np.array([self.angle], dtype=np.float32)

    def step(self, action):
        if action == 0:
            self.angle -= 0.01 * np.pi
        elif action == 1:
            self.angle += 0.01 * np.pi

        self.angle = np.clip(self.angle, self.min_angle, self.max_angle)  # 限制角度範圍

        distance = self.calculate_distance(self.angle)  # 計算拋射距離
        reward = distance  # 設定回報為距離

        done = True  # 每次只計算一次拋射

        return np.array([self.angle], dtype=np.float32), reward, done, {}

    def calculate_distance(self, angle):
        # 物理計算斜拋運動的水平距離
        g = 9.81  # 重力加速度
        v0 = 10  # 初速度
        theta = angle  # 使用弧度制
        distance = (v0**2 * np.sin(2 * theta)) / g
        return distance

# 訓練PPO
import tensorflow as tf
from tensorflow.keras import layers

# 定義策略網絡
class PolicyNetwork(tf.keras.Model):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.logits = layers.Dense(2)  # 輸出兩個動作的logits

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.logits(x)

# 定義價值網絡
class ValueNetwork(tf.keras.Model):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.value = layers.Dense(1)  # 輸出狀態的價值

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.value(x)

def get_action(policy_net, state):
    logits = policy_net(state)
    action_prob = tf.nn.softmax(logits)  # 計算動作概率
    action = np.random.choice(2, p=action_prob.numpy()[0])  # 根據概率選擇動作
    return action, action_prob

def compute_advantages(rewards, values, gamma=0.99):
    advantages = []
    gae = 0
    for reward, value in zip(rewards[::-1], values[::-1]):
        delta = reward - value + gamma * gae
        gae = delta
        advantages.append(gae)
    return advantages[::-1]

# 訓練PPO
env = ProjectileEnv()
policy_net = PolicyNetwork()
value_net = ValueNetwork()
policy_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
value_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
gamma = 0.99
clip_ratio = 0.2

for episode in range(1000):
    state = env.reset()
    state = np.expand_dims(state, axis=0)
    done = False
    rewards = []
    states = []
    actions = []
    action_probs = []
    values = []

    while not done:
        action, action_prob = get_action(policy_net, state)
        value = value_net(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.expand_dims(next_state, axis=0)
        
        rewards.append(reward)
        states.append(state)
        actions.append(action)
        action_probs.append(action_prob)
        values.append(value)

        state = next_state

    advantages = compute_advantages(rewards, values, gamma)

    for state, action, old_prob, advantage in zip(states, actions, action_probs, advantages):
        with tf.GradientTape() as tape:
            logits = policy_net(state)
            new_prob = tf.nn.softmax(logits)[0][action]
            ratio = new_prob / old_prob[0][action]
            clipped_ratio = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio)
            policy_loss = -tf.minimum(ratio * advantage, clipped_ratio * advantage)  # 策略損失
        
        grads = tape.gradient(policy_loss, policy_net.trainable_variables)
        policy_optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))

    for state, reward in zip(states, rewards):
        with tf.GradientTape() as tape:
            value = value_net(state)
            value_loss = tf.square(reward - value)  # 價值損失

        grads = tape.gradient(value_loss, value_net.trainable_variables)
        value_optimizer.apply_gradients(zip(grads, value_net.trainable_variables))

    if episode % 100 == 0:
        print(f'Episode {episode}, Reward: {sum(rewards)}')
