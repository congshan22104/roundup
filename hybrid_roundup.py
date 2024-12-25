import pybullet as p
import pybullet_data
import numpy as np
from gym import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

class MultiUAVCaptureEnv:
    def __init__(self):
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        self.plane = p.loadURDF("plane.urdf")

        # Initialize UAVs and target
        self.uavs = [p.loadURDF("sphere2.urdf", [np.random.uniform(-5, 5), np.random.uniform(-5, 5), 1]) for _ in range(3)]
        self.target = p.loadURDF("sphere2.urdf", [0, 0, 1])

        # Obstacles
        self.obstacles = [p.loadURDF("cube.urdf", [np.random.uniform(-5, 5), np.random.uniform(-5, 5), 0.5]) for _ in range(5)]

        # State and action spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)

        self.max_steps = 200
        self.step_count = 0

    def reset(self):
        self.step_count = 0

        for i, uav in enumerate(self.uavs):
            p.resetBasePositionAndOrientation(uav, [np.random.uniform(-5, 5), np.random.uniform(-5, 5), 1], [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.target, [np.random.uniform(-3, 3), np.random.uniform(-3, 3), 1], [0, 0, 0, 1])

        for obstacle in self.obstacles:
            p.resetBasePositionAndOrientation(obstacle, [np.random.uniform(-5, 5), np.random.uniform(-5, 5), 0.5], [0, 0, 0, 1])

        return self._get_obs()

    def step(self, actions):
        self.step_count += 1

        for i, action in enumerate(actions):
            pos, _ = p.getBasePositionAndOrientation(self.uavs[i])
            new_pos = np.clip(np.array(pos[:2]) + action[:2], -10, 10)
            p.resetBasePositionAndOrientation(self.uavs[i], [new_pos[0], new_pos[1], 1], [0, 0, 0, 1])

        # Update target position (simulate random movement)
        target_pos, _ = p.getBasePositionAndOrientation(self.target)
        target_new_pos = np.clip(np.array(target_pos[:2]) + np.random.uniform(-0.1, 0.1, size=2), -10, 10)
        p.resetBasePositionAndOrientation(self.target, [target_new_pos[0], target_new_pos[1], 1], [0, 0, 0, 1])

        reward = self._compute_reward()
        done = self._check_done()

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        obs = []
        for uav in self.uavs:
            pos, _ = p.getBasePositionAndOrientation(uav)
            obs.extend(pos[:2])

        target_pos, _ = p.getBasePositionAndOrientation(self.target)
        obs.extend(target_pos[:2])

        for obstacle in self.obstacles:
            pos, _ = p.getBasePositionAndOrientation(obstacle)
            obs.extend(pos[:2])

        return np.array(obs, dtype=np.float32)

    def _compute_reward(self):
        reward = 0
        target_pos, _ = p.getBasePositionAndOrientation(self.target)

        for uav in self.uavs:
            uav_pos, _ = p.getBasePositionAndOrientation(uav)
            distance = np.linalg.norm(np.array(uav_pos[:2]) - np.array(target_pos[:2]))
            reward -= distance

        return reward

    def _check_done(self):
        if self.step_count >= self.max_steps:
            return True

        target_pos, _ = p.getBasePositionAndOrientation(self.target)
        for uav in self.uavs:
            uav_pos, _ = p.getBasePositionAndOrientation(uav)
            distance = np.linalg.norm(np.array(uav_pos[:2]) - np.array(target_pos[:2]))
            if distance < 0.5:
                return True

        return False

    def render(self):
        pass

    def close(self):
        p.disconnect()

# Check environment
env = MultiUAVCaptureEnv()
check_env(env)

# Train the model
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Test the model
obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        break
