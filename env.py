import pygame
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from functools import partial
import random
import math
from robot import Robot, Config

# ==================== Параметры среды ====================
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
PPM = 1  # пикселей на метр (pixels per meter)

# Параметры манипулятора
L1 = 100.0
L2 = 150.0
THETA1_LIM = (-np.pi, np.pi)
THETA2_LIM = (-np.pi, np.pi)
MAX_DELTA = 0.1  # максимальное изменение угла за шаг

# Цель (фиксированная) - in robot coordinate system
TARGET = (200.0, 50.0)  # coords relative to robot base at (0, 0)
TARGET_THRESH = 15.0  # порог достижения цели (in robot coords)

# Параметры RL
GAMMA = 0.99
LR = 1e-3
HIDDEN_DIM = 128
NUM_EPISODES = 2000
MAX_STEPS_PER_EPISODE = 200

# Параметры штрафа за близость theta2 к границам
THETA2_PENALTY_THRESH = 0.6      # радиан (≈11.5°)
THETA2_PENALTY_SCALE = 0.9       # масштаб штрафа

# Параметры вознаграждения за достижение цели
GOAL_REWARD = 10.0                # дополнительная награда при достижении цели

# ==================== Класс среды (двухзвенный манипулятор) ====================
class TwoLinkArmEnv:
    def __init__(self, target=TARGET, render_mode=True):
        self.target = np.array(target, dtype=np.float32)
        self.render_mode = render_mode
        
        # Create Robot using Robot class
        cfg = Config(base_xy=(0.0, 0.0), link_lengths=(L1, L2), wrap_angles=True, dtheta_max=MAX_DELTA)
        self.robot = Robot(cfg, seed=42)
        
        if render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Two-Link Arm RL (REINFORCE + JAX, начальное положение 0,0)")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 18)

        self.steps = 0
        self.reset()

    def reset(self):
        # Reset robot to initial state (theta=0, or random if randomize=True)
        self.robot.set_theta(np.array([0.0, 0.0]))
        self.steps = 0
        return self.get_state()

    def step(self, action):
        # action: целое число от 0 до 8
        # декодируем: сначала для первого сустава, потом для второго
        delta1 = ((action // 3) - 1) * MAX_DELTA   # -1, 0, 1 -> -0.1, 0, 0.1
        delta2 = ((action % 3) - 1) * MAX_DELTA

        # Use robot.step() to update angles
        self.robot.step(np.array([delta1, delta2]))
        self.steps += 1

        # Get end-effector position from robot
        ee = self.robot.end_effector_xy()

        # расстояние до цели
        dist = np.linalg.norm(ee - self.target)

        # Основная награда: отрицательное расстояние (стимулирует приближение)
        reward = -dist

        # Штраф за близость theta2 к границам ±π
        theta2 = self.robot.theta[1]
        dist_to_boundary = np.pi - abs(theta2)
        if dist_to_boundary < THETA2_PENALTY_THRESH:
            reward -= THETA2_PENALTY_SCALE * (THETA2_PENALTY_THRESH - dist_to_boundary)

        # Бонус за достижение цели
        goal_reached = dist < TARGET_THRESH
        if goal_reached:
            reward += GOAL_REWARD

        # проверка завершения эпизода
        done = goal_reached or self.steps >= MAX_STEPS_PER_EPISODE

        return self.get_state(), reward, done, {}

    def get_state(self):
        # Get observation from robot and add target info
        ee = self.robot.end_effector_xy()
        dist = np.linalg.norm(ee - self.target)
        dx = self.target[0] - ee[0]
        dy = self.target[1] - ee[1]
        
        # Use robot's obs: [sin(th1), cos(th1), sin(th2), cos(th2), x_ee, y_ee]
        robot_obs = self.robot.obs()
        
        # Normalize spatial values for better learning (all in range ~[-1, 1])
        max_dist = L1 + L2 + 100  # typical max distance
        return np.array([
            robot_obs[1], robot_obs[0],  # cos(th1), sin(th1)  - match original order
            robot_obs[3], robot_obs[2],  # cos(th2), sin(th2)  - match original order
            dist / max_dist,             # normalize distance
            dx / max_dist,               # normalize direction
            dy / max_dist
        ], dtype=np.float32)

    def render(self):
        if not self.render_mode:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.screen.fill((255, 255, 255))

        # преобразование координат в пиксели (ось Y вниз)
        def to_pixels(x, y):
            px = int(x * PPM + SCREEN_WIDTH // 2)
            py = int(-y * PPM + SCREEN_HEIGHT // 2)  # минус, чтобы Y вверх
            return px, py

        # рисуем цель
        tx, ty = to_pixels(self.target[0], self.target[1])
        pygame.draw.circle(self.screen, (255, 0, 0), (tx, ty), 8)
        pygame.draw.circle(self.screen, (200, 0, 0), (tx, ty), 4)

        # Get robot joints from Robot class
        joints = self.robot.joints_xy()  # (3, 2) array: [base, joint1, ee]
        base_pos = joints[0]
        joint1_pos = joints[1]
        ee_pos = joints[2]

        # рисуем манипулятор
        base = to_pixels(base_pos[0], base_pos[1])
        j1 = to_pixels(joint1_pos[0], joint1_pos[1])
        ee = to_pixels(ee_pos[0], ee_pos[1])

        # звенья
        pygame.draw.line(self.screen, (0, 0, 0), base, j1, 5)
        pygame.draw.line(self.screen, (0, 0, 0), j1, ee, 5)

        # суставы
        pygame.draw.circle(self.screen, (0, 0, 255), base, 8)
        pygame.draw.circle(self.screen, (0, 0, 255), j1, 6)
        pygame.draw.circle(self.screen, (0, 255, 0), ee, 6)

        # информация
        ee_actual = self.robot.end_effector_xy()
        dist = np.linalg.norm(ee_actual - self.target)
        theta = self.robot.theta
        text = self.font.render(f"Dist: {dist:.3f}  Step: {self.steps}  θ1={theta[0]:.2f} θ2={theta[1]:.2f}", True, (0,0,0))
        self.screen.blit(text, (10, 10))

        pygame.display.flip()
        self.clock.tick(FPS)

# ==================== Политика (нейросеть на Flax) ====================
class PolicyNetwork(nn.Module):
    hidden_dim: int = 128
    n_actions: int = 9

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_actions)(x)
        return x  # logits

def select_action(params, state, rng_key):
    # state: numpy array (7,)
    state_jnp = jnp.expand_dims(jnp.array(state), axis=0)
    logits = model.apply(params, state_jnp)[0]  # убираем batch dim
    probs = jax.nn.softmax(logits)
    rng_key, subkey = jax.random.split(rng_key)
    action = jax.random.categorical(subkey, logits)
    log_prob = jnp.log(probs[action])
    return int(action), float(log_prob), rng_key

def compute_returns(rewards, gamma):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return np.array(returns, dtype=np.float32)

def loss_fn(params, states, actions, returns):
    # states: [T, 7], actions: [T], returns: [T]
    logits = model.apply(params, states)  # [T, n_actions]
    log_probs = jax.nn.log_softmax(logits)
    chosen_log_probs = jnp.take_along_axis(log_probs, jnp.expand_dims(actions, axis=1), axis=1).squeeze()
    loss = -jnp.sum(chosen_log_probs * returns)
    return loss

@jax.jit
def update(params, opt_state, states, actions, returns):
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(params, states, actions, returns)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

# ==================== Основной цикл обучения ====================
def main():
    global model, optimizer
    model = PolicyNetwork(hidden_dim=HIDDEN_DIM, n_actions=9)

    rng = jax.random.PRNGKey(42)
    dummy_state = jnp.ones((1, 7))  # состояние размерности 7
    params = model.init(rng, dummy_state)

    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    episode_rewards = []
    episode_lengths = []

    env = TwoLinkArmEnv(render_mode=True)

    rng_episode = rng
    for episode in range(NUM_EPISODES):
        state = env.reset()
        done = False
        rewards = []
        states = []
        actions = []

        while not done:
            env.render()
            action, log_prob, rng_episode = select_action(params, state, rng_episode)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        returns = compute_returns(rewards, GAMMA)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        states_jnp = jnp.array(np.array(states))
        actions_jnp = jnp.array(actions)
        returns_jnp = jnp.array(returns)

        params, opt_state = update(params, opt_state, states_jnp, actions_jnp, returns_jnp)

        total_reward = sum(rewards)
        episode_rewards.append(total_reward)
        episode_lengths.append(len(rewards))

        if episode % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:]) if episode_rewards else 0
            avg_length = np.mean(episode_lengths[-20:]) if episode_lengths else 0
            print(f"Episode {episode}\tAvg reward (last 20): {avg_reward:.2f}\tAvg length: {avg_length:.2f}")

    print("Обучение завершено!")
    pygame.quit()

if __name__ == "__main__":
    main()