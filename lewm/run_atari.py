import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt

gym.register_envs(ale_py)
env = gym.make("ALE/Pacman-v5", render_mode="rgb_array")
obs, _ = env.reset()

fig, ax = plt.subplots(figsize=(6, 8))
plt.ion()

terminated = truncated = False
total_reward = 0.0
step = 0
while not (terminated or truncated):
    frame = env.render()
    ax.cla()
    ax.imshow(frame)
    ax.axis("off")
    ax.set_title(f"step {step}  |  total reward: {total_reward:.0f}")
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)
    input("Press Enter for next step...")
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    print(f"action: {action} | reward: {reward:.3f} | terminated: {terminated} | truncated: {truncated}")
    # print(f"obs: {obs[:]}")  # print first few values of the observation vector
    total_reward += reward
    step += 1

print(f"total reward: {total_reward:.0f}")
env.close()
plt.close()
