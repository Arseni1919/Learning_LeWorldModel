import gymnasium as gym

# env = gym.make("Ant-v5", render_mode="human")
env = gym.make("Humanoid-v5", render_mode="human")
obs, _ = env.reset()

terminated = truncated = False
total_reward = 0.0
step = 0
while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    step += 1
    print(f"\rstep {step:4d}  |  total reward: {total_reward:.1f}", end="")

print()
env.close()
