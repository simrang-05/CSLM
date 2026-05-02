from cr_env import CR_Environment

env = CR_Environment()

state, _ = env.reset()

for _ in range(5):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, _ = env.step(action)

    print("Reward:", reward)