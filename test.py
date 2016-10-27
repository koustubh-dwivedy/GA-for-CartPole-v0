import gym
env = gym.make('CartPole-v0')
env.monitor.start('/home/koustubh/Desktop/EDO/cartpole-experiment-1', force=True)
for i_episode in range(20):
    observation = env.reset()
    action = 0
    for t in range(100):
        env.render()
        print(observation)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.monitor.close()
