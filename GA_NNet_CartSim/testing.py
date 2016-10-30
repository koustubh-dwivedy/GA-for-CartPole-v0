import gym
import numpy as np
import math
import os


HIDDEN_LAYER_SIZE = 10
POPULATION_SIZE = 50
LOCATION = "cartpole-experiment-1-high_res-run2"



def sigmoid(a):
    return 1.0/(1 + pow(math.e, -a))

def action_function(agent, observation):
    a = np.zeros(shape=(5, 1))
    a[0, 0] = 1
    a[1, 0] = observation[0]
    a[2, 0] = observation[1]
    a[3, 0] = observation[2]
    a[4, 0] = observation[3]
    a = np.matrix(a)

    temp_agent = agent[0:(HIDDEN_LAYER_SIZE*(4+1))]
    np_agent_1d_hidden = np.array(temp_agent)
    np_agent_2d_hidden = np_agent_1d_hidden.reshape((HIDDEN_LAYER_SIZE), (4+1))
    np_agent_2d_hidden = np.matrix(np_agent_2d_hidden)

    temp_agent = agent[(HIDDEN_LAYER_SIZE*(4+1)):]
    np_agent_1d_final = np.array(temp_agent)
    np_agent_2d_final = np_agent_1d_final.reshape(1, (HIDDEN_LAYER_SIZE + 1))
    np_agent_2d_final = np.matrix(np_agent_2d_final)

    temp_mat = np.matrix(sigmoid(np.array(np_agent_2d_hidden*a)))
    temp_mat = np.matrix(np.append([1], temp_mat))
    temp_mat = temp_mat.transpose()
    value = np.matrix(sigmoid(np.array(np_agent_2d_final*temp_mat)))

    if value < 0.5:
        return 0
    else:
        return 1


os.chdir(LOCATION)
solution = np.loadtxt('solution')
for i in range(POPULATION_SIZE):
    score = 0
    env = gym.make('CartPole-v0')
    observation = env.reset()
    for t in range(1000):
    	env.render()
    	action = action_function(solution[i], observation)
    	observation, reward, done, info = env.step(action)
    	score = score + reward
    	if done:
    		break
    print("Episode finished after {} timesteps".format(t+1))
    print "Score: "
    print score