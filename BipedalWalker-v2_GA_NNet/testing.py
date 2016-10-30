import gym
import numpy as np
import math
import os


HIDDEN_LAYER_SIZE = 30
POPULATION_SIZE = 50
LOCATION = "bipedalwalker_run1"
OBSERVATION_SPACE_DIM = 24
ACTION_SPACE_DIM = 4


def sigmoid(a):
    return 1.0/(1 + pow(math.e, -a))

def action_function(agent, observation):
    a = np.zeros(shape=(OBSERVATION_SPACE_DIM + 1, 1))
    a[0, 0] = 1
    a[1, 0] = observation[0]
    a[2, 0] = observation[1]
    a[3, 0] = observation[2]
    a[4, 0] = observation[3]
    a[5, 0] = observation[4]
    a[6, 0] = observation[5]
    a[7, 0] = observation[6]
    a[8, 0] = observation[7]
    a[9, 0] = observation[8]
    a[10, 0] = observation[9]
    a[11, 0] = observation[10]
    a[12, 0] = observation[11]
    a[13, 0] = observation[12]
    a[14, 0] = observation[13]
    a[15, 0] = observation[14]
    a[16, 0] = observation[15]
    a[17, 0] = observation[16]
    a[18, 0] = observation[17]
    a[19, 0] = observation[18]
    a[20, 0] = observation[19]
    a[21, 0] = observation[20]
    a[22, 0] = observation[21]
    a[23, 0] = observation[22]
    a[24, 0] = observation[23]
    a = np.matrix(a)

    temp_agent = agent[0:(HIDDEN_LAYER_SIZE*(OBSERVATION_SPACE_DIM+1))]
    np_agent_1d_hidden = np.array(temp_agent)
    np_agent_2d_hidden = np_agent_1d_hidden.reshape((HIDDEN_LAYER_SIZE), (OBSERVATION_SPACE_DIM+1))
    np_agent_2d_hidden = np.matrix(np_agent_2d_hidden)

    temp_agent = agent[(HIDDEN_LAYER_SIZE*(OBSERVATION_SPACE_DIM+1)):]
    np_agent_1d_final = np.array(temp_agent)
    np_agent_2d_final = np_agent_1d_final.reshape(ACTION_SPACE_DIM, (HIDDEN_LAYER_SIZE + 1))
    np_agent_2d_final = np.matrix(np_agent_2d_final)

    temp_mat = np.matrix(sigmoid(np.array(np_agent_2d_hidden*a)))
    temp_mat = np.matrix(np.append([1], temp_mat))
    temp_mat = temp_mat.transpose()
    value = np.matrix(sigmoid(np.array(np_agent_2d_final*temp_mat)))

    value = value*2
    value = value - 1
    return value


os.chdir(LOCATION)
solution = np.loadtxt('solution')
score_mat = []
for i in range(POPULATION_SIZE):
    score = 0
    env = gym.make('BipedalWalker-v2')
    observation = env.reset()
    for t in range(100):
    	env.render()
    	action = action_function(solution[i], observation)
    	observation, reward, done, info = env.step(action)
    	score = score + reward
    	if done:
    		break
    score_mat.append(score)
    print("Episode finished after {} timesteps".format(t+1))
    print "Score: "
    print score_mat