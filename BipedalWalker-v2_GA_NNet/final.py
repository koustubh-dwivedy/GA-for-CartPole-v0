'This is the code which uses Genetic Algorithm to train the Neural Network which balances a cart'
'''
The configuration of NNet is:
Input Layer: 4 nodes
Hidden Layer: 10 nodes (chosen arbitrarily)
Output Layer: 1 node
'''

import gym
import math
import numpy as np

'The following are the observation types which have been considered for taking actions'
'We have not considered the observations which are not bounded'
'NOTE that the values of the agents will be either 0 or 1 (corresponding to -1 or +1 force respectively)'

HIDDEN_LAYER_SIZE = 30
POPULATION_SIZE = 50
NUM_GENERATION = 400
EPISODES_PER_EVAL = 2
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

def evaluate(agent):
	env = gym.make('BipedalWalker-v2')
	score = 0
	for i_episode in range(EPISODES_PER_EVAL):
	    observation = env.reset()
	    for t in range(100):
	        env.render()
	        action = action_function(agent, observation)
	        observation, reward, done, info = env.step(action)
	        score = score + reward
	        if done:
	            break
	env.close()
	print("Episode finished after {} timesteps".format(t+1))
	print "score: "
	print score
	return score,


'The following is a test for "evaluate" function'
'''
from random import randint
test = []
for i in range((X_DIVISIONS+1)*(Y_DIVISIONS+1)):
	test.append(randint(0, 1))
evaluate(test)
'''

from deap import base, creator
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

import random
from deap import tools

IND_SIZE = (HIDDEN_LAYER_SIZE*(OBSERVATION_SPACE_DIM+1)) + ACTION_SPACE_DIM*(HIDDEN_LAYER_SIZE + 1)

toolbox = base.Toolbox()
toolbox.register("attribute", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

def main():
    pop = toolbox.population(n=POPULATION_SIZE)
    CXPB, MUTPB, NGEN = 0.5, 0.2, NUM_GENERATION

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = map(toolbox.clone, offspring)

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

    return pop
'''************************************'''
solution = main()
score = 0
env = gym.make('BipedalWalker-v2')
env.monitor.start('/home/koustubh/Desktop/EDO/BipedalWalker-v2_GA_NNet/bipedalwalker_run1', force=True)
for i_episode in range(EPISODES_PER_EVAL):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = action_function(solution[0], observation)
        observation, reward, done, info = env.step(action)
        score = score + reward
        if done:
            break
    print("Episode finished after {} timesteps".format(t+1))
env.monitor.close()

print "SOLUTION: "
print solution
print "SOLUTION END"

print "final_score: "
print score

import sys
sys.stdout = open('bipedalwalker_run1/solution', 'w')
print solution
