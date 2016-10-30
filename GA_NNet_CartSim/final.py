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

HIDDEN_LAYER_SIZE = 10
POPULATION_SIZE = 50
NUM_GENERATION = 400
EPISODES_PER_EVAL = 2

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

def evaluate(agent):
	env = gym.make('CartPole-v0')
	score = 0
	for i_episode in range(EPISODES_PER_EVAL):
	    observation = env.reset()
	    for t in range(1000):
	        env.render()
	        action = action_function(agent, observation)
	        observation, reward, done, info = env.step(action)
	        score = score + reward
	        if done:
	            print("Episode finished after {} timesteps".format(t+1))
	            break
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

IND_SIZE = (HIDDEN_LAYER_SIZE*(4+1)) + (HIDDEN_LAYER_SIZE + 1)

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
env = gym.make('CartPole-v0')
env.monitor.start('/home/koustubh/Desktop/EDO/GA_NNet_CartSim/cartpole-experiment-1-high_res-run2', force=True)
for i_episode in range(EPISODES_PER_EVAL):
    observation = env.reset()
    for t in range(1000):
        env.render()
        action = action_function(solution[0], observation)
        observation, reward, done, info = env.step(action)
        score = score + reward
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.monitor.close()

print "SOLUTION: "
print solution
print "SOLUTION END"

print "final_score: "
print score

import sys
sys.stdout = open('solution', 'w')
print solution
