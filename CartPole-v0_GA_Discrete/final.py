import gym
import math
environ = gym.make('CartPole-v0')

'The following are the observation types which have been considered for taking actions'
'We have not considered the observations which are not bounded'
'NOTE that the values of the agents will be either 0 or 1 (corresponding to -1 or +1 force respectively)'

x_max = environ.observation_space.high[0]
x_min = environ.observation_space.low[0]
y_max = environ.observation_space.high[2]
y_min = environ.observation_space.low[2]

X_DIVISIONS = 200
Y_DIVISIONS = 100
POPULATION_SIZE = 50
NUM_GENERATION = 400
EPISODES_PER_EVAL = 10

a = (x_max - x_min)/X_DIVISIONS
b = (y_max - y_min)/Y_DIVISIONS

def action_function(agent, observation):
	x = observation[0]
	y = observation[2]

	val_A = agent[int((Y_DIVISIONS/2 - (math.floor(y/b) + 1))*(X_DIVISIONS + 1) + (Y_DIVISIONS + math.floor(x/a)))]
	val_B = agent[int((Y_DIVISIONS/2 - (math.floor(y/b) + 1))*(X_DIVISIONS + 1) + (Y_DIVISIONS + 1 + math.floor(x/a)))]
	val_C = agent[int((Y_DIVISIONS/2 + 1 - (math.floor(y/b) + 1))*(X_DIVISIONS + 1) + (Y_DIVISIONS + 1 + math.floor(x/a)))]
	val_D = agent[int((Y_DIVISIONS/2 + 1 - (math.floor(y/b) + 1))*(X_DIVISIONS + 1) + (Y_DIVISIONS + math.floor(x/a)))]
	weight_x1 = (x - math.floor(x/a)*a)
	weight_x2 = a - weight_x1
	weight_y1 = (y - math.floor(y/b)*b)
	weight_y2 = b - weight_y1

	value = math.floor(2*(val_A*weight_x1*weight_y2 + val_B*weight_x2*weight_y2 + val_C*weight_x2*weight_y1 + val_D*weight_x1*weight_y1)/(a*b))

	if value == 0:
		return 0
	else:
		return 1

def evaluate(agent):
	env = gym.make('CartPole-v0')
	score = 0
	for i_episode in range(EPISODES_PER_EVAL):
	    observation = env.reset()
	    for t in range(100):
	        env.render()
	        action = action_function(agent, observation)
	        observation, reward, done, info = env.step(action)
	        score = score + reward
	        if done:
	            print("Episode finished after {} timesteps".format(t+1))
	            break
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

IND_SIZE = (X_DIVISIONS+1)*(Y_DIVISIONS+1)

toolbox = base.Toolbox()
toolbox.register("attribute", random.randint, 0, 1)
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
env.monitor.start('/home/koustubh/Desktop/EDO/cartpole-experiment-1-high_res', force=True)
for i_episode in range(EPISODES_PER_EVAL):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = action_function(solution[0], observation)
        observation, reward, done, info = env.step(action)
        score = score + reward
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.monitor.close()

print score