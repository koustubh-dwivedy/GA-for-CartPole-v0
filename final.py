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

X_DIVISIONS = 20
Y_DIVISIONS = 10

a = (x_max - x_min)/X_DIVISIONS
b = (y_max - y_min)/Y_DIVISIONS

EPISODES_PER_EVAL = 20

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
	return -score


'The following is a test for "evaluate" function'
'''
from random import randint
test = []
for i in range((X_DIVISIONS+1)*(Y_DIVISIONS+1)):
	test.append(randint(0, 1))
evaluate(test)
'''