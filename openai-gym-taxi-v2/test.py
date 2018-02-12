import gym
env = gym.make("Taxi-v2")
observation = env.reset()
print('observation: ', observation)
for _ in range(3):
	env.render()
	action = env.action_space.sample() # your agent here (this takes random actions)
	observation, reward, done, info = env.step(action)
	print(action, observation, reward, done, info)
	print('----------------')

# action: 0 -> down, 1 -> up, 2 -> right, 3 -> left, 4 -> pickup, 5 -> dropoff
# (379, -1, False, {'prob': 1.0}), (observation, reward, done, )
