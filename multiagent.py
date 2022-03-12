import numpy as np
import matplotlib.pyplot as plt

### The input to multinomial sample is a probability distribution whose entries sum to 1
def multinomial_sample(prob_distribution):

	rand_value = np.random.random()
	cum_sum = 0
	for i in range(len(prob_distribution)):
		if rand_value < cum_sum + prob_distribution[i]:
			return i
		cum_sum += prob_distribution[i]
	raise ValueError("Something strange happened")

class AgentEXP3:
	def __init__(self, num_arms ):
		self.num_arms = num_arms
		self.reward_sums = [0]*num_arms


	def get_probability_distribution(self, learning_rate):
		potentials = [np.exp(learning_rate*self.reward_sums[i] ) for i in range(self.num_arms) ]
		normalization_factor = np.sum(potentials)

		probability_distribution = [x/normalization_factor for x in potentials]
		return probability_distribution

	def propose_action(self, learning_rate):
		probability_distribution = self.get_probability_distribution(learning_rate)
		#print("probability distribution ", probability_distribution)
		return multinomial_sample(probability_distribution)

	def update_reward_sums_bandits(self, reward, index, learning_rate):
		probability_distribution = self.get_probability_distribution(learning_rate)
		unbiased_reward = reward*1.0/probability_distribution[index]
		self.reward_sums[index]+= unbiased_reward

	def update_reward_sums_full_information(self, reward_vector):
		for i in range(self.num_arms):
			self.reward_sums[i] += reward_vector[i]


class MultiAgentWorld:
	def __init__(self, arm_means):
		self.arm_means = arm_means
		self.num_arms = len(arm_means)

	def get_arm_sample(self, arm_index):
		samples = [np.random.normal(arm_mean, .1) for arm_mean in self.arm_means]
		return samples[arm_index]


	def get_rewards(self, agent_actions):
		actions_counts = [agent_actions.count(a) for a in agent_actions]
		mask = [count == 1 for count in actions_counts]
		arm_samples = [self.get_arm_sample(agent_action) for agent_action in agent_actions]
		masked_arm_samples = [mask[i]*arm_samples[i] for i in range(len(agent_actions))]

		return masked_arm_samples






def run_experiment(num_arms, num_agents, T, arm_means):
	agents = [AgentEXP3(num_arms) for _ in range(num_agents)]
	world = MultiAgentWorld(arm_means)
	learning_rates = [(i+1)/np.sqrt(T) for i in range(num_agents)]
	joint_rewards = []
	#import IPython
	#IPython.embed()
	for t in range(T):
		agents_actions = [agent.propose_action(learning_rate) for agent, learning_rate in zip(agents, learning_rates)]
		print("Agents actions ", agents_actions)
		rewards = world.get_rewards(agents_actions)
		#print("Rewards ", rewards)
		
		if agents_actions[0] == agents_actions[1]:
			print("Collision!!!")
		[agent.update_reward_sums_bandits(reward, action, learning_rate) for agent, reward, action, learning_rate in zip(agents, rewards, agents_actions, learning_rates)]
		joint_rewards.append(np.sum(rewards))
	#print("Doing stuff")


	return joint_rewards






def main():
	num_experiments = 10

	num_arms = 10
	num_agents = 2
	T = 10000

	arm_means  = (np.arange(10) + 5)/15.0

	list_arm_means = list(arm_means)
	list_arm_means.sort()
	list_arm_means.reverse()

	optimal_reward =  np.sum(list_arm_means[:num_agents])

	joint_rewards_all = []
	for _ in range(num_experiments):
		joint_rewards_all.append(run_experiment(num_arms, num_agents, T, arm_means))

	#IPython.embed()

	joint_rewards_all_numpy = np.array(joint_rewards_all)

	plt.title("Joint rewards")
	plt.plot(np.arange(T)+1, joint_rewards_all_numpy.mean( axis = 0 ), label = "EXP3 Agents", color = "red")
	plt.plot(np.arange(T)+1, [optimal_reward]*T, label = "Optimal Reward", color = "black")
	plt.legend(loc = "lower right")

	plt.show()




if __name__ == "__main__":
	main()