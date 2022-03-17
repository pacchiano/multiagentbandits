import numpy as np
import matplotlib.pyplot as plt
import ray
import IPython
from decimal import Decimal, localcontext

ray.init()
USE_RAY = True

def reshape_averaging(input_list, averaging_window):
	return np.mean(input_list.reshape(-1, averaging_window), axis = 1)



### The input to multinomial sample is a probability distribution whose entries sum to 1
def multinomial_sample(prob_distribution):

	rand_value = np.random.random()
	cum_sum = 0
	for i in range(len(prob_distribution)):
		if rand_value < cum_sum + prob_distribution[i]:
			return i
		cum_sum += prob_distribution[i]

	raise ValueError("Something strange happened ", "rand value ", rand_value, " cum sum ", cum_sum, " prob distribution ", prob_distribution)

class AgentEXP3:
	def __init__(self, num_arms ):
		self.num_arms = num_arms
		self.reward_sums = [0]*num_arms


	def get_probability_distribution(self, learning_rate):
		potentials = [np.exp(learning_rate*self.reward_sums[i] ) for i in range(self.num_arms) ]
		normalization_factor = np.sum(potentials)
		print("potentials ", potentials)
		print("reward sums ", self.reward_sums)
		probability_distribution = [x/normalization_factor for x in potentials]
		nan_map = np.isnan(probability_distribution)
		print("nan map ", nan_map)
		print("normalization factor ", normalization_factor)
		if nan_map.any():
			#IPython.embed()
			return nan_map*1.0/np.sum(nan_map)
			# with localcontext() as cont:

			# 	cont.prec = 100
			# 	IPython.embed()
			# 	probability_distribution = [float(Decimal(x)/Decimal(normalization_factor)) for x in potentials]
		return probability_distribution

	def propose_action(self, learning_rate, exploration_prob):
		probability_distribution = self.get_probability_distribution(learning_rate)
		#print("probability distribution ", probability_distribution)
		rand_value = np.random.random()
		if rand_value <= exploration_prob:
			return np.random.choice(range(self.num_arms))

		return multinomial_sample(probability_distribution)




	def update_reward_sums_bandits(self, reward, index, learning_rate, exploration_prob):
		probability_distribution = self.get_probability_distribution(learning_rate)
		unbiased_reward = reward*1.0/(exploration_prob/self.num_arms + (1-exploration_prob)*probability_distribution[index])
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


	def get_pseudo_rewards(self, agent_actions):
		actions_counts = [agent_actions.count(a) for a in agent_actions]
		mask = [count == 1 for count in actions_counts]
		arm_samples_means = [self.arm_means[agent_action] for agent_action in agent_actions]
		masked_arm_means = [mask[i]*arm_samples_means[i] for i in range(len(agent_actions))]

		return masked_arm_means





def run_experiment(num_arms, num_agents, T, arm_means):
	agents = [AgentEXP3(num_arms) for _ in range(num_agents)]
	world = MultiAgentWorld(arm_means)
	joint_rewards = []
	#import IPython
	exploration_prob = 1
	learning_rates = [1 for i in range(num_agents)]

	#IPython.embed()
	for t in range(T):
		#learning_rates = [(i+1)/np.sqrt(t+1) for i in range(num_agents)]
		learning_rates = [(i+1)/np.sqrt(t+1) for i in range(num_agents)]


		exploration_prob =  1.0/np.sqrt(t+1)

		agents_actions = [agent.propose_action(learning_rate, exploration_prob) for agent, learning_rate in zip(agents, learning_rates)]
		print("Agents actions ", agents_actions)
		rewards = world.get_rewards(agents_actions)
		pseudo_rewards = world.get_pseudo_rewards(agents_actions)
		#print("Rewards ", rewards)
		## This only works for the two agent case.
		if len(set(agents_actions)) != num_agents:
			print("Collision!!!")
		

		[agent.update_reward_sums_bandits(reward, action, learning_rate, exploration_prob) for agent, reward, action, learning_rate in zip(agents, rewards, agents_actions, learning_rates)]
		joint_rewards.append(np.sum(pseudo_rewards))
	#print("Doing stuff")


	return joint_rewards




@ray.remote
def run_experiment_remote(num_arms, num_agents, T, arm_means):
	return run_experiment(num_arms, num_agents, T, arm_means)



def main():
	num_experiments = 2
	T = 1000000

	averaging_window = 10

	nums_arms = [10, 20]
	nums_agents = [2,3, 4]

	for num_arms in nums_arms:
		for num_agents in nums_agents:
			arm_means  = (np.arange(num_arms) + 5)/(num_arms*3.0/2)

			list_arm_means = list(arm_means)
			list_arm_means.sort()
			list_arm_means.reverse()

			optimal_reward =  np.sum(list_arm_means[:num_agents])


			#joint_rewards_all = []
			
			if USE_RAY:
				joint_rewards_all = [run_experiment_remote.remote(num_arms, num_agents, T, arm_means) for _ in range(num_experiments)]
				joint_rewards_all = ray.get(joint_rewards_all)
			else:
				joint_rewards_all = [run_experiment(num_arms, num_agents, T, arm_means) for _ in range(num_experiments)]

			# for _ in range(num_experiments):
			# 	joint_rewards_all.append(run_experiment(num_arms, num_agents, T, arm_means))

			#IPython.embed()
			joint_rewards_all_numpy = np.array(joint_rewards_all)


			timesteps = np.arange(T)+1
			mean_joint_rewards = joint_rewards_all_numpy.mean( axis = 0 )
			std_joint_rewards = joint_rewards_all_numpy.std( axis = 0 )
			
			mean_joint_rewards = reshape_averaging(mean_joint_rewards, averaging_window)
			std_joint_rewards = reshape_averaging(std_joint_rewards, averaging_window)
			timesteps = reshape_averaging(timesteps, averaging_window)


			plt.title("Joint rewards - num players {} - num arms {} ".format(num_agents, num_arms))
			plt.plot(timesteps, mean_joint_rewards,  label = "EXP3 Agents", color = "red")

			plt.fill_between(timesteps, mean_joint_rewards - .5*std_joint_rewards, 
					mean_joint_rewards + .5*std_joint_rewards, color = "red", alpha = .2)


			plt.plot(np.arange(T)+1, [optimal_reward]*T, label = "Optimal Reward", color = "black")
			plt.legend(loc = "lower right")
			plt.xlabel("Timesteps")
			plt.ylabel("Agents' reward")
			plt.savefig("./figs/multiagent_T{}_A{}_M{}.png".format(T, num_arms, num_agents))

			plt.close("all")



if __name__ == "__main__":
	main()