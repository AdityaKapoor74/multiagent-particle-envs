import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import webcolors
import random


class Scenario(BaseScenario):
	def make_world(self):
		world = World()
		# set any world properties first
		# world.dim_c = 2
		self.num_predator = 1
		self.num_prey = 1
		self.penalty_of_existence = -0.01
		print("NUMBER OF PREDATORS:",self.num_predator)
		print("NUMBER OF PREYS:",self.num_prey)
		world.collaborative = True

		# add predator
		world.agents = [Agent() for i in range(self.num_predator)]
		for i, predator in enumerate(world.agents):
			predator.name = 'predator %d' % i
			predator.collide = True
			predator.silent = True
			predator.size = 0.1 #was 0.15
			predator.prevDistance = 0.0

		# add prey
		for i in range(self.num_prey):
			world.agents.append(Agent())
		for i, prey in enumerate(world.agents[self.num_predator:]):
			prey.name = 'prey %d' % i
			prey.collide = False
			prey.silent = True
			prey.size = 0.1 #was 0.15
			prey.prevDistance = 0.0

		# make initial conditions
		self.reset_world(world)
		return world

	def check_collision_before_spawning(self,agent,agent_list):

		for other_agent in agent_list:
			if agent.name == other_agent.name:
				continue
			delta_pos = agent.state.p_pos - other_agent.state.p_pos
			dist = np.sqrt(np.sum(np.square(delta_pos)))
			dist_min = (agent.size + other_agent.size) * 1.5
			if dist < dist_min:
				return True 

		return False

	def reset_world(self, world):
		agent_list = []
		for i in range(self.num_predator+self.num_prey):
			if "predator" in world.agents[i].name:
				rgb = np.random.uniform(-1,1,3)
				rgb = np.random.randint(0,255,3)
				world.agents[i].color = rgb
			else:
				world.agents[i].color = np.array([0.0,0.0,0.0])
			world.agents[i].state.p_pos = np.random.uniform(-1, +1, world.dim_p)
			while self.check_collision_before_spawning(world.agents[i], agent_list):
				world.agents[i].state.p_pos = np.random.uniform(-1, +1, world.dim_p)
			agent_list.append(world.agents[i])

			world.agents[i].state.p_vel = np.zeros(world.dim_p)
			world.agents[i].state.c = np.zeros(world.dim_c)
			world.agents[i].prevDistance = 0.0

		for i in range(self.num_predator,self.num_predator+self.num_prey):
			world.agents[i].color = np.array([0.0,0.0,0.0])

	def benchmark_data(self, agent, world):
		rew = 0
		collisions = 0
		occupied_landmarks = 0
		min_dists = 0
		for l in world.landmarks:
			dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
			min_dists += min(dists)
			rew -= min(dists)
			if min(dists) < 0.1:
				occupied_landmarks += 1
		if agent.collide:
			for a in world.agents:
				if self.is_collision(a, agent):
					rew -= 1
					collisions += 1
		return (rew, collisions, min_dists, occupied_landmarks)


	def is_collision(self, agent1, agent2):
		if agent1.name == agent2.name:
			return False
		delta_pos = agent1.state.p_pos - agent2.state.p_pos
		dist = np.sqrt(np.sum(np.square(delta_pos)))
		dist_min = (agent1.size + agent2.size) * 1.5
		return True if dist < dist_min else False


	def reward(self, agent, world):

		if "prey" in agent.name:
			return None

		if agent.collide:
			for a in world.agents[self.num_agents:]:
				if self.is_collision_agent(a, agent):
					rew += 30.0

		rew += self.penalty_of_existence
		
		return rew


	def observation(self, agent, world):

		if "predator" in agent.name:
			curr_agent_index = world.agents.index(agent)
			current_predator_critic = [agent.state.p_pos,agent.state.p_vel]
			current_predator_actor = [agent.state.p_pos,agent.state.p_vel]
			return np.concatenate(current_predator_critic),np.concatenate(current_predator_actor)

		elif "prey" in agent.name:
			current_prey_critic = [agent.state.p_pos,agent.state.p_vel]
			current_prey_actor = [agent.state.p_pos,agent.state.p_vel]
			return np.concatenate(current_prey_critic),np.concatenate(current_prey_actor)


	def isFinished(self,agent,world):

		if "prey" in agent.name:
			return None
		else:
			return False
		# NEED TO THINK
		
