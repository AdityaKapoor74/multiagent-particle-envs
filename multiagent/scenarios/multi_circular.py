import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import webcolors


class Scenario(BaseScenario):
	def make_world(self):
		world = World()
		# set any world properties first
		# world.dim_c = 2
		self.num_agents = 4
		self.num_landmarks = 4
		print("NUMBER OF AGENTS:",self.num_agents)
		print("NUMBER OF LANDMARKS:",self.num_landmarks)
		world.collaborative = True

		# add agents
		world.agents = [Agent() for i in range(self.num_agents)]
		for i, agent in enumerate(world.agents):
			agent.name = 'agent %d' % i
			agent.collide = True
			agent.silent = True
			agent.size = 0.15 #was 0.15
			agent.prevDistance = 0.0
		# add landmarks
		world.landmarks = [Landmark() for i in range(self.num_landmarks)]
		for i, landmark in enumerate(world.landmarks):
			landmark.name = 'landmark %d' % i
			landmark.collide = False
			landmark.movable = False
		# make initial conditions
		self.reset_world(world)
		return world

	def reset_world(self, world):
		color_choice = [np.array([255,0,0]), np.array([0,255,0]), np.array([0,0,255]), np.array([0,0,0]), np.array([128,0,0]), np.array([0,128,0]), np.array([0,0,128]), np.array([128,128,128])]
		# AGENT 0 : red
		# AGENT 1 : lime
		# AGENT 2 : blue
		# AGENT 3 : black

		base_color = np.array([0.1, 0.1, 0.1])

		for i in range(self.num_agents):
			# rgb = np.random.uniform(-1,1,3)
			# world.agents[i].color = rgb
			# world.landmarks[i].color = rgb
			world.agents[i].color = color_choice[i]
			world.landmarks[i].color = color_choice[i]
			# print("AGENT", world.agents[i].name[-1], ":", webcolors.rgb_to_name((color_choice[i][0],color_choice[i][1],color_choice[i][2])))

		# set random initial states
		for i, agent in enumerate(world.agents):
			if i == 0:
				agent.state.p_pos = np.array([-0.15,-0.85])
			elif i== 3:
				agent.state.p_pos = np.array([-0.85,-0.85])
			elif i== 1:
				agent.state.p_pos = np.array([0.15,-0.85])
			elif i== 2:
				agent.state.p_pos = np.array([0.85,-0.85])

			agent.state.p_vel = np.zeros(world.dim_p)
			agent.state.c = np.zeros(world.dim_c)
			agent.prevDistance = 0.0

		for i, landmark in enumerate(world.landmarks):
			if i == 0:
				landmark.state.p_pos = np.array([-0.85,0.85])
			elif i== 3:
				landmark.state.p_pos = np.array([-0.15,0.85])
			elif i== 1:
				landmark.state.p_pos = np.array([0.85,0.85])
			elif i== 2:
				landmark.state.p_pos = np.array([0.15,0.85])

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
		dist_min = agent1.size + agent2.size
		return True if dist < dist_min else False


	def reward(self, agent, world):
		my_index = int(agent.name[-1])
		
		agent_dist_from_goal = np.sqrt(np.sum(np.square(world.agents[my_index].state.p_pos - world.landmarks[my_index].state.p_pos)))

		rew = agent.prevDistance - agent_dist_from_goal
		agent.prevDistance = agent_dist_from_goal

		if world.agents[my_index].collide:
			for a in world.agents:
				if self.is_collision(a, world.agents[my_index]):
					rew -= 0.1
		
		return rew


	def observation(self, agent, world):

		curr_agent_index = world.agents.index(agent)

		current_agent_critic = [agent.state.p_pos,agent.state.p_vel,world.landmarks[curr_agent_index].state.p_pos]
		
		
		current_agent_actor = [agent.state.p_pos,agent.state.p_vel,world.landmarks[curr_agent_index].state.p_pos]
		other_agents_actor = []

		# for other_agent in world.agents:
		# 	if other_agent is agent:
		# 	  continue
		# 	other_agents_actor.append(other_agent.state.p_pos-agent.state.p_pos)
		# 	other_agents_actor.append(other_agent.state.p_vel-agent.state.p_vel)

		return np.concatenate(current_agent_critic),np.concatenate(current_agent_actor+other_agents_actor)


	def isFinished(self,agent,world):
		index = int(agent.name[-1])
		dist = np.sqrt(np.sum(np.square(world.agents[index].state.p_pos - world.landmarks[index].state.p_pos)))
		if dist<0.1:
			return True
		return False
		
