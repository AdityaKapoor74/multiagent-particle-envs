import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import webcolors
import math
import random


class Scenario(BaseScenario):
	def make_world(self):
		world = World()
		# set any world properties first
		# world.dim_c = 2
		self.num_agents = 12
		self.num_landmarks = 12
		self.threshold_dist = 1e-1
		self.goal_reward = 1e-1
		self.pen_existence = 1e-2
		self.team_size = 4
		self.agent_size = 0.1
		self.landmark_size = 0.05
		print("NUMBER OF AGENTS:",self.num_agents)
		print("NUMBER OF FOOD PELLETS:",self.num_landmarks)
		print("TEAM SIZE", self.team_size)
		world.collaborative = True

		# add agents
		world.agents = [Agent() for i in range(self.num_agents)]
		for i, agent in enumerate(world.agents):
			agent.name = 'agent %d' % i
			agent.collide = False
			agent.silent = True
			agent.size = self.agent_size
			agent.prevDistance = None
			if i<self.team_size:
				agent.team_id = 1
				agent.base_camp = np.array([-1,-1])
			elif i>= self.team_size and i<2*self.team_size:
				agent.team_id = 2
				agent.base_camp = np.array([-1,1])
			elif i>= 2*self.team_size and i<3*self.team_size:
				agent.team_id = 3
				agent.base_camp = np.array([1,1])
			elif i>= 3*self.team_size and i<4*self.team_size:
				agent.team_id = 4
				agent.base_camp = np.array([1,-1])
		# add landmarks
		world.landmarks = [Landmark() for i in range(self.num_landmarks)]
		for i, landmark in enumerate(world.landmarks):
			landmark.name = 'landmark %d' % i
			landmark.collide = False
			landmark.movable = False
			if i<self.team_size:
				landmark.team_id = 1
			elif i>= self.team_size and i<2*self.team_size:
				landmark.team_id = 2
			elif i>= 2*self.team_size and i<3*self.team_size:
				landmark.team_id = 3
			elif i>= 3*self.team_size and i<4*self.team_size:
				landmark.team_id = 4
		# make initial conditions
		self.reset_world(world)
		return world

	def reset_world(self, world):
		agent_list = []
		color_choice = [np.array([255,0,0]), np.array([0,255,0]), np.array([0,0,255]), np.array([0,0,0]), np.array([255,0,255])]
		
		for i in range(self.num_agents):
			# rgb = np.random.uniform(-1,1,3)
			if i < self.team_size:
				world.agents[i].color = color_choice[0]
				world.landmarks[i].color = color_choice[0]
			elif i>=self.team_size and i<2*self.team_size:
				world.agents[i].color = color_choice[1]
				world.landmarks[i].color = color_choice[1]
			elif i>=2*self.team_size and i<3*self.team_size:
				world.agents[i].color = color_choice[2]
				world.landmarks[i].color = color_choice[2]
			elif i>=3*self.team_size and i<4*self.team_size:
				world.agents[i].color = color_choice[3]
				world.landmarks[i].color = color_choice[3]

		for agent in world.agents:
			x = random.uniform(-1,1)
			y = random.uniform(-1,1)
			agent.state.p_pos = np.array([x,y])
			agent.state.p_vel = np.zeros(world.dim_p)
			agent.state.c = np.zeros(world.dim_c)

		for landmark in world.landmarks:
			x = random.uniform(-1,1)
			y = random.uniform(-1,1)
			landmark.state.p_pos = np.array([x,y])
			landmark.state.p_vel = np.zeros(world.dim_p)
				


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


	def reward(self, agent, world):
		# np.sqrt(np.sum(np.square(delta_pos)))
		my_index = int(agent.name[-1])
		
		agent_dist_from_goal = np.array([np.sqrt(np.sum(np.square(agent.state.p_pos - landmark.state.p_pos))) for landmark in world.landmarks])

		rew = np.amin(agent_dist_from_goal)

		# on reaching goal we reward the agent
		goal_reached = 0
		if agent_dist_from_goal<self.threshold_dist:
		# 	# rew += self.goal_reward
			# agent.goal_reached = True
			goal_reached = 1
		# else:
		# 	agent.goal_reached = False
		# 	rew -= self.pen_existence
			

		return rew, collision_count, goal_reached


	def observation(self, agent, world):
		if agent.state.p_pos[0]>1.0:
			agent.state.p_pos[0] = 1.0
		if agent.state.p_pos[0]<-1.0:
			agent.state.p_pos[0] = -1.0
		if agent.state.p_pos[1]<-1.0:
			agent.state.p_pos[1] = -1.0
		if agent.state.p_pos[1]>1.0:
			agent.state.p_pos[1] = 1.0

		curr_agent_index = world.agents.index(agent)
		# team = [0 for i in range(self.num_agents//self.team_size)]
		# team[agent.team_id-1] = 1
		# team = np.array(team)
		team = np.array([agent.team_id])
		current_agent_critic = [agent.state.p_pos, agent.state.p_vel, team, world.landmarks[curr_agent_index].state.p_pos]
		current_agent_actor = [agent.state.p_pos, agent.state.p_vel, team, world.landmarks[curr_agent_index].state.p_pos]
		for other_agent in world.agents:
			if other_agent.name == agent.name:
				continue
			# relative pose, velocity wrt current_agent and team id of other agent 
			current_agent_actor.extend([other_agent.state.p_pos-agent.state.p_pos,other_agent.state.p_vel-agent.state.p_vel,np.array([other_agent.team_id])])
		
		return np.concatenate(current_agent_critic), np.concatenate(current_agent_actor)


	def isFinished(self,agent,world):
		for other_agent in world.agents:
			index = world.agents.index(other_agent)
			dist = np.sqrt(np.sum(np.square(other_agent.state.p_pos - world.landmarks[index].state.p_pos)))
			if dist>self.threshold_dist:
				return False
		return True
		
