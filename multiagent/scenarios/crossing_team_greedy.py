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
		self.num_agents = 24

		self.agent_ids = []
		num_bits_required_agent_id = self.num_agents.bit_length()
		for agent_num in range(self.num_agents):
			binary_rep = bin(agent_num).replace("0b", "")
			binary_rep += "0"*(num_bits_required_agent_id-len(binary_rep))
			encoding = list(map(int, binary_rep))
			self.agent_ids.append(encoding)

		self.num_landmarks = 24
		self.threshold_dist = 0.25
		self.goal_reward = 0.0
		self.pen_existence = 1e-2
		self.team_size = 8

		self.num_teams = self.num_agents // self.team_size

		# self.team_ids = []
		# num_bits_required_team_id = self.num_teams.bit_length()
		# for team_num in range(self.num_teams):
		# 	binary_rep = bin(team_num).replace("0b", "")
		# 	binary_rep += "0"*(num_bits_required_team_id-len(binary_rep))
		# 	encoding = list(map(int, binary_rep))
		# 	self.team_ids.append(encoding)

		self.agent_ids = []
		for i in range(self.num_agents):
			agent_id = np.zeros(self.num_agents)
			agent_id[i] = 1
			self.agent_ids.append(agent_id)
		self.agent_ids = np.array(self.agent_ids)

		self.team_ids = []
		for i in range(self.num_teams):
			team_id = np.zeros(self.num_teams)
			team_id[i] = 1
			self.team_ids.append(team_id)
		self.team_ids = np.array(self.team_ids)

		# full observation_shape
		# self.transformer_observation_shape = 2*3 + num_bits_required_agent_id + num_bits_required_team_id
		self.critic_observation_shape = 2*3 + self.num_teams + self.num_agents
		# self.observation_shape = 2*3 + num_bits_required_agent_id + num_bits_required_team_id + (self.num_agents-1)*(2*2+num_bits_required_team_id+num_bits_required_agent_id)
		self.actor_observation_shape = 2*3 + self.num_agents + self.num_teams + (self.num_agents-1)*(2*2+self.num_teams+self.num_agents)

		self.pen_collision = 0.1
		self.agent_size = 0.1
		self.landmark_size = 0.1
		print("NUMBER OF AGENTS:",self.num_agents)
		print("NUMBER OF LANDMARKS:",self.num_landmarks)
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
			for ts in range(self.num_agents//self.team_size):
				if i >= ts*self.team_size and i < (ts+1)*self.team_size:
					agent.team_id = ts
					break
			
		# add landmarks
		world.landmarks = [Landmark() for i in range(self.num_landmarks)]
		for i, landmark in enumerate(world.landmarks):
			landmark.name = 'landmark %d' % i
			landmark.collide = False
			landmark.movable = False
			for ts in range(self.num_agents//self.team_size):
				if i >= ts*self.team_size and i < (ts+1)*self.team_size:
					landmark.team_id = i
			
		# make initial conditions
		self.reset_world(world)
		return world

	def check_collision_before_spawning(self, entity, entity_list):
		for other_entity in entity_list:
			if (entity.name == other_entity.name) or (entity.team_id != other_entity.team_id):
				continue
			delta_pos = entity.state.p_pos - other_entity.state.p_pos
			dist = np.sqrt(np.sum(np.square(delta_pos)))
			dist_min = max(self.agent_size, self.landmark_size) * 4
			if dist < dist_min:
				return True 

		return False

	def reset_world(self, world):
		agent_list = []
		landmark_list = []
		color_choice = [np.array([255,0,0]), np.array([0,255,0]), np.array([0,0,255]), np.array([255,255,0]), np.array([255,0,255]), np.array([0,255,255])]
		
		for i in range(self.num_agents):
			for t_id in range(self.num_agents//self.team_size):
				if i >= t_id*self.team_size and i < (t_id+1)*self.team_size:
					world.agents[i].color = color_choice[t_id]
					world.landmarks[i].color = color_choice[t_id]
					break


			x = random.uniform(-1, 1)
			y = random.uniform(-1, 1)
			world.agents[i].state.p_pos = np.array([x,y])
			while self.check_collision_before_spawning(world.agents[i], agent_list):
				x = random.uniform(-1, 1)
				y = random.uniform(-1, 1)
				world.agents[i].state.p_pos = np.array([x,y])

			agent_list.append(world.agents[i])

			x = random.uniform(-1, 1)
			y = random.uniform(-1, 1)
			world.landmarks[i].state.p_pos = np.array([x,y])
			while self.check_collision_before_spawning(world.landmarks[i], landmark_list):
				x = random.uniform(-1, 1)
				y = random.uniform(-1, 1)
				world.landmarks[i].state.p_pos = np.array([x,y])

			landmark_list.append(world.landmarks[i])

			# SPAWN IN THE CORNER OF THE ROOM
			# if i%self.team_size == 0:
			# 	y = random.uniform(-1,1)
			# 	x = -1
			# 	world.agents[i].state.p_pos = np.array([x,y])
			# 	world.landmarks[i].state.p_pos = np.array([-x,y])
			# 	while self.check_collision_before_spawning(world.agents[i], None, agent_list, None):
			# 		y = random.uniform(-1,1)
			# 		world.agents[i].state.p_pos = np.array([x,y])
			# 		world.landmarks[i].state.p_pos = np.array([-x,y])
			# 	world.agents[i].direction = "y"
			# elif i%self.team_size == 1:
			# 	x = random.uniform(-1,1)
			# 	y = -1
			# 	world.agents[i].state.p_pos = np.array([x,y])
			# 	world.landmarks[i].state.p_pos = np.array([x,-y])
			# 	while self.check_collision_before_spawning(world.agents[i], None, agent_list, None):
			# 		x = random.uniform(-1,1)
			# 		world.agents[i].state.p_pos = np.array([x,y])
			# 		world.landmarks[i].state.p_pos = np.array([x,-y])
			# 	world.agents[i].direction = "x"
			# elif i%self.team_size == 2:
			# 	y = random.uniform(-1,1)
			# 	x = 1
			# 	world.agents[i].state.p_pos = np.array([x,y])
			# 	world.landmarks[i].state.p_pos = np.array([-x,y])
			# 	while self.check_collision_before_spawning(world.agents[i], None, agent_list, None):
			# 		y = random.uniform(-1,1)
			# 		world.agents[i].state.p_pos = np.array([x,y])
			# 		world.landmarks[i].state.p_pos = np.array([-x,y])
			# 	world.agents[i].direction = "-y"
			# elif i%self.team_size == 3:
			# 	x = random.uniform(-1,1)
			# 	y = 1
			# 	world.agents[i].state.p_pos = np.array([x,y])
			# 	world.landmarks[i].state.p_pos = np.array([x,-y])
			# 	while self.check_collision_before_spawning(world.agents[i], None, agent_list, None):
			# 		x = random.uniform(-1,1)
			# 		world.agents[i].state.p_pos = np.array([x,y])
			# 		world.landmarks[i].state.p_pos = np.array([x,-y])
			# 	world.agents[i].direction = "-x"

			# agent_list.append(world.agents[i])

			world.agents[i].state.p_vel = np.zeros(world.dim_p)
			world.agents[i].state.c = np.zeros(world.dim_c)
			world.agents[i].prevDistance = None
			world.landmarks[i].state.p_vel = np.zeros(world.dim_p)



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
		if (agent1.name == agent2.name) or (agent1.team_id != agent2.team_id):
			return False
		delta_pos = agent1.state.p_pos - agent2.state.p_pos
		dist = np.sqrt(np.sum(np.square(delta_pos)))
		dist_min = agent1.size*2
		return True if dist < dist_min else False


	def reward(self, agent, world):
		my_index = int(agent.name[-1])
		
		agent_dist_from_goal = np.sqrt(np.sum(np.square(world.agents[my_index].state.p_pos - world.landmarks[my_index].state.p_pos)))

		# if agent.prevDistance is None:
		# 	rew = 0
		# else:
		# 	rew = (agent.prevDistance - agent_dist_from_goal)

		# agent.prevDistance = agent_dist_from_goal

		rew = -agent_dist_from_goal

		collision = 0
		for other_agent in world.agents:
			if self.is_collision(agent, other_agent):
				rew -= self.pen_collision
				collision = 1

		# on reaching goal we reward the agent
		goal_reached = 0
		if agent_dist_from_goal<self.threshold_dist:
			rew += self.goal_reward
			# agent.goal_reached = True
			goal_reached = 1
		# else:
		# 	agent.goal_reached = False
		# 	rew -= self.pen_existence

		# scaling reward
		rew /= (2.0*self.num_agents)

		return rew, collision, goal_reached


	def observation(self, agent, world):
		if agent.state.p_pos[0]>1.0:
			agent.state.p_pos[0] = 1.0
		if agent.state.p_pos[0]<-1.0:
			agent.state.p_pos[0] = -1.0
		if agent.state.p_pos[1]<-1.0:
			agent.state.p_pos[1] = -1.0
		if agent.state.p_pos[1]>1.0:
			agent.state.p_pos[1] = 1.0

		map_x = map_y = 2.0

		curr_agent_index = world.agents.index(agent)
		# curr_agent_id = np.array([world.agents.index(agent)])
		curr_agent_id = np.array(self.agent_ids[curr_agent_index])
		# curr_agent_team_id = np.array([agent.team_id])
		curr_agent_team_id = np.array(self.team_ids[agent.team_id])

		agent_x, agent_y = agent.state.p_pos[0]/map_x, agent.state.p_pos[1]/map_y
		# agent.state.p_vel[0], agent.state.p_vel[1] = agent.state.p_vel[0]/agent.max_speed, agent.state.p_vel[1]/max_speed
		landmark_x, landmark_y = world.landmarks[curr_agent_index].state.p_pos[0]/map_x, world.landmarks[curr_agent_index].state.p_pos[1]/map_y
		current_agent_critic = [curr_agent_id, curr_agent_team_id, np.array([agent_x,agent_y]), agent.state.p_vel, np.array([landmark_x, landmark_y])]
		current_agent_actor = [curr_agent_id, curr_agent_team_id, np.array([agent_x,agent_y]), agent.state.p_vel, np.array([landmark_x, landmark_y])]

		for other_agent in world.agents:
			if other_agent.name == agent.name:
				continue
			# agent_id, team_id, relative pose, relative velocity wrt current_agent
			
			# agent_id = np.array([world.agents.index(other_agent)])
			other_agent_x, other_agent_y = other_agent.state.p_pos[0]/map_x, other_agent.state.p_pos[1]/map_y
			# other_agent.state.p_vel[0], other_agent.state.p_vel[1] = other_agent.state.p_vel[0]/agent.max_speed, other_agent.state.p_vel[1]/max_speed
			relative_pose = np.array([other_agent_x, other_agent_y])-np.array([agent_x, agent_y])
			relative_vel = other_agent.state.p_vel-agent.state.p_vel
			agent_id = np.array(self.agent_ids[world.agents.index(other_agent)])
			agent_team_id = np.array(self.team_ids[other_agent.team_id])
			# agent_team_id = np.array([other_agent.team_id])
			current_agent_actor.extend([agent_id, agent_team_id, relative_pose, relative_vel])

		return np.concatenate(current_agent_critic, axis=-1), np.concatenate(current_agent_actor, axis=-1)


	def isFinished(self, agent, world):
		index = world.agents.index(agent)
		dist = np.sqrt(np.sum(np.square(agent.state.p_pos - world.landmarks[index].state.p_pos)))
		if dist>self.threshold_dist:
			return False
		return True
		
