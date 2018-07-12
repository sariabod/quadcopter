import numpy as np
from physics_sim import PhysicsSim
import torch
import torch.nn as nn

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class LandTask():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation, lets always start in random position
        runtime = 5
        init_pose = np.random.rand(6) * np.random.randint(1, 100)
        init_velocities = None
        init_angle_velocities = None

        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal to land safely
        self.target_pos = np.array([0., 0., 0., 0., 0., 0.])

        self.loss_func = nn.SmoothL1Loss()
        self.current_distance = 0
        self.current_loss = 0

    def get_reward(self):
        """Calculate Huber loss and Euclidean distance and assign rewards accordingly"""
        # start at 0 points
        reward = 0
        pose = torch.FloatTensor(self.sim.pose).to(device)
        target = torch.FloatTensor(self.target_pos).to(device)

        distance = torch.dist(pose,target)
        loss = self.loss_func(pose, target)

        #print('old loss {}'.format(self.current_loss))
        #print('old distance {}'.format(self.current_distance))
        #print('new loss {}'.format(loss.item()))
        #print('new distance {}'.format(distance.item()))

        # give a point for closing the distance or subtract a point for missing
        if distance.item() < self.current_distance:
            reward += 1
            reward = reward + (self.current_distance - distance.item())
        else:
            reward -= 1
            reward = reward - (distance.item() - self.current_distance)

        # give a point for improving loss or subtract a point for missing
        if loss.item() < self.current_loss:
            reward += 1
            reward = reward + (self.current_loss - loss.item())
        else:
            reward -= 1
            reward = reward - (loss.item() - self.current_loss)

        # set loss and distance for next round
        self.current_loss = loss.item()
        self.current_distance = distance.item()

        #reward = 1.-.3*(abs(self.sim.pose - self.target_pos)).sum()
        return reward


    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state

class TakeOffTask():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation, lets always start on ground
        runtime = 5
        init_pose = np.array([0., 0., 0., 0., 0., 0.])
        init_velocities = None
        init_angle_velocities = None

        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal to take off straight
        self.target_pos = np.array([0., 0., 10., 0., 0., 0.])

        self.loss_func = nn.SmoothL1Loss()
        self.current_distance = 0
        self.current_loss = 0

    def get_reward(self):
        """Calculate Huber loss and Euclidean distance and assign rewards accordingly"""
        reward = 0
        pose = torch.FloatTensor(self.sim.pose).to(device)
        target = torch.FloatTensor(self.target_pos).to(device)

        distance = torch.dist(pose, target)
        loss = self.loss_func(pose, target)

        #print('old loss {}'.format(self.current_loss))
        #print('old distance {}'.format(self.current_distance))
        #print('new loss {}'.format(loss.item()))
        #print('new distance {}'.format(distance.item()))

        # give a point for closing the distance
        if distance.item() < self.current_distance:
            reward += 1

        # give a point for improving loss
        if loss.item() < self.current_loss:
            reward += 1

        # set loss and distance for next round
        self.current_loss = loss.item()
        self.current_distance = distance.item()

        #reward = 1.-.3*(abs(self.sim.pose - self.target_pos)).sum()
        return reward


    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state

