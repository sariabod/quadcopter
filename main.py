import torch
import torch.nn as nn
import torch.optim as optim

from agents.agent import ValueNetwork, PolicyNetwork, OUNoise, ReplayBuffer, ddpg_update
from task import LandTask

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

state_dim = 18
action_dim = 4
hidden_dim = 256
action_space = {'low': 0, 'high': 900, 'action': 4}
ou_noise = OUNoise()

value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

target_value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)

for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
    target_param.data.copy_(param.data)

value_lr = 1e-3
policy_lr = 1e-4

value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)

value_criterion = nn.MSELoss()

replay_buffer_size = 1000000
replay_buffer = ReplayBuffer(replay_buffer_size)

max_frames = 1000
max_steps = 500
frame_idx = 0
rewards = []
steps = []
batch_size = 128

env = LandTask()

while frame_idx < max_frames:
    state = env.reset()
    ou_noise.reset()
    episode_reward = 0

    for step in range(max_steps):
        action = policy_net.get_action(state)
        action = ou_noise.get_action(action, step)
        next_state, reward, done = env.step(action)

        replay_buffer.push(state, action, reward, next_state, done)
        if len(replay_buffer) > batch_size:
            ddpg_update(batch_size,replay_buffer, value_net, policy_net, target_policy_net, target_value_net, value_criterion, policy_optimizer, value_optimizer)

        state = next_state
        episode_reward += reward
        frame_idx += 1

        if frame_idx % 10 == 0:
            print('episode reward: {}'.format(episode_reward))
            print('steps: {}'.format(step))

        if done:
            print('iis done')
            rewards.append(episode_reward)
            steps.append(step)
            break