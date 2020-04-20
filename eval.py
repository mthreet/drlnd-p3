from unityagents import UnityEnvironment
import torch
import numpy as np
import time

from agent import Agent

env = UnityEnvironment(file_name='Tennis_Linux/Tennis.x86_64')
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]
env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
state = env_info.vector_observations                  # get the current state (for each agent)
agent = Agent(state_size, action_size, 1245)
agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))
num_episodes = 5
scores = np.zeros((num_episodes, num_agents))                          # initialize the score (for each agent)
for i in range(num_episodes):
    while True:
        action = agent.act(state)  # select an action (for each agent)
        action = np.clip(action, -1, 1)                  # all actions between -1 and 1
        env_info = env.step(action)[brain_name]           # send all actions to tne environment
        next_state = env_info.vector_observations         # get next state (for each agent)
        reward = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores[i] += env_info.rewards                         # update the score (for each agent)
        state = next_state                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
avg_score = np.mean(np.max(scores, axis=1))
print('Total score (averaged over agents) this episode: {}'.format(avg_score))
env.close()