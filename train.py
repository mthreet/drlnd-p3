from unityagents import UnityEnvironment
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from agent import Agent


env = UnityEnvironment(file_name='Tennis_Linux/Tennis.x86_64')
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
# size of each action
action_size = brain.vector_action_space_size
# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]

agent = Agent(state_size, action_size, 10)
# agent.actor_local.load_state_dict(torch.load('checkpoint_actor_save.pth'))
# agent.critic_local.load_state_dict(torch.load('checkpoint_critic_save.pth'))

def ddpg(n_episodes=10000, max_t=10000, print_every=100):
    scores = []
    scores_deque = deque(maxlen=print_every)
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        agent.reset()
        state = env_info.vector_observations            # get the current state
        score = np.zeros(num_agents)
        for t in range(max_t):
            action = agent.act(state)  # select an action
            env_info = env.step(action)[brain_name]             # send the action to the environment
            next_state = env_info.vector_observations        # get the next state
            reward = env_info.rewards                        # get the reward
            done = env_info.local_done                       # see if episode has finished
            agent.step(state, action, reward, next_state, done) # take step with agent (including learning)
            score += reward                                     # update the score
            state = next_state                                  # roll over the state to next time step
            if np.any(done):                                            # exit loop if episode finished
                break
        
        scores_deque.append(np.max(score))       # save most recent score
        scores.append(np.max(score))             # save most recent score

        print('\rEpisode {}\tAverage Score: {:.6f}'.format(i_episode, np.mean(scores_deque)), end="")
        
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.6f}'.format(i_episode, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        
        if np.mean(scores_deque)>=0.5:
            print('Environment solved with average score {:.3f} in {} episodes'.format(np.mean(scores_deque), i_episode))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break
            
    return scores

scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('scores_plot.png')

env.close()