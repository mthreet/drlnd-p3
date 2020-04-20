# Project 3: Collaboration and Competition

## Introduction

This project involved training a pair of tennis racket "agents" to play tennis with each other. The goal for each agent is to keep the tennis ball in the air  by hitting it over the net for as long as possible, without letting the ball hit the ground. The environment allows for access to both agents' state and action spaces, which allows for one model to be used for both agents. One episdode consits of the ball being dropped in to play, the agents rallying, and the ball eventually hitting the ground (or the time limit being reached).

A reward of +0.1 is given to an agent when it hits a ball over the net. A reward of -0.1 is given if an agent lets a ball hit the ground. This task was considered solved when the max score between both agents averages above +0.5 over 100 consecutive episodes. The observation space for each agent is a vector of length 24, containing information on the position and velocity of the ball and racket. The state space is continuous between [-1, 1] for two actions, corresponding to movement forward and backward and "jumping".

This problem was solved by using the Deep Deterministic Policy Gradient (DDPG) algorithm. DDPG is tailored to work well with continous action spaces. DDPG is an acotr-critic method, so it is not as susceptible to a bias-variance tradeoff as a pure policy-based method or value-based method. DDPG uses an actor network to deterministically output the optimal policy for each state, and a critic network to learn the action-value function for each state and optimal action. This allows the acotr and critic network to jointly optimize their weights, which helps avoid a bias-variance tradeoff.

## Installation
Follow the instuctions at [this link](https://github.com/udacity/deep-reinforcement-learning#dependencies) (specifically the "Dependencies" section) for information on installing the environment. **Step 4 about the iPython kernel can be ignored**. This will require Anaconda for Python 3.6 or higher.

After the environment is ready, clone this repository:
```
git clone https://github.com/mthreet/drlnd-p2
```

## Running the code
To run a pretrained model (that received an average score of +13.0 over 100 episodes), simply run [eval.py](eval.py):
```
python eval.py
```

To train a model with [train.py](train.py), simply run:
```
python train.py
```
**Note that this will overwrite the checkpoint unless the save name is changed on line 58 of [train.py](train.py). Line 21 of [eval.py](eval.py) must also then be changed to the new corresponding name.**