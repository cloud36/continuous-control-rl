# Actor-Critic Methods Overview 

Actor-critic methods aim to take advantage of both policy based methods and value based methods. Generally speaking, one can think of policy based methods as Monte Carlo (MC) methods as they take a full episode or many before updated the parameters. Additionally, one can think of Q-networks as temporal difference (TD) methods as the action-values or q-values are, generally, updated at every step of an episode. Furthermore, it is well known that MC methods have high variance, but no bias whereas TD methods have no bias, but high variance. So, actor-critic methods aim to combine the best of both worlds. In actor-critic methods, policy based methods (actors) are used to select actions and value based methods (critics) are used to calculate an advantage function (Q(s,a) - V(a)) which are used to train the actor networks. 

## DDPG Overview

Deep Deterministic Policy Gradient (DDPG) was first proposed by TP Lillicrap et al in 2015 and is described as a model-free, off-policy actor-critic algorithm using deep function approximators that can learn policies in high-dimensional, continuous action spaces. 

### Algorithm

![GitHub Logo](https://github.com/cloud36/continuous-control/blob/master/continuous-control/ddpg.png)

**Actor**: As we can see the actor is a policy based network, that picks actions given a state and weights from a DNN. 

**Critic**: The critic is a standard DQN, that assigns values to a state/action pair according to the weights of a DNN.  

**Critic-Update**: The critic is updated by minimizing the loss between Q-Targets and Q-Expected:
* Q-Target  = rewards + gamma * Q(state, p_i( state | target_actor_dnn) | target_critic_dnn)
* Q-Expect = Q(state, action | local_critic_dnn)

**Actor-Update**: The actor is updated by maximizing the value given by the critic network where actions are given by the local actor
* Value = Average( Q(state, p_i(state | local_actor_dnn))
* Minimize(-value) i.e. the same as maximizing Maximize(value) 

**Important Considerations**: There are many important considerations to take into account when implementing DDPG outlined below. 
* Replay Buffer
    * As in DQN, a replay buffer is needed to decorrelate sequential s, a, r, a pairs making it possible to update weights of a DNN without divergence. 
        
* Local and Target Networks
    * As in DQN, local and target networks are used again. This helps prevent against divergence when learning -- if we were to use on network then both the q-target and q-estimate would change when updating network parameters, which is equivalent to chasing a moving target. 
* Soft updates to target networks
    * Unlike in the original DQN paper, soft updates are used for local and target networks. Soft updates slowly change the parameters of a target network to track that of a local network whereas hard updates would simply copy the parameter weights every N steps. 
    Batch Normalization 
        - This technique normalizes each dimension across the samples in a minibatch to have unit mean and variance.
    Exploration 
        - This adds noise to a continuous action space by adding N (noise) to the actions selected by the actor policy i.e. p_i(state | local_actor_dnn) + N where N = noise process.

As we can see, there are many hyperparameters to manage here. Not just in the DDPG algorithm itself, but also in the DNN’s as well e.g. batch normalization.  Below, we can see the various hyperparameters and what the final version used to solve the Reacher Environment. 

### HyperParameters 

    BUFFER_SIZE = int(1e6)     # replay buffer size
    BATCH_SIZE = 256           # minibatch size
    GAMMA = 0.99               # discount factor
    TAU = 1e-3                 # for soft update of target parameters
    LR_ACTOR = 1e-3            # learning rate of the actor
    LR_CRITIC = 1e-3           # learning rate of the critic
    WEIGHT_DECAY = 0           # L2 weight decay
    UPDATE_EVERY = 20          # timesteps between updates
    NUM_UPDATES = 10           # num of update passes when updating
    EPSILON = 1.0              # epsilon for the noise process added to the actions
    EPSILON_DECAY = 1e-6       # decay for epsilon above
    NOISE_SIGMA = 0.01         # sigma for Ornstein-Uhlenbeck noise

### Network Architecture 

### Results

### Improvements
* Google Cloud Hyperparameter Search
* PPO
* A3C