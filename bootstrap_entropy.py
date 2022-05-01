#!/usr/bin/env python
# -*- coding: utf-8 -*

import gym
#from numpy import dtype, ndarray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class BS(nn.Module):
    def __init__(self,learning_rate) -> None:
        super().__init__()

        self.conv_layer = nn.Linear(4,256)
        self.pi_layer = nn.Linear(256,2)
        self.v_layer = nn.Linear(256,1)
        self.buffer = []
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self,input,softmax_dim = 0)-> float:

        """
        network pi value function
        :param input: input data for pi network
        :return: 2 probabilitis 
        """
        output1 = F.relu(self.conv_layer(input))
        output2 = self.pi_layer(output1)
        # return probabitliy
        return F.softmax(output2, dim=softmax_dim)

    def v(self,input) -> float:
        output1 = F.relu(self.conv_layer(input))
        output2 = self.v_layer(output1)
        return output2

    def put_into_buffer(self,transition) -> list:
        return self.buffer.append(transition)

    def make_batch(self):
        """
        transfer batch into several lists
        return: batch in tensor datafomat
        """
        s_lst,a_lst,r_lst,s__lst,done_lst = [],[],[],[],[]
        for transition in self.buffer:
            s,a,r,s_,done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r/1.00])
            s__lst.append(s_)
            done_lst.append([0.0 if done else 1.0])

        s,a,r,s_,done = torch.tensor(s_lst,dtype=torch.float),\
                torch.tensor(a_lst),torch.tensor(r_lst, dtype=torch.float),\
                torch.tensor(s__lst, dtype=torch.float),\
                torch.tensor(done_lst, dtype=torch.float)

        self.buffer = []
        return s,a,r,s_,done
  

    def train(self,gamma,eta):
        s, a, r, s_, done = self.make_batch()
        Q = r + gamma*self.v(s_)* done

        # value update
        loss_v = (Q.detach()- self.v(s))**2.
        # policy update
        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1,a)

        # without entropy
        #loss_p = Q.detach() * (-torch.log(pi_a))

        # entropy 
        entropy = Categorical(pi_a).entropy()
        loss_p = Q.detach() * (-torch.log(pi_a)) + eta * entropy

        loss = loss_v + loss_p
        loss.mean().backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


def bootstrap(MAX_EPISODES = 10000,n_depth = 10,gamma=0.99,learning_rate=0.0002,eta = 0.98):
    '''
    main func

    :param MAX_EPISODES: the learning rounds user define
    :param n_depth: for n-bootstrapping method, the number of which future rewards are considered.
    :param gamma: penalty value when calculating the cumulating rewards.
    :param learning_rate: for network optimization parameter
    :param eta: for entropy regularization
    :return: reward
    '''
    env = gym.make('CartPole-v1')
    ac = BS(learning_rate = learning_rate)

    flag_reward = 0.0
    total_reward = []
    
    # error and episodes can count as two conditions to jump out of circulation
    for episode in range(MAX_EPISODES):
        state = env.reset()
        done = False
        epi_reward = 0.0

        while not done:

            # start to sample
            for n in range(n_depth):

                prob = ac.pi(torch.from_numpy(state).float())
                # sample
                action = Categorical(prob).sample().item()

                state_,reward, done,info = env.step(action)
                ac.put_into_buffer((state,action,reward,state_,done))

                state = state_
                # one episode cumulates its all score
                epi_reward += reward
                flag_reward += reward

                if done:
                    break
            
            ac.train(gamma=gamma,eta=eta)
        
        # reward for plotting
        total_reward.append(epi_reward)

        # flag result
        if episode % 100 == 0 and episode != 0:
            #print(f"epidoes: {episode}, reward: {flag_reward/100.}")
            flag_reward = 0.0

    env.close()

    return total_reward
