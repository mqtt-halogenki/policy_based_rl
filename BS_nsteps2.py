#!/usr/bin/env python
# -*- coding: utf-8 -*

import gym
from numpy import dtype
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

    def n_strap(self,s,r,n,gamma):
        ''''
        calculate Q_n(discount rewards) in n-steps from general policy gradient formulation
        :param s: batch state
        :param r: batch reward
        :param 
        :return: Q as a vetor contains each state rollout result.
        '''

        # r length < 10 then return only 
        # should be simplified
        length_batch = r.shape[0]
        if length_batch < n:
            n = length_batch
            last_item = 0
        else:
            last_item = (gamma**n)*self.v(s[-1]).item()
            
        gamma_vector = torch.tensor([gamma**i for i in range(n)],dtype=torch.float)
        all_gamma_r = torch.mul(gamma_vector,r.reshape(-1))

        # the left part of Q_n
        discount_reward=torch.tensor([torch.sum(all_gamma_r[i:-1]) for i in range(n)],dtype=torch.float)
        # right part
        Q_n = discount_reward.reshape(n,1) + last_item

        return Q_n

    def train(self,gamma,eta,n_depth):
        s, a, r, s_, done = self.make_batch()
        Q = self.n_strap(s=s,r=r,gamma=gamma, n=n_depth)
        
        # value update
        loss_v = F.smooth_l1_loss(self.v(s), Q.detach())

        # policy update
        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1,a)

        loss_p = Q.detach() * (-torch.log(pi_a))

        loss = loss_v + loss_p
        loss.mean().backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


def bootstrap(MAX_EPISODES = 10000,n_depth = 10,gamma=0.98,learning_rate=0.001,eta = 0.98):
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
            
            ac.train(gamma=gamma,eta=eta,n_depth=n_depth)
        
        # reward for plotting
        total_reward.append(epi_reward)

        # flag result
        if episode % 100 == 0 and episode != 0:
            print(f"epidoes: {episode}, reward: {flag_reward/100.}")
            flag_reward = 0.0

    env.close()

    return total_reward
