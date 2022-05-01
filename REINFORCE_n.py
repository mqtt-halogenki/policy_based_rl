import torch
import numpy as np
from torch import nn
import gym
from torch import optim
import itertools

class REINFORCEAgent(object):
    def __init__(self,learning_rate, input_size, n_actions=2):
        self.gamma = 0.99
        self.lr =learning_rate
        self.input_size = input_size
        self.n_actions = n_actions
        self.trajectory = []
        self.policy_network = self.get_network()
        self.action_space = [i for i in range(n_actions)]
        self.traces_batch = []
        self.optimizer = optim.Adam(self.policy_network.parameters(),lr=self.lr)

    def get_network(self):
        model = nn.Sequential(
            nn.Linear(self.input_size, 16),
            nn.ReLU(),
            nn.Linear(16, self.n_actions),
            nn.Softmax(dim=0))
        return model
    def predict(self,s):
        action_probs = self.policy_network(torch.FloatTensor(s))
        return action_probs

    def get_action(self, s):
        action_probs = self.predict(s)
        a = np.random.choice(self.action_space, p=action_probs.detach().numpy())
        return a

    def store_transition(self, s, a, r):
        self.trajectory.append((s,a,r))

    def add_to_batch(self):
        self.traces_batch.append(self.trajectory)

    def discount_rewards(self,rewards):
        rewards =[self.gamma ** x for x in rewards]
        rewards = np.flip(list(itertools.accumulate(rewards)), axis=0)
        return rewards


    def learn(self):
        losses = []
        for trace in self.traces_batch:
            rewards = []
            actions = []
            states = []

            states.extend([t[0] for t in trace])
            actions.extend([t[1] for t in trace])
            rewards.extend(self.discount_rewards([t[2] for t in trace]))

            self.optimizer.zero_grad()
            state_tensor = torch.FloatTensor(states)
            reward_tensor = torch.FloatTensor(rewards)
            action_tensor = torch.LongTensor(actions)

            log_action_probs = torch.log(self.predict(state_tensor))
            selected_logprobs = reward_tensor * log_action_probs[np.arange(len(action_tensor)), action_tensor]
            loss = -selected_logprobs.sum()
            losses.append(loss)

        mean_l = torch.mean(torch.stack(losses))
        mean_l.backward()
        self.optimizer.step()
        self.trajectory = []
        self.traces_batch = []

def REINFORCE_algorithm(num_episodes,learning_rate):
    env = gym.make('CartPole-v1')
    s = env.reset()
    done = False
    learning_rate = learning_rate
    reinforce_agent = REINFORCEAgent(learning_rate,env.observation_space.shape[0], env.action_space.n)

    scores = []
    num_episodes = num_episodes
    #num of trace samples M
    for z in range(1):
        for ep in range(num_episodes):
            print(ep)
            for trace in range(0,2):
                s = env.reset()
                score = 0
                done = False
                while not done:
                    a = reinforce_agent.get_action(s)
                    s1, r, done, _ = env.step(a)
                    reinforce_agent.store_transition(s,a,r)
                    s = s1
                    score += 1
                reinforce_agent.add_to_batch()
            scores.append(score)
            reinforce_agent.learn()
    return scores

