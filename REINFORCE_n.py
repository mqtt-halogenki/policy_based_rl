import gym
import numpy as np
import torch
from torch import nn,optim
class REINFORCEAgent:
    def __init__(self,i_S,o_S,learning_rate):
        self.n_inputs = i_S
        self.n_outputs = o_S
        self.gamma = 0.99
        self.policy_network = self.get_model()
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.01)
        self.states = []
        self.rewards = []
        self.actions = []
        self.learning_rate = learning_rate

    def store(self,ss,rs,sa):
        self.states.extend(ss)
        self.rewards.extend(self.discount_rewards(rs))
        self.actions.extend(sa)

    def clean_buffers(self):
        self.states = []
        self.rewards = []
        self.actions = []

    def predict(self, state):
        action_probs = self.policy_network(torch.FloatTensor(state))
        return action_probs


    def discount_rewards(self,rewards):
        r = np.array([self.gamma**i * rewards[i] for i in range(len(rewards))])
        r = np.flip(r).cumsum()
        return np.flip(r)

    def get_model(self):
        model = nn.Sequential(
            nn.Linear(self.n_inputs, 16),
            nn.ReLU(),
            nn.Linear(16, self.n_outputs),
            nn.Softmax(dim=-1))
        return model
    def choose_action(self,s):
        action_probs = self.predict(s).detach().numpy()
        action = np.random.choice([0,1], p=action_probs)
        return action

def reinforce(num_episodes, learning_rate=0.01):
    env = gym.make('CartPole-v1')

    REINFORCE_agent = REINFORCEAgent(env.observation_space.shape[0], env.action_space.n,learning_rate)

    episode_rewards = []
    traces_c = 1

    for ep in range(num_episodes):
        s = env.reset()
        states = []
        rewards = []
        actions = []
        done = False
        score = 0
        while not done:
            a = REINFORCE_agent.choose_action(s)
            s1, r, done, _ = env.step(a)

            states.append(s)
            rewards.append(r)
            actions.append(a)
            s = s1
            score+=1

            if done:
                REINFORCE_agent.store(states,rewards,actions)
                traces_c += 1
                episode_rewards.append(score)
                score =0
                if traces_c == 10:
                    REINFORCE_agent.optimizer.zero_grad()
                    state_tensor = torch.FloatTensor(REINFORCE_agent.states)
                    reward_tensor = torch.FloatTensor(REINFORCE_agent.rewards)
                    action_tensor = torch.LongTensor(REINFORCE_agent.actions)

                    lp = torch.log(REINFORCE_agent.predict(state_tensor))
                    action_lp = reward_tensor * lp[np.arange(len(action_tensor)), action_tensor]
                    loss = -action_lp.mean()

                    loss.backward()
                    REINFORCE_agent.optimizer.step()

                    REINFORCE_agent.clean_buffers()
                    traces_c = 1

    return episode_rewards