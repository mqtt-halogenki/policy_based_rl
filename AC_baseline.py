import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import multiprocessing
from sklearn.model_selection import ParameterGrid
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

gamma = 0.98
n_rollout = 10


class ActorCritic_baseline(nn.Module):
    def __init__(self,learning_rate):
        super(ActorCritic_baseline, self).__init__()
        self.data = []
        #self.alpha = alpha
        self.learn_rate = learning_rate
        self.fc1 = nn.Linear(4, 256) #input
        #actor
        self.fc_pi = nn.Linear(256, 2)  # output:l or r
        #critic
        self.fc_v = nn.Linear(256, 1)  # out:one choice
        self.optimizer = optim.Adam(self.parameters(), lr=self.learn_rate)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob_policy = F.softmax(x, dim=softmax_dim)
        return prob_policy

    def v(self, x):
        x = F.relu(self.fc1(x))
        value = self.fc_v(x)
        return value

    def put_data(self, transition):
        self.data.append(transition)
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = transition

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r / 100.0])
            s_prime_lst.append(s_prime)

            #speeds up learning
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(
            a_lst), torch.tensor(r_lst, dtype=torch.float), torch.tensor(
            s_prime_lst, dtype=torch.float), torch.tensor(done_lst, dtype=torch.float)
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)
        pi = self.pi(s, softmax_dim=1)         #probabilities
        pi_a = pi.gather(1, a)
        #output = loss(input, target)
        #output.backward()
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())

        #cross entropy - > softmax and negative log liklihood
        #loss = (torch.log(1/pi[pi_a])) / 1

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

def plot(all_rewards,all_lengths,average_lengths):
    # Plot results
    smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()
    smoothed_rewards = [elem for elem in smoothed_rewards]
    plt.plot(all_rewards)
    plt.plot(smoothed_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    plt.plot(all_lengths)
    plt.plot(average_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Episode Avergae length')
    plt.show()

def evaluate_single(args):
    index, params = args
    print('Evaluating params: {}'.format(params))
    params = {**params, **fixed_params}

    scores = []
    for i in range(N_RUNS):
        solver = qcartpole.QCartPoleSolver(**params)
        score = solver.run()
        scores.append(score)

    score = np.mean(scores)
    print('Finished evaluating set {} with score of {}.'.format(index, score))
    return score

def experiments():

    grid_params = {
        'min_alpha': [0.1, 0.2, 0.5],
        'gamma': [1.0, 0.99, 0.9]
    }

    fixed_params = {
        'quiet': True
    }

    grid = list(ParameterGrid(grid_params))
    final_scores = np.zeros(len(grid))

    print('About to evaluate {} parameter sets.'.format(len(grid)))
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    final_scores = pool.map(evaluate_single, list(enumerate(grid)))

    print('Best parameter set was {} with score of {}'.format(grid[np.argmin(final_scores)], np.min(final_scores)))
    print('Worst parameter set was {} with score of {}'.format(grid[np.argmax(final_scores)], np.max(final_scores)))


def AC_baseline(n_episodes,learning_rate):
    env = gym.make('CartPole-v1')
    model = ActorCritic_baseline(learning_rate)
    print_interval = 20
    score = 0.0
    all_s = []
    all_lengths = []
    average_lengths = []
    step = 0

    for n_epi in range(n_episodes):
        done = False
        s = env.reset()
        score2=0.0

        while not done:
            step +=1

            prob = model.pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample().item()
            s_prime, r, done, info = env.step(a)
            model.put_data((s, a, r, s_prime, done))

            s = s_prime
            score += r
            score2 += r

            if done:

                all_s.append(score2)
                print(score2)
                all_lengths.append(step)
                average_lengths.append(np.mean(all_lengths[-10:]))
                break

            model.train_net()


        if n_epi % print_interval == 0 and n_epi != 0:
            #print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            score = 0.0
    env.close()
    return all_s


