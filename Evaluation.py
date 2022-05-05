import matplotlib.pyplot as plt
import numpy as np
from REINFORCE_n  import reinforce
from scipy.signal import savgol_filter
import seaborn as sns
import pandas as pd
import argparse
from BS_nsteps2 import bootstrap
from AC_baseline import AC_baseline
from Bootstrapping_basesubtraction import Bootstrapping_basesubtraction

#Source of the plotting code:A1
class LearningCurvePlot:

    def __init__(self, title=None):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Duration')
        if title is not None:
            self.ax.set_title(title)

    def add_lines(self,df):
        self.ax = sns.lineplot(data=df, x="episode", y="score", hue="algorithm")

    def add_curve(self, y, label=None):
        if label is not None:
            self.ax.plot(y, label=label)
        else:
            self.ax.plot(y)

    def set_ylim(self, lower, upper):
        self.ax.set_ylim([lower, upper])

    def save(self, name):
        self.ax.legend()
        self.fig.savefig(name, dpi=300)

def smooth(y, window, poly=1):
    return savgol_filter(y, window, poly)



def average_over_repetitions(n_repetitions,n_episodes,smoothing_widnow,algo):
    learning_rate = 0.0001

    reward_results = np.empty([n_repetitions, n_episodes])  # Result array
    for rep in range(n_repetitions):
        print("Repetition:", rep)
        if algo == "REINFORCE":
            rewards = reinforce(n_episodes,learning_rate=0.02)
        elif algo == 'AC Bootstrapping':
            rewards = bootstrap(n_episodes,learning_rate=0.001)
        elif algo == 'AC Baseline':
            rewards = AC_baseline(n_episodes,learning_rate=learning_rate)
        elif algo == "AC Baseline + Bootstrapping":
            rewards = Bootstrapping_basesubtraction(n_episodes)



        rewards = smooth(rewards, smoothing_widnow)
        reward_results[rep] = rewards

    repetitions_with_indexes = list([list(enumerate(x)) for x in reward_results for t in reward_results])
    flattened = [item for sublist in repetitions_with_indexes for item in sublist]
    df = pd.DataFrame(flattened, columns={'episode',"score"})
    df['algorithm'] = algo
    return df


def get_REINFORCE():
    fig, ax = plt.subplots()
    n_episodes = 1000
    df = average_over_repetitions(30,n_episodes,11,algo="REINFORCE")
    lp = sns.lineplot(data=df, x="episode", y="score")
    fig = lp.get_figure()
    fig.savefig("REINFORCE.png")


def get_comparison_plot():
    plot = LearningCurvePlot()

    n_episodes = 1000
    df1 = average_over_repetitions(30,n_episodes,51, algo="REINFORCE")
    df2 = average_over_repetitions(30,n_episodes,51, algo="AC Bootstrapping")
    df3 = average_over_repetitions(30,n_episodes,51, algo="AC Baseline")
    df4 = average_over_repetitions(30,n_episodes,51,algo="AC Baseline + Bootstrapping")
    bigdf = pd.concat([df1,df2,df3,df4],ignore_index=True)
    bigdf.columns = ['episode','score','algorithm']
    plot.add_lines(bigdf)
    plot.save('algorithm_comparison.png')


get_REINFORCE()
#get_comparison_plot()




def main(args):
    args = args['experiment_name']
    if args == 'REINFORCE':
        get_REINFORCE()
    elif args == 'comparison':
        get_comparison_plot()



if __name__ == "__main__":
    parser = argparse.ArgumentParser("experiment")
    parser.add_argument("--experiment_name", help="Specified experiment will be performed.", type=str)
    args = vars(parser.parse_args())
    main(args)