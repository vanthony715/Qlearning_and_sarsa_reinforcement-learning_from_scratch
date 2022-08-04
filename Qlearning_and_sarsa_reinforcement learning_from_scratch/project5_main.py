# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877
Class: Introduction to Machine Learning
Description: Project 5 - Main
"""

import gc
import sys, warnings
import time
import pandas as pd
import argparse
import numpy as np

from utilities.terminalStates import TerminalStates
from algorithms.valueIteration import ValueIteration
from algorithms.sarsa import SARSA
from algorithms.qLearning import QLearning

##turn off all warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

gc.collect()

##command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--data_folder_name', type = str ,default = './data',
                    help='Name of the folder where the data and names files are located'),

parser.add_argument('--dataset_name', type = str ,default = '/L.txt',
                    help='Name of the folder where the data and names files are located'),

parser.add_argument('--savefig', type = bool , default = True,
                    help='Name of the folder where the data and names files are located'),

parser.add_argument('--savepath', type = str , default = './R.png',
                    help='Name of the folder where the data and names files are located'),

args = parser.parse_args()

## =============================================================================
##                                  MAIN
## =============================================================================

if __name__ == "__main__":
    ##start timer
    tic = time.time()

    ##read in track
    track = pd.read_csv(args.data_folder_name + args.dataset_name)
    track = np.array(track.values)

    # ##get the start coordinates of the track
    terminal_states = TerminalStates(track)
    startline = terminal_states.findStart()

    # ## out of all possible start positions, get a random start coordinate and store as tuple
    x_start = np.random.choice(startline[0])
    y_start = np.random.choice(startline[1])
    start = (x_start, y_start)

    # ##find the finishline coordinates
    finishline = terminal_states.findEnd()

    ##translate the finishline coordinated to a state number
    terminal_states = []
    for x, y in zip(finishline[1], finishline[0]):
        terminal_states.append(x * track.shape[0] + y + 2)

##=============================================================================
##                      Value Iteration
##=============================================================================

    # ##hyperparameters
    ITERATIONS = 1
    GAMMA = 0.9
    THRESHOLD = 0.1
    ##kinematic constraints
    VELOCITIES = np.arange(-5, 5)
    ACCELERATIONS = np.arange(-1, 2)

    # # ##initialize Bellman method
    value_iterator = ValueIteration(track, start, terminal_states, finishline,
                                    GAMMA, THRESHOLD, VELOCITIES, ACCELERATIONS,
                                    ITERATIONS)
    ##train value iteration algorithm
    state_vals = value_iterator.train()
    square = value_iterator.inference()

##=============================================================================
##                      SARSA
##=============================================================================
    # ##hyperparameters
    ITERATIONS = 10
    EPS = 0.1
    ALPHA = .01

    ##instantiate sarsa
    sarsa = SARSA(track, start, terminal_states, finishline, GAMMA, EPS, ALPHA,
                  ITERATIONS, ACCELERATIONS, VELOCITIES)

    # policy, Q = sarsa.train()

##=============================================================================
##                      Q-Learning
##=============================================================================

    qlearning = QLearning(track, start, terminal_states, finishline, GAMMA, EPS, ALPHA,
                  ITERATIONS, ACCELERATIONS, VELOCITIES)

    # policy, Q = qlearning.train()

##=============================================================================
##                      SARSA
##=============================================================================

# if args.savefig:
    #     ##get a heatmap of the plot
    #     sns.heatmap(state_vals, linewidth=0.5)
    #     plt.yticks(color='w')
    #     plt.xticks(color='w')
    #     plt.savefig(args.savepath)
    #     plt.show()


    toc = time.time()
    tf = round((toc - tic), 2)
    print('Total Time: ', tf)