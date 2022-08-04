# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Introduction to Machine Learning

Description: Project 5 - Q-Learning Algorithm
"""
import numpy as np
import operator

class QLearning:

    '''
    Implementation of the Q-learning Algorithm
    '''
    def __init__(self, track, start, terminal_states, finishline, GAMMA, EPS,
                 ALPHA, ITERATIONS, ACCELERATIONS, VELOCITIES):
        ##initialize the track
        self.track = track
        self.start = start
        self.terminal_states = terminal_states
        self.finishline = finishline
        self.GAMMA = GAMMA
        self.EPS = EPS
        self.ALPHA = ALPHA
        self.ITERATIONS = ITERATIONS
        self.ACCELERATIONS = ACCELERATIONS
        self.VELOCITIES = VELOCITIES
        self.track_template = np.copy(track)

    def _getStateActionSpace(self):

        '''
        Desc: Iterates through each space and calculates the transition matrix

        Output: stae and action spaces
        '''
        ##define the track dimensions
        action_space = self.ACCELERATIONS

        ##index map of the transition matrix
        state_space = {i + 1: (i // self.track_template.shape[0], i % self.track_template.shape[0]) for i in range(self.track_template.shape[0])}

        return state_space, action_space

    def _resetState(self):
        '''
        Desc: resets the state to zero

        Output: reset state
        '''
        self.pos = 0
        self.acc = 0
        ##index map of the transition matrix
        return (self.pos, self.acc)

    def _getRandomPolicy(self, states, actions):
        '''
        Desc: Uses a random agent to take actions and learns according to random agent's policy'

        Output: random agent's policy'
        '''
        policy = {}
        na = len(actions)
        for s in states:
            policy[s] = {a: 1/na for a in actions}
            
        return policy

    def _chooseAction(self, state, policy):
        '''
        Desc: chooses an action based on the stae and policy

        Output: action
        '''
        prob_a = policy[state]
        action = np.random.choice(a=list(prob_a.keys()),p=list(prob_a.values()))
        return action

    def _getEpsGreedy(actions, EPS, a_best):
        '''
        Desc: calculates the probability given an action and epsolon

        Output: Probability of a
        '''
        prob_a = {}
        na = len(actions)
        
        for a in actions:
            if a == a_best:
                prob_a[a] = 1 - EPS + EPS/na
            else:
                prob_a[a] = EPS/na
        return prob_a

    def _isTerminal(self, state):
        '''
        Desc: checks if state is terminal

        Output: Bool
        '''
        pos, acc = state
        if pos == 0:
            return True

    def train(self):

        '''
        Desc: Trains network

        Output: agent's policy and Q value

        '''
        ##unpack start values
        x_start, y_start = self.start
        ##translate start value of track to array values
        start_idx = (self.track.shape[0] * y_start) + x_start

        np.random.seed(0)

        states, actions = self._getStateActionSpace()

        Q = {s: {a: 0 for a in actions} for s in states}

        policy = self._getRandomPolicy(states, actions)

        s = self._resetState()
        for i in range(self.ITERATIONS):
            if i % 1000 == 0:
                print("Iteration:", i)
                
            # a_best = max(Q[s].items(), key=operator.itemgetter(1))[0]

            # policy[s] = self._getEpsGreedy(actions, self.EPS, a_best)

            # a = self._chooseAction(s, policy)


            # Q[s][a] += self.ALPHA * (reward + self.GAMMA * max(Q[s_next].values()) - Q[s][a])

            # if done:
            #     s = self._resetState()
            # else:
            #     s = s_next

        # policy = {s: {max(policy[s].items(), key=operator.itemgetter(1))[0]: 1} for s in states}

        # return policy, Q

