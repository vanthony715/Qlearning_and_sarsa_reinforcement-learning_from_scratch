# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Introduction to Machine Learning

Description: Project 5 - Bellman Calculations
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from algorithms.kinematics import Kinematics
import random
import math
import time

class ValueIteration:

    '''
    Uses Belman's Equations for Value iteration
    '''
    def __init__(self, track, start, terminal_states, finishline, GAMMA, THRESHOLD,
                 VELOCITIES, ACCELERATIONS, TRANS_ITERATIONS):
        ##initialize the track
        self.track = track
        self.start = start
        self.terminal_states = terminal_states
        self.finishline = finishline
        self.GAMMA = GAMMA
        self.THRESHOLD = THRESHOLD
        self.VELOCITIES = VELOCITIES
        self.ACCELERATIONS = ACCELERATIONS
        self.TRANS_ITERATIONS = TRANS_ITERATIONS
        self.track_template = np.copy(track)

    def _getTransitionArray(self, p_stay, p_u, p_ur, p_r, p_dr, p_d, p_dl, p_l, p_ul):

        '''
        Desc: Iterates through each space and calculates the transition matrix

        Output: the transition matrix
        '''
        ##define the track dimensions
        dimensions = self.track.shape[0]**2

        ##index map of the transition matrix
        idx_map = {i + 1: (i // self.track.shape[0], i % self.track.shape[0]) for i in range(dimensions)}

        ##transisiton array
        self.P = np.zeros((dimensions, dimensions))

        ##iterate through each state
        for i in tqdm(range(dimensions)):

            ##agent's current position
            x0, y0 = idx_map[i + 1]
            for j in range(dimensions):
                ##agent's next position
                x1, y1 = idx_map[j + 1]

                ##calculate the differences for each time step
                delta_x = x0 - x1
                delta_y = y0 - y1

                ##if not a crash state
                if self.track[x1, y1] == '.':

                    if delta_x == 0:

                        ##stays at the same place
                        if delta_y == 0:
                            self.P[i, j] = p_stay

                        ##goes up
                        if delta_y == 1:
                            self.P[i, j] = p_u

                        ##goes down
                        if delta_y == -1:
                            self.P[i, j] == p_d

                    ##x moves to the left
                    if delta_x == -1:

                        ##goes left
                        if delta_y == 0:
                            self.P[i, j] = p_l

                        ##goes up and to the left
                        if delta_y == 1:
                            self.P[i, j] = p_ul

                        ##goes down and to the left
                        if delta_y == -1:
                            self.P[i, j] == p_dl

                    if delta_x == 1:

                        ##goes right
                        if delta_y == 0:
                            self.P[i, j] = p_r

                        ##goes up and to the right
                        if delta_y == 1:
                            self.P[i, j] = p_ur

                        ##goes down and to the right
                        if delta_y == -1:
                            self.P[i, j] == p_dr
        return self.P

    def _transIterator(self):

        '''
        Desc: Iterates through episodes

        Output: Transition probabilities
        '''
        return np.linalg.matrix_power(self.P, self.TRANS_ITERATIONS)

    def _applyMask(self, state_vals):

        '''
        Desc: Converts all values that are 0 to -10 for state values except for terminal states

        Output: masked state values
        '''
        ##run all zero values to -10 these are the crash values
        state_vals[state_vals == 0] = -10

        ##terminal states are turned to 0
        for x in self.finishline[0]:
            for y in self.finishline[1]:
                state_vals[x, y] = 0
        return state_vals

    def _estimateStateVals(self, Pn):

        '''
        Desc: Estimates state values

        Output: State value matrix
        '''
        ##initialize values to hold state values
        v = np.zeros(self.track.shape[0]**2 + 1)

        ##this is the threshold that initiates a stop
        max_change = self.THRESHOLD

        iterations = 0
        
        while max_change >= self.THRESHOLD:

            if iterations % 2 == 0:
                print("Train Iteration:", iterations)
            
            ##this is the change that will stop the state value iterations
            max_change = 0
            for s in range(self.track.shape[0]**2):
                v_new = 0

                ##sum the rewards and values for each new state
                for s_prime in range(self.track.shape[0]**2):
                    r = - 1 * (s_prime not in self.terminal_states)
                    v_new += Pn[s, s_prime] * (r + self.GAMMA*v[s_prime])

                ##check that the change has not dipped below the threshold value
                max_change = max(max_change, np.abs(v[s] - v_new))
                v[s] = v_new

            iterations += 1
        return np.round(v, 2)

    def train(self):

        '''
        Desc: Trains network

        Output: State values

        '''
        print('\n Value Iteration')
        ##get the initial transition array based on initial probabilities
        self._getTransitionArray(0.5, 0.063, 0.063, 0.063, 0.063,
                               0.063, 0.063, 0.063, 0.063)

        ##unpack start values
        x_start, y_start = self.start

        ##translate start value of track to array values
        start_idx = (self.track.shape[0] * y_start) + x_start

        ##get q value by setting an array of zeros the size of the track
        q = np.zeros(self.track.shape[0]**2)
        q[start_idx] = 1

        ##iterate n times to generate Pn
        Pn = self._transIterator()
        np.round(np.matmul(q, Pn), 5).reshape(self.track.shape[0], -1)

        ##estimate the state vals then transform to grid array
        self.state_vals = self._estimateStateVals(Pn)
        self.state_vals = self.state_vals[:-1].reshape(self.track.shape[0], -1)

        ##appy a mask to the state_vals
        self.state_vals = self._applyMask(self.state_vals)

        return self.state_vals

    def _getSquare(self, x, y):
        '''
        Desc: Get the square of values around x, y coordinates

        Output: s_prime candidate dictionary
        '''
        square = {'coord': [], 'val': []}
        ##eye
        square['coord'].append((x, y))
        square['val'].append(self.state_vals[x, y])
        ##upper
        square['coord'].append((x, y + 1))
        square['val'].append(self.state_vals[x, y + 1])

        ##upper right
        square['coord'].append((x, y - 1))
        square['val'].append(self.state_vals[x + 1, y - 1])

        ##right
        square['coord'].append((x + 1, y))
        square['val'].append(self.state_vals[x + 1, y])

        ##lower right
        square['coord'].append((x + 1, y - 1))
        square['val'].append(self.state_vals[x + 1, y - 1])

        ##lower
        square['coord'].append((x + 1, y + 1))
        square['val'].append(self.state_vals[x + 1, y - 1])

        ##lower left
        square['coord'].append((x - 1, y - 1))
        square['val'].append(self.state_vals[x - 1, y - 1])

        ##left
        square['coord'].append((x - 1, y))
        square['val'].append(self.state_vals[x - 1, y])

        ##upper left
        square['coord'].append((x - 1, y + 1))
        square['val'].append(self.state_vals[x - 1, y + 1])

        return square


    def _getArgmax(self, square):
        '''
        Desc: Get the max value of the square defined in _getSquare

        Output: argmax of the square
        '''
        max_val = max(square['val'])
        max_val_idx = square['val'].index(max_val)
        next_s = square['coord'][max_val_idx]
        return next_s, max_val, max_val_idx

    def _calcSmallestDistance(self, x, y):

        '''
        Desc: Once a wall has been crashed into, get the nearest moveable space

        Output: Nearest Moveable space
        '''
        move_spaces = np.where(self.track == '.')

        ms_dict = {'coord': [], 'dist': []}
        for x_prime, y_prime in zip(move_spaces[0], move_spaces[1]):

            distance = np.sqrt((x_prime - x)**2 + (y_prime - y)**2)

            ms_dict['coord'].append((x_prime, y_prime))
            ms_dict['dist'].append(distance)

        min_dist = min(ms_dict['dist'])
        min_dist_idx = ms_dict['dist'].index(min_dist)
        min_dist_coord = ms_dict['coord'][min_dist_idx]
        x, y = ms_dict['coord'][min_dist_idx]

        return x, y

    def inference(self):
        crash_x, crash_y = 0, 0

        ##get random start position
        x, y = self.start
        self.track[x, y] = '1'
        
        print('x_start: ', self.start[1], 'y_start: ', self.start[0])

        ##initialize kinematics
        kin_calc = Kinematics()

        ##initialize velocities
        x_dot_0, y_dot_0 = 0, 0

        # ##this is the value that we actually move to
        for i in range(105):
            x_dot_0, y_dot_0 = kin_calc.checkVelConstraints(x_dot_0, y_dot_0)
            # ##iterate through states starting at a random starting position
            # while max_val:
            square = self._getSquare(x, y)
            next_s, max_val, max_val_idx = self._getArgmax(square)

            x1, y1 = next_s

            if x1 > x:
                x_ddot = 1
            elif x1 == x:
                x_ddot = 0
            elif x1 < x:
                x_ddot = -1

            if y1 > y:
                y_ddot = 1
            elif y1 == y:
                y_ddot = 0
            elif y1 < y:
                y_ddot = -1

            if x == x1 and y1 == y:
                new_state_val = round(self.state_vals[x1, y1] - 0.1, 3)
                self.state_vals[x, y] = new_state_val

            if self.track_template[x1, y1] == '#':
                self.track[x1, y1] = 'c'
                x_dot, y_dot = 0, 0
                crash_x, crash_y = x1, y1
                print('Crash at: ', (crash_x, crash_y))
                x1, y1 = self._calcSmallestDistance(next_s[0], next_s[1])
                print('Restart at Beginning Coordinates: ', (y, x))
                self.track_template[x, y] = 'X'
                df = pd.DataFrame(self.track_template)
                time.sleep(0.2)
                # print(df)
                time.sleep(20)

            ##calculate the new velocity
            x_dot = kin_calc.velocity(x_dot_0, x_ddot)
            y_dot = kin_calc.velocity(y_dot_0, y_ddot)

            x_pos, y_pos = kin_calc.distance(x, y, x_dot, y_dot)
            
            if x_pos > 0 and y_pos > 0 and x_pos < self.track.shape[0] and y_pos < self.track.shape[0]:
                x, y = x_pos, y_pos
            else:
                x_pos, y_pos = x1, y1
            
            self.track[x_pos, y_pos] = 'Y'
            
            x, y == x_pos, y_pos
            crash_x, crash_y = 0, 0
            x_dot_0 += x_dot
            y_dot_0 += y_dot
            x_ddot, y_ddot = 0, 0

            # if i % 20 == 0:
            #     print('Smallest Distance From Crash:: ', (x1,y1-1))
            # if i % 11 == 0:
            #     print('Inferencing: ')
            #     self.track_template[x_pos, y_pos] = 'X'
            #     df = pd.DataFrame(self.track_template)
            #     time.sleep(0.2)
            #     print(df)

        return self.state_vals

