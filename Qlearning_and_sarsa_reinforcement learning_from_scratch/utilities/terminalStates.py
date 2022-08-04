# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Introduction to Machine Learning

Description: Project 5 - Helper functions
"""

import numpy as np

class TerminalStates:

    '''
    Desc: Helper functions
    '''

    def __init__(self, track):
        self.track = track


    def findStart(self):

        '''
        Desc: Find the agent's start position'

        Output: array
        '''
        return np.where(self.track == 'S')

    def findEnd(self):

        '''Function return an array of indices that contains values == S'''

        return np.where(self.track == 'F')