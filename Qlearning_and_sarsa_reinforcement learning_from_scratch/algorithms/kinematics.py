# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Introduction to Machine Learning

Description: Project 5 - Kinematic Equations
"""

class Kinematics:

    '''Kinematic Equations that the Agent Uses to Interact with the Environment'''

    def distance(self, x_0, y_0, x_dot, y_dot):

        '''Calculates the the distance given the initial position and the velocity'''

        return x_0 + x_dot, y_0 + y_dot

    def velocity(self, v_0, acceleration):

        '''
        Desc: Calculates New Velocity given the initial velocity and acceleration

        new velocity
        '''
        return v_0 + acceleration

    def position(self, x_0, y_0, x_1, y_1):
        '''
        Desc: Calculates new position given initial position values and distance traveled

        Output: new position (x, y)
        '''
        return x_0 + x_1, y_0 + y_1

    def checkVelConstraints(self, x_dot, y_dot):
        '''
        Desc: Checks that the velocity do not exceed velocty constraints

        Output: v_dot
        '''
        if x_dot <= -5:
            x_dot = -4
        if x_dot >= 5:
            x_dot = 4
        if y_dot <= -5:
            y_dot = -4
        if y_dot >= 5:
            y_dot = 4
        return x_dot, y_dot