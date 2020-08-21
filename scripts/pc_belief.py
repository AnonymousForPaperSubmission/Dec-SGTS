#!/usr/bin/env python
# coding=utf-8

import numpy as np
import random
import math
import sys
import copy
import rospy
import pdb


class PC:
    #intention is the distribution 
    def __init__(self):
        self._num_robots =3
        self._robot_ID = 0
        self._distributions = [[]]#_distributions[robot_ID][joint_path_ID]=probability
        self._paths=[[[]]]#_paths[robot_ID][paths][indices]=pose_ID
        message_ids = 0 #stores the message_id for the last message received from each robot (message_ids are set by teh robot that it belongs to)
        
        
    def drawRandomRealisation(self, paths, path_indices):
        pass
    
    def normaliseDistribution(self, distribution):
        pass


