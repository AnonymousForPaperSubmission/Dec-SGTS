#!/usr/bin/env python
# coding=utf-8

import numpy as np
import random
import math
import sys
import copy
import rospy
import pdb

'''
class PC:
    #intention is the distribution 
    def __init__(self, self_intention, teammate_intention, iteration):
        self._self_intention = self_intention
        self._teammate_intention = teammate_intention
        self._iteration = iteration
        
    def update_intention():
        print updated_intention
'''

#Do update by sampling the distributions
#This should work in general cases
#However it is an approximation

#global_intention 可以认为是 ACTION_SEQUENCE_PROBABILITY_DICT = {} 订阅到的其他机器人的规划好的动作序列概率分布, {key:robot_ID, value: macro_action_sequence_probability_list}
#Intention:
#   MacroActionSequenceProbability[] macro_action_sequence_probability_list
#   int32 robot_ID
#   int32 time_step


#global_intention在这里是一个字典
#pc_belief和global_intention还是不同的, pc_belief表示一种积分布q,大家都在当前的这种积分布上面做事.
#environment用来evaluateSampleSolution, 用来计算WorldUtility
#pc_belief用来维护一个积分布
def update_intention(robot_ID, pc_belief, environment, iteration):
    print ("update intention")
    alpha = 0.1#a fixted step size for the gradient descent, a constant
    T = pow(math.e, -float(1+iteration)/2.0)
    if(T<0.0001):
        T=0.0001;
    int num_samples = 50
    
    #draw a set of joint sample paths

    sample_paths=[[[]]]    #[sample_ID][robot_ID][pose_ID]
    path_ids=[0]*num_samples
    path_indice =0
    for sample in range(num_samples):
        path_indice = 0#path_indices.clear()
        pc_belief.drawRandomRealisation(sample_paths[sample], path_indice);
        path_ids[sample]=path_indice
    
    #evaluate ojective for each sample
    #sample_paths[sample_ID][robot_ID][pose_ID], each sample is a team joint path
    scores_with_i=[]
    scores_without_i=[]
    for sample in range(num_samples):
        scores_with_i.append(environment.evaluateSampleSolution(sample_paths[sample]))
        copy=[[]]
        #r means robot_ID
        for(r in range(len(sample_paths[sample]))):
            if not r==robot_ID:
                copy(r)=sample_paths[sample][r]                
        scores_without_i.append(environment.evaluateSampleSolution(copy))
    
    #Expected Value
    expected_global_utility=0.0
    expected_global_utility_without_i=0.0
    #cover_probability=[]
    #cover_probability_without_i=[]
    
    for(sample in range(num_samples)):
        expected_global_utility += float(scores_with_i[sample])/num_samples
        expected_global_utility_without_i += float(scores_without_i[sample])/num_samples
        
    expected_local_utility=expected_global_utility - expected_global_utility_without_i
    epsilon=0.0000001    
    
    #calculate_entropy
    sum=0.0
    for j in range(len(pc_belief.distributions[robot_ID])):
        qi_xj = float(pc_belief.distributions[robot_ID][j])
        if qi_xj > epsilon:
            sum+= -qi_xj*math.log(qi_xj)
        else:
            sum+=0.0
            
    S_qi = float(sum)
    
    for sample in range(num_samples):
        i=path_ids[sample]
        qi_xi = pc_belief.distributions[robot_ID][i]#每一个sample是每一个joint_paths, sample等同于索引号i,qi_xi返回的是这个joint_paths
        E_G_xi = scores_with_i[sample]-scores_without_i[sample]
        E_G = expected_local_utility
        
        ln_qi_xi = math.log(qi_xi)
        
        pc_belief.distributions[robot_ID][i]=qi_xi-alpha*qi_xi*( float(E_G-E_G_xi)/float(T) + S_qi + ln_qi_xi)#T is a cooling parameter

        if(pc_belief.distributions[robot_ID][i]>epsilon):
            pass#all good
        else:
            pc_belief.distributions[robot_ID][i]=epsilon
        
        if pc_belief.distributions[robot_ID][i]>1.0:
            pc_belief.distributions[robot_ID][i]=1.0
            
    pc_belief.normaliseDistribution(pc_belief.distributions[robot_ID])
    return expected_global_utility
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
