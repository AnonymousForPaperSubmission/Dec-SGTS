#!/usr/bin/env python
# coding=utf-8
import numpy as np
import random
import math
import sys
import copy
import rospy
import pdb
from dec_sgts.msg import Decide
from dec_sgts.msg import ActionSequence
from dec_sgts.msg import ActionSequenceProbability
from dec_sgts.msg import Policy
from numpy import * 

SCALAR = math.sqrt(2.0)#balance the exploration and exploitation
#BUDGET = 40000#iterative budget for the tree growth
NODE_NUMBER = 0#number of the environment locations
#the Graph of the environment is represented by the adjacent list
#i.e. NEXT_POSE_DICT
NEXT_POSE_DICT={}#key:pose_ID; value:[next_pose_0, next_pose_1]
NODE_LIST = []#the nodes of the environment
#HORIZON = 150#the time for the robot finish its intrusion
#NOTICE: HORIZON should not be set too large, you should set a suitable HORIZON
HORIZON = 40#the time for the robot finish its intrusion
#FORCE_TREE_EXPLORATION_PROBABILITY = 0.1#probability to force tree exploration
GAMMA = 1.0
ACTION_COVERAGE = 0.95
TOLERATED_ERROR=0.05
GOAL_HISTORY_VISITED_DICT = {}#key: pose_ID; value: visited.
DOOR_DICT={}#key:door pose_ID; value: any
GOAL_NUMBER=0
UNVISITED_GOAL_NUMBER=0

POSITIVE_REWARD = 1.0
NEGATIVE_REWARD = -0.01

INITIAL_ROBOT_POSES = []
INTERCEPT_DEPTH = 5#TODO: INTERCEPT_DEPTH是一个可以调节的参数
BUDGET_OUTSIDE =20
BUDGET_INSIDE = 800#iterative budget for the tree growth
ACTION_SEQUENCE_PROBABILITY_DICT = {}#订阅到的其他机器人的规划好的动作序列概率分布, {key:robot_ID, value: macro_action_sequence_probability_list}
ACTION_SEQUENCE_PROBABILITY_TIMESTEP = {}#保存订阅到的其他机器人动作序列概率分布所对应的时间步，{key:robot_ID, value: time_step}
ALL_ROBOT_POLICIES_DICT={}
ALL_ROBOT_POLICIES_TIMESTEP={}
class EnvMap():
    def __init__(self):
        '''
        self._env = [[]]
                                 (".....#...........#....G"
                                  ".....#.....#.....#....."
                                  ".....#.....#.....#....."
                                  "...........#.....#....."
                                  ".....#.....#..........."
                                  ".#####.....#####.#....."
                                  ".....####.##.....##.###"
                                  ".....#...........#....."
                                  ".....#.....#.....#....."
                                  ".....#.....#.....#....."
                                  "X..........#...........",)
        
                                 (".....#...........#....G................. ....G"
                                  ".....#.....#.....#..........#.....#.....#....."
                                  ".....#.....#.....#....G.....#.....#.....#....."
                                  ".G.........#.....#..........####.##.....##.###"
                                  ".....#.....#.............G..#.....#.....#....."
                                  ".#####.....#####.#.......G..####.##..G..##.###"
                                  ".....####.##.....########.###.....#.....#....."
                                  "G....#...........#..........#.....#.....#..G.."
                                  ".....#.....#.....#.......G..#.....#.....#....."
                                  ".....#.....#.....#..........#.....#.....#....."
                                  "X........G.#..........G..............G........",)
        '''
        
        #in the environment, 0 free, 1 means obstacle, 2 means door, 3 means goal.
        #the subgoals include the obstacle and the goal
        #in the matrix of the self._env, left corner means (0,0), 
        #the 0th vertical column is x axis, 
        #the 0th horizontal column is y axis.

        '''        
        self._env = [[0,0,0,1,0,0,0,1,0,0,0],
                     [0,0,0,2,0,0,0,2,0,3,0],
                     [0,0,0,1,0,3,0,1,0,0,0],
                     [0,0,0,1,1,2,1,1,1,2,1],
                     [0,0,0,1,0,0,0,1,0,3,0],
                     [1,2,1,1,0,0,0,1,0,0,0],
                     [0,0,0,1,0,0,0,1,0,0,0]]

        self._env = [[0,3,0,0,0,1,0,0,0,0,0,2,0,0,0,0,0,1,0,0,0,0,0],
                     [0,0,0,0,0,2,0,0,3,0,0,1,0,0,0,0,0,1,0,0,0,0,0],
                     [0,0,0,0,0,1,0,0,0,0,0,1,0,0,3,0,0,1,0,0,0,0,0],
                     [0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0],
                     [0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,2,3,0,0,0,0],
                     [1,2,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
                     [0,0,0,0,0,1,1,1,2,1,1,1,0,0,0,0,0,1,1,2,1,1,1],
                     [0,0,0,0,0,1,0,0,0,0,0,2,0,0,0,0,0,1,0,0,0,0,0],
                     [0,0,0,0,0,1,0,0,3,0,0,1,0,0,0,0,0,1,0,0,0,0,0],
                     [0,0,0,0,0,1,0,0,0,0,0,1,0,0,3,0,0,1,0,3,0,0,0],
                     [0,0,0,0,0,2,0,0,0,0,0,1,0,0,0,0,0,2,0,0,0,0,0]]
        '''

        self._env = [[0,3,0,0,0,1,0,0,0,0,0,2,0,0,0,0,0,1,0,0,0,0,0],
                     [0,0,0,0,0,2,0,0,3,0,0,1,0,0,0,0,0,1,0,0,0,0,0],
                     [0,0,0,0,0,1,0,0,0,0,0,1,0,0,3,0,0,1,0,0,0,0,0],
                     [0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0],
                     [0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,2,3,0,0,0,0],
                     [1,2,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
                     [0,0,0,0,0,1,1,1,2,1,1,1,0,0,0,0,0,1,1,2,1,1,1],
                     [0,0,0,0,0,1,0,0,0,0,0,2,0,0,0,0,0,1,0,0,0,0,0],
                     [0,0,0,0,0,1,0,0,3,0,0,1,0,0,0,0,0,1,0,0,0,0,0],
                     [0,0,0,0,0,1,0,0,0,0,0,1,0,0,3,0,0,1,0,3,0,0,0],
                     [0,0,0,3,0,2,0,0,0,0,0,1,0,0,0,0,0,2,0,0,0,0,0],
                     [1,2,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
                     [0,0,0,0,0,1,1,1,2,1,1,1,0,0,0,0,0,1,1,2,1,1,1],
                     [0,0,0,0,0,1,0,0,0,0,0,2,0,0,0,0,0,1,0,0,0,0,0],
                     [0,0,0,0,0,1,0,0,3,0,0,1,0,0,0,0,0,1,0,0,0,0,0],
                     [0,0,0,0,0,1,0,0,0,0,0,1,0,0,3,0,0,1,0,3,0,0,0],
                     [0,0,0,3,0,2,0,0,0,0,0,1,0,0,0,0,0,2,0,0,0,0,0]]

        self._x_len = len(self._env)
        self._y_len = len(self._env[0])

class EnvNode():
    def __init__(self):
        self._x = -1
        self._y = -1
        self._pose_ID = -1
        self._is_goal = False
        self._visited = False

#represent the state information of the nodes in the tree
class State():
    def __init__(self, pose_ID, distance):
        self._pose_ID = pose_ID#pose_ID index of the vertex list
        self._is_terminal_state = False
        self._distance=distance
        #self._is_goal=False
        self._goal_visited_dict = {}
        self._unvisited_goal_number = -1
        self._reward=0.0#a reward of the num_of_steps cost from parent node to self, and plus self goal reward.
        self._num_of_steps=0

    '''
    #a copied state using as successor_state by copy.deepcopy
    #update self._goal_visited_dict and self._unvisited_goal_number
    def evolve(self, max_steps, gamma):
        global NEGATIVE_REWARD
        global POSITIVE_REWARD
        global HORIZON
        parent_pose_ID = self._pose_ID
        self._reward=0.0#immediate reward, set it to 0.
        parent_distance = self._distance
        assert parent_distance>=0
        current_discount=1.0
        self._num_of_steps=0
        #test_continue_count=0
        terminal_flag=False
        while(not rospy.is_shutdown()):
            self._pose_ID = next_random_pose(self._pose_ID)
            assert self._unvisited_goal_number>=0
            if self._unvisited_goal_number>0:
                if not subgoal_predicate(self._pose_ID, 0):
                    self._reward+=current_discount * NEGATIVE_REWARD
                else:
                    if (self._pose_ID in self._goal_visited_dict) and (self._goal_visited_dict[self._pose_ID]==False): 
                        self._reward+=current_discount *(NEGATIVE_REWARD+POSITIVE_REWARD)
                        self._goal_visited_dict[self._pose_ID]=True
                        self._unvisited_goal_number-=1
                        assert self._unvisited_goal_number>=0
                    else:
                        self._reward+=current_discount * NEGATIVE_REWARD
            else:#self._unvisited_goal_number==0
                pass
            current_discount*=gamma
            self._distance+=1
            self._num_of_steps+=1

            assert self._distance<=HORIZON
            if (   ( subgoal_predicate(self._pose_ID, 0) and (self._pose_ID != parent_pose_ID) )\
                or ( self.check_is_final() and (self._pose_ID != parent_pose_ID))\
                or ( self._num_of_steps>=max_steps )\
                or ( self._distance>=HORIZON ) ):
                terminal_flag=True
                #when the attemption times of reaching horizon is is larger than a threshold
                #set parent node as fully_expanded.
                break
            else:
                continue
        return terminal_flag
    '''
    
    #predicate whether it is a goal
    def goal_predicate(self):
        global NODE_LIST
        if NODE_LIST[self._pose_ID]._is_goal == True:
            self._is_goal = True
            return True
        else:
            self._is_goal = False
            return False

    '''
    #the subgoals include door and goals
    def subgoal_predicate(self, subgoal_heuristics_type):
        global NEXT_POSE_DICT
        global DOOR_DICT
        global GOAL_HISTORY_VISITED_DICT
        assert (subgoal_heuristics_type>=0) and (subgoal_heuristics_type<=2)
        #type==0, 只有门口的节点和目标点G，符合subgoal heuristics
        if subgoal_heuristics_type == 0:
            current_pose = self._pose_ID
            if (current_pose in GOAL_HISTORY_VISITED_DICT) or (current_pose in DOOR_DICT):
                return True
            else:
                return False
        #type==1, 角落的节点和目标点G，都符合subgoal heuris tics
        elif subgoal_heuristics_type ==1:
            current_pose = self._pose_ID
            next_pose_list = NEXT_POSE_DICT[current_pose]
            if current_pose in GOAL_HISTORY_VISITED_DICT or len(next_pose_list) == 2:
                return True
            else:
                return False

        #typd==2, 靠墙的节点，角落的节点，门口的节点和目标点也算作是subgoal heuristics.
        else:
            current_pose = self._pose_ID
            next_pose_list = NEXT_POSE_DICT[current_pose]
            if current_pose in GOAL_HISTORY_VISITED_DICT or len(next_pose_list) == 3:
                return True
            else:
                return False
    '''

    
    #TODO: check_is_final 扩展到多机器人的时候，会变为一个不确定性的变量，因为是否unvisited_goal_number==0，不只取决于主体自己。
    #all goals have been visited
    def check_is_final(self):
        global UNVISITED_GOAL_NUMBER
        assert self._unvisited_goal_number>=0
        if self._unvisited_goal_number==0:
            return True
        else:
            return False

    def check_is_terminal_distance_state(self):
        global HORIZON
        assert self._distance<=HORIZON
        if self._distance==HORIZON:
            return True
        else:
            return False

#represent the structure information of the tree
class TreeNode():
    def __init__(self, state):
        self._visits = 0.01#to avoid the division error
        self._reward = 0.0
        self._q_value = 0.0
        self._state = state
        self._children = []
        self._parent = None
        self._is_fully_expanded = False
    
    '''
    def update(self, q_value):
        self._q_value = (self._q_value * self._visits + q_value)/(self._visits+1.0)
        self._visits += 1.0
    '''

    def update(self, reward):
        self._reward = (self._reward * self._visits + reward)/(self._visits+1.0)
        self._visits += 1.0
    
    '''
    def uct(self):
        global SCALAR
        visits_parent = self._parent._visits 
        visits_child = self._visits
        return self._q_value + SCALAR * math.sqrt((math.log(visits_parent))/visits_child)
    '''
    
    #reward对于单主体的程序而言, 就是自己的reward; 对于多主体而言, 就是所有人的joint reward.
    def uct(self):
        global SCALAR
        visits_parent = self._parent._visits 
        visits_child = self._visits
        return self._reward + SCALAR * math.sqrt((math.log(visits_parent))/visits_child)

    def check_is_leaf_node(self):
        return len(self._children)==0

    #find the next random pose to expand, for adding child.
    def next_random_pose_to_expand(self):
        global NEXT_POSE_DICT#key:pose_ID; value:[next_pose_0, next_pose_1]
        #current pose
        pose = self._state._pose_ID
        #current poses of self children
        existed_child_pose_set = set()
        next_pose = -1
        for i in range(len(self._children)):
            existed_child_pose_set.add(self._children[i]._state._pose_ID)
            #print "len(list(set(NEXT_POSE_DICT[pose])-existed_child_pose_set)): ",len(list(set(NEXT_POSE_DICT[pose])-existed_child_pose_set))
        next_pose = random.choice(list(set(NEXT_POSE_DICT[pose])-existed_child_pose_set))
        if next_pose == -1:
            print "Error! next_pose is -1 in next_random_pose_to_expand(self)!"
            sys.exit(1)
        else:
            return next_pose
    
    def check_is_fully_expanded(self):
        max_len = len(NEXT_POSE_DICT[self._state._pose_ID])
        current_len = len(self._children)
        assert current_len <= max_len
        if current_len==max_len:
            return True
        else:
            return False

    '''
    def check_is_fully_expanded(self):
        return self._is_fully_expanded
    '''
    
    #walk randomly to the next adjacent 
    def next_random_pose_to_walk(self):
        global NEXT_POSE_DICT#key:pose_ID; value:[next_pose_0, next_pose_1]
        current_pose_ID = self._state._pose_ID
        return random.choice(NEXT_POSE_DICT[current_pose_ID])
    
    #add a child of type TreeNode
    def add_child(self, child_node):
        #temp_node = TreeNode(child_state)
        self._children.append(child_node)
        child_node._parent = self
        return child_node

    #return the child with the best uct()
    def best_uct_child(self):
        assert not self._children ==0
        temp_uct = self._children[0].uct()
        best_uct_child = self._children[0]
        for i in range(len(self._children)):
            if i==0:
                continue
            elif temp_uct > self._children[i].uct():
                continue
            else:
                best_uct_child = self._children[i]
                temp_uct = best_uct_child.uct()
        return best_uct_child

    '''
    #most q_value child
    def best_q_child(self):
        assert not len(self._children) == 0
        temp_q_value = self._children[0]._q_value
        best_q_child = self._children[0]
        for i in range(len(self._children)):
            if i==0:
                continue
            elif temp_q_value > self._children[i]._q_value:
                continue
            else:
                best_q_child = self._children[i]
                temp_q_value = best_q_child._q_value
        return best_q_child
    '''
    
    #return the child with the child with the best reward
    def best_reward_child(self):
        assert not len(self._children) == 0
        temp_reward = self._children[0]._reward
        best_reward_child = self._children[0]
        for i in range(len(self._children)):
            if i==0:
                continue
            elif temp_reward > self._children[i]._reward:
                continue
            else:
                best_reward_child = self._children[i]
                temp_reward = best_rewad_child._reward_value
        return best_reward_child
    
    def most_visits_child(self):
        if len(self._children) == 0:
            print "Error! Without any child in most_visits_child()"
            sys.exit(1)
        else:
            temp_visits = self._children[0]._visits
            most_visits_child = self._children[0]
            for i in range(len(self._children)):
                if i==0:
                    continue
                elif temp_visits > self._children[i]._visits:
                    continue
                else:
                    most_visits_child = self._children[i]
                    temp_visits = most_visits_child._visits
            return most_visits_child

'''
#state所表示的节点编号不会是障碍物。
def subgoal_predicate(pose, subgoal_heuristics_type):
        global NEXT_POSE_DICT
        global DOOR_DICT
        global GOAL_HISTORY_VISITED_DICT
        #To make sure that subgoal_heuristics_type should be larger than 0 and smaller than 2.
        assert (subgoal_heuristics_type>=0) and (subgoal_heuristics_type<=2)

        #type==0, 只有门口的节点和目标点G，符合subgoal heuristics
        if subgoal_heuristics_type == 0:
            current_pose = pose
            if (current_pose in GOAL_HISTORY_VISITED_DICT) or (current_pose in DOOR_DICT):
                return True
            else:
                return False

        #type==1, 角落的节点和目标点G，都符合subgoal heuristics
        elif subgoal_heuristics_type ==1:
            current_pose = pose
            next_pose_list = NEXT_POSE_DICT[current_pose]
            if current_pose in GOAL_HISTORY_VISITED_DICT or len(next_pose_list) == 2:
                return True
            else:
                return False

        #typd==2, 靠墙的节点，角落的节点，门口的节点和目标点也算作是subgoal heuristics。                
        else:
            current_pose = pose
            next_pose_list = NEXT_POSE_DICT[current_pose]
            if current_pose in GOAL_HISTORY_VISITED_DICT or len(next_pose_list) == 3:
                return True
            else:
                return False
'''

'''
EVOLVE_TERMINAL_THRESHOLD=2000
def expand_with_macro_actions(parent_node, max_steps, gamma):
    global THRESHOLD
    global EVOLVE_TERMINAL_THRESHOLD
    assert not parent_node.check_is_fully_expanded()
    temp_count=0
    expanded_node=None
    terminal_distance_count = 0
    while(not rospy.is_shutdown()):
        successor_state=copy_another_state(parent_node._state)
        terminal_distance_flag=successor_state.evolve(max_steps, gamma)
        ######################################################
        if terminal_distance_flag == True:
            terminal_distance_count+=1
        #如果最终的terminal_distance_count到达了一定的阈值，这里主要是控制successor_state在evolve的时候，不够evolve到max_steps的时候，进行一个采样进化。
        if terminal_distance_count==EVOLVE_TERMINAL_THRESHOLD:
            parent_node._is_fully_expanded=True
            return parent_node
        ######################################################
        if (not update_existing_node(parent_node, successor_state)):
            if subgoal_predicate(successor_state._pose_ID,0) and successor_state._pose_ID!=parent_node._state._pose_ID:#to avoid state is a terminal state with max_steps and is not subgoal
                new_node = TreeNode(successor_state)
                assert parent_node._state._unvisited_goal_number>=0
                #if the parent node has no final state, i.e. unvisited_goal_number>=0
                if not parent_node._state.check_is_final():
                    expanded_node= parent_node.add_child(new_node)
                else:
                    #if parent node has final state, return parent node directly without adding any other child.
                    expanded_node= parent_node
        else:
            temp_count+=1
        if temp_count==THRESHOLD:
            break
    parent_node._is_fully_expanded=True
    assert not expanded_node ==None
    return expanded_node
'''
'''
#check whether a child is existed, and update the node info.
def update_existing_node(parent_node, successor_state):
    for child_node in parent_node._children:
        if(child_node._state._pose_ID==successor_state._pose_ID):
            #pdb.set_trace()
            if (child_node._state._distance > successor_state._distance):
                child_node._state=successor_state#replace with better successor_state
                return True
            else:
                return True
    return False

def threshold(action_coverage, tolerated_error):
    assert(action_coverage>0 and action_coverage<1)
    assert(tolerated_error>0 and tolerated_error<1)
    return math.ceil(math.log(tolerated_error)/math.log(action_coverage))
'''
#walk randomly to the next adjacent pose
def next_random_pose(pose):
    global NEXT_POSE_DICT#key:pose_ID; value:[next_pose_0, next_pose_1]
    return random.choice(NEXT_POSE_DICT[pose])

#reward = calculate_reward(rollout_poses, temp_goal_visited_dict, temp_unvisited_goal_number)
#we assume gamma=1.0
def calculate_reward(poses, temp_goal_visited_dict, temp_unvisited_goal_number):
    global POSITIVE_REWARD
    global NEGATIVE_REWARD
    reward = 0.0
    assert temp_unvisited_goal_number>=0
    for i in range(len(poses)):
        #temp_goal_visited_dict is used only once in one rollout, we need not copy.deepcopy()
        if( (goal_predicate(poses[i]) ) and (temp_goal_visited_dict[poses[i]]==False) ):
            assert temp_unvisited_goal_number>0
            temp_unvisited_goal_number-=1
            temp_goal_visited_dict[poses[i]]=True
            reward+=NEGATIVE_REWARD
            reward+=POSITIVE_REWARD
        else:
            assert temp_unvisited_goal_number>=0
            if temp_unvisited_goal_number==0:#if there is no temp_unvisited_goal_number
                pass#it means reward+=0
            else:
                reward+=NEGATIVE_REWARD
    return reward

def goal_predicate(pose):
    global GOAL_HISTORY_VISITED_DICT
    return pose in GOAL_HISTORY_VISITED_DICT

#rollout并不是按照宏观动作进行的rollout, rollout是根据微观动作进行的rollout.
#path: the tree nodes from root to leaf
def rollout(path):
    global POSITIVE_REWARD
    global NEGATIVE_REWARD
    global HORIZON
    global NODE_LIST
    global UNVISITED_GOAL_NUMBER
    global GOAL_HISTORY_VISITED_DICT
    reward=0.0
    #temp_reward
    temp_node_list = path
    leaf_node=path[-1]
    assert leaf_node._state._distance<=HORIZON
    #path[-1]: leaf
    #randomly rollout from the leafnode
    rollout_poses = []
    leaf_distance = path[-1]._state._distance
    assert leaf_distance<=HORIZON
    '''
    #if leaf is the terminal_state
    if leaf_distance==HORIZON:
        reward = 0
        return reward
    '''

    assert leaf_node._state._distance<=HORIZON
    #add the pose_ID from the root to the leaf_node, 加入从根节点到叶子节点的点。
    for i in range(len(path)):
        rollout_poses.append(path[i]._state._pose_ID)
    #add one initial pose of the next pose of leafnode， 加入一个初始位置到叶子节点位置的列表。
    leaf_node_pose = leaf_node._state._pose_ID
    rollout_poses.append(next_random_pose(leaf_node_pose))
    leaf_distance+=1
    while(not rospy.is_shutdown()):
        if leaf_distance==HORIZON:
            break
        rollout_poses.append(next_random_pose(rollout_poses[-1]))
        leaf_distance+=1
    reward = calculate_reward(rollout_poses, copy_another_dict(GOAL_HISTORY_VISITED_DICT), UNVISITED_GOAL_NUMBER)
    return reward

def rollout_for_joint_reward(path, with_shared_intention):
    #print "rollout_for_joint_reward() start!"
    global POSITIVE_REWARD
    global NEGATIVE_REWARD
    global HORIZON
    global NODE_LIST
    joint_reward=0.0
    #temp_reward
    leaf_node=path[-1]
    assert leaf_node._state._distance<=HORIZON
    #path[-1]: leaf
    #randomly rollout from the leaf_node
    
    rollout_poses = []#rollout_poses里面存着类型为Action的actions, actions可能是宏观动作，也可能是微观动作。
    leaf_distance = path[-1]._state._distance
    assert leaf_distance<=HORIZON
    #注释的这一部分可以省略，因为对于joint_reward而言，不一定到了HORIZON就没了奖励了。
    '''
    #if leaf is the terminal_state
    if leaf_distance==HORIZON:
        joint_reward = 0
        return joint_reward
    '''
    assert leaf_node._state._distance<=HORIZON
    '''
    #add one initial pose of the next pose of leafnode pose
    for i in range(len(path)):
        temp_action = Action(is_macro_action=True, pose_ID=path[i]._state._pose_ID, num_of_steps=path[i]._state._num_of_steps)
        rollout_poses.append(temp_action)
    '''
    #path里面存储了所有的节点信息．
    for i in range(len(path)):
        rollout_poses.append(path[i]._state._pose_ID)
    
    leaf_node_pose = leaf_node._state._pose_ID
    if leaf_distance < HORIZON:
        rollout_poses.append(next_random_pose(leaf_node_pose))
        leaf_distance+=1
        #print "in rollout(), hello 0."
        while(not rospy.is_shutdown()):
            assert leaf_distance <=HORIZON
            if leaf_distance==HORIZON:
                break
            rollout_poses.append(next_random_pose(rollout_poses[-1]))
            leaf_distance+=1
        #print "in rollout(), hello 1."
    assert leaf_distance <= HORIZON
    #传入参数:　群体联合动作rollout_joint_poses， 已经访问的目标字典goal_visited_dict.
    #joint_reward = get_joint_reward(rollout_poses, copy_another_dict(path[-1]._state._goal_visited_dict), path[-1]._state._unvisited_goal_number)    
    #with_shared_intention=False
    #print "rollout_poses: ", rollout_poses
    #print "len(rollout_poses): ", len(rollout_poses)
    '''
    total_distance=0
    for i in range(len(rollout_poses)):
        total_distance+=rollout_poses[i]._num_of_steps
    '''
    #print "total_distance: ", total_distance
    #print "before get_joint_reward()"
    #print "with_shared_intention: ", with_shared_intention
    joint_reward = get_joint_reward(rollout_poses, with_shared_intention)
    #print "after get_joint_reward()"
    return joint_reward

'''
def back_propagate(path, reward):
    #path is used only once, we do not need copy.deepcopy().
    temp_node=path[-1]#leaf_node
    temp_reward = reward
    while(not rospy.is_shutdown()):
        temp_reward+=path.pop()._state._reward
        temp_node.update(temp_reward)
        temp_node = temp_node._parent
        #if temp_node is the parent of the root. It does not exist.
        if temp_node == None:
            break
'''
#只要是从树中提取的节点，一定是subgoal节点。
def get_random_robot_poses(robot_ID):
    global INITIAL_ROBOT_POSES
    global HORIZON#这里面的HORIZON主要是对distance的长度有限制。
    returned_poses = []
    initial_robot_pose = INITIAL_ROBOT_POSES[robot_ID]
    returned_poses.append(initial_robot_pose)
    for i in range(HORIZON):
        returned_poses.append(next_random_pose(returned_poses[-1]))
    return returned_poses
    
#因为这部分的get_joint_reward, 是从根节点开始计算的，所以没必要过多地执着于unvisited_goal和unvisited_goal_number
def get_joint_reward(rollout_poses, with_shared_intention):
    global POSITIVE_REWARD
    global NEGATIVE_REWARD
    global SELF_ROBOT_ID
    global ROBOT_NUMBER
    global HORIZON
    temp_all_robots_poses_dict={}
    for i in range(ROBOT_NUMBER):
        if i==SELF_ROBOT_ID:
            temp_all_robots_poses_dict[i]=rollout_poses
        else:
            if (with_shared_intention==True):
                temp_all_robots_poses_dict[i]=sample_from_shared_intention(robot_ID=i)
                #print "len(temp_all_robots_poses_dict[i]): ", len(temp_all_robots_poses_dict[i])
                assert len(temp_all_robots_poses_dict[i])==HORIZON+1
                #TODO: sample_from_shared_intention只跟intercept_depth有关, 而跟后面要获得的其他数据无关.
            else:#(with_shared_intention==False)
                temp_all_robots_poses_dict[i]=get_random_robot_poses(robot_ID=i)
    return calculate_joint_reward(temp_all_robots_poses_dict)
    
def spread_all_robots_poses_dict(all_robots_poses_dict):
    global ROBOT_NUMBER
    global HORIZON
    spreaded_poses_dict = {}
    for i in range(ROBOT_NUMBER):
        spreaded_poses_dict[i]={}
    for robot_ID in range(ROBOT_NUMBER):
        for time_step in range(HORIZON+1):
            '''
            print "len(all_robots_poses_dict[0]): ", len(all_robots_poses_dict[0])
            print "len(all_robots_poses_dict[1]): ", len(all_robots_poses_dict[1])
            '''
            spreaded_poses_dict[robot_ID][time_step]=all_robots_poses_dict[robot_ID][time_step]
    return spreaded_poses_dict

def calculate_joint_reward(all_robots_poses_dict):
    global POSITIVE_REWARD
    global NEGATIVE_REWARD
    global GOAL_HISTORY_VISITED_DICT
    global UNVISITED_GOAL_NUMBER
    global HORIZON
    global ROBOT_NUMBER
    temp_goal_visited_dict=copy_another_dict(GOAL_HISTORY_VISITED_DICT)
    temp_unvisited_goal_number = UNVISITED_GOAL_NUMBER
    joint_reward=0.0
    '''
    #all_robots_poses_dict: key is robot_ID, value is Action list. (Action: _is_macro_action, _pose_ID, _num_of_steps)
    for robot_ID in all_robots_poses_dict:
        for time_step in all_robots_poses_dict:
            if all_robots_poses_dict[robot_ID][time_step]._is_macro_action==True:
                temp_pose_ID = all_robots_poses_dict[robot_ID][time_step]._pose_ID
                if goal_predicate(temp_pose_ID)==True and temp_goal_visited_dict[temp_pose_ID]==False:
    '''
    spreaded_poses_dict = spread_all_robots_poses_dict(all_robots_poses_dict)#spreaded_poses_dict, key_0 is robot_ID, key_1 is time_step, value is pose_ID.
    #spreaded_poses_dict = all_robots_poses_dict#这里的all_robots_poses_dict本身不是二维字典，是一维字典加列表，key is robot_ID, value is robot_poses list
    #transformed_poses_dict = transform_poses_dict(spreaded_poses_dict)
    #TODO: 明天早上对这一段做进一步的检察，然后发现是否有错误．
    '''
    print "spreaded_poses_dict: ", spreaded_poses_dict
    print "len(spreaded_poses_dict[0]): ", len(spreaded_poses_dict[0])
    print "len(spreaded_poses_dict[1]): ", len(spreaded_poses_dict[1])
    print "HORIZON: ", HORIZON
    '''
    for time_step in range(HORIZON):
        for robot_ID in range(ROBOT_NUMBER):
            temp_pose_ID = spreaded_poses_dict[robot_ID][time_step]
            if temp_pose_ID == -1:
                assert temp_unvisited_goal_number>=0
                if temp_unvisited_goal_number==0:
                    return joint_reward#如果所有的目标都访问完了，直接就返回joint_reward.
                else:
                    joint_reward+=NEGATIVE_REWARD#肯定不是goal节点，但是也行走了一个时间步。
                continue
            else:
                if( goal_predicate(temp_pose_ID) and temp_goal_visited_dict[temp_pose_ID]==False):
                    assert temp_unvisited_goal_number>0
                    temp_unvisited_goal_number-=1
                    temp_goal_visited_dict[temp_pose_ID]=True
                    joint_reward+=POSITIVE_REWARD#该节点是Goal节点，获得奖励reward，POSITIVE_REWARD
                    joint_reward+=NEGATIVE_REWARD#获得奖励reward的同时，也获得了一次惩罚。
                else:
                    assert temp_unvisited_goal_number>=0
                    if temp_unvisited_goal_number==0:
                        return joint_reward
                    else:
                        joint_reward+=NEGATIVE_REWARD
    return joint_reward

def spread_all_final_robots_poses_dict(all_robots_poses_list):
    global ROBOT_NUMBER
    global HORIZON
    spreaded_poses_dict = {}
    for i in range(ROBOT_NUMBER):
        spreaded_poses_dict[i]={}
    for robot_ID in range(ROBOT_NUMBER):
        for time_step in range(len(all_robots_poses_list[robot_ID])):
            spreaded_poses_dict[robot_ID][time_step]=all_robots_poses_list[robot_ID][time_step]
    return spreaded_poses_dict

def calculate_final_joint_reward(all_robots_poses_list):
    global POSITIVE_REWARD
    global NEGATIVE_REWARD
    global GOAL_HISTORY_VISITED_DICT
    global UNVISITED_GOAL_NUMBER
    global HORIZON
    global ROBOT_NUMBER

    temp_goal_visited_dict=copy_another_dict(GOAL_HISTORY_VISITED_DICT)
    temp_unvisited_goal_number = UNVISITED_GOAL_NUMBER
    joint_reward=0.0
    '''
    spreaded_poses_dict = spread_all_final_robots_poses_dict(all_robots_poses_list)#spreaded_poses_dict, key_0 is robot_ID, key_1 is time_step, value is pose_ID.
    for time_step in range(HORIZON):
        for robot_ID in range(ROBOT_NUMBER):
            if not time_step in spreaded_poses_dict:
                spreaded_poses_dict[robot_ID][time_step] = -1#数据对齐
    '''
    
    for robot_ID in range(ROBOT_NUMBER):
        for time_step in range(len(all_robots_poses_list[robot_ID])):
            temp_pose_ID = all_robots_poses_list[robot_ID][time_step]
            if temp_pose_ID == -1:
                assert temp_unvisited_goal_number>=0
                if temp_unvisited_goal_number==0:
                    return joint_reward#如果所有的目标都访问完了，直接就返回joint_reward.
                else:
                    joint_reward+=NEGATIVE_REWARD#肯定不是goal节点，但是也行走了一个时间步。
                continue
            else:
                if( goal_predicate(temp_pose_ID) and temp_goal_visited_dict[temp_pose_ID]==False):
                    assert temp_unvisited_goal_number>0
                    temp_unvisited_goal_number-=1
                    temp_goal_visited_dict[temp_pose_ID]=True
                    joint_reward+=POSITIVE_REWARD#该节点是Goal节点，获得奖励reward，POSITIVE_REWARD
                    joint_reward+=NEGATIVE_REWARD#获得奖励reward的同时，也获得了一次惩罚。
                else:
                    assert temp_unvisited_goal_number>=0
                    if temp_unvisited_goal_number==0:
                        return joint_reward
                    else:
                        joint_reward+=NEGATIVE_REWARD
    return joint_reward

#用reward去更新枝杈上面的每一个节点。
def back_propagate(path, reward):
    #path is used onlyonce, we do not need copy.deepcopy().
    temp_node = path[-1]
    while(not rospy.is_shutdown()):
        temp_node.update(reward)
        temp_node = temp_node._parent
        if temp_node == None:
            break


def build_the_environment():
    global NEXT_POSE_DICT
    global NODE_NUMBER
    global NODE_LIST
    global GOAL_HISTORY_VISITED_DICT
    global DOOR_DICT
    global GOAL_NUMBER
    global UNVISITED_GOAL_NUMBER
    env = EnvMap()
    x_len = env._x_len
    y_len = env._y_len
    pose_ID = 0
    #build a node list including all of the locations in the environment map
    for i in range(x_len):
        for j in range(y_len):
            if env._env[i][j] == 0 or env._env[i][j] == 2:
                node = EnvNode()
                node._x = i
                node._y = j
                node._pose_ID = pose_ID
                node._is_goal = False
                NODE_LIST.append(node)
                if env._env[i][j]==2:
                    DOOR_DICT[pose_ID]=False
                pose_ID += 1
            elif env._env[i][j] == 3:
                node = EnvNode()
                node._x = i
                node._y = j
                node._pose_ID = pose_ID
                node._is_goal = True
                GOAL_HISTORY_VISITED_DICT[pose_ID]=False#add goal ID pose_ID to the GOAL_HISTORY_VISITED_DICT[]
                GOAL_NUMBER+=1
                NODE_LIST.append(node)
                pose_ID += 1
            else:#==1, obstacle
                pass

    NODE_NUMBER = len(NODE_LIST)
    UNVISITED_GOAL_NUMBER=GOAL_NUMBER
    print "UNVISITED_GOAL_NUMBER: ", UNVISITED_GOAL_NUMBER
    #build an adjacent list of the graph, i.e. NEXT_POSE_DICT
    for i in range(NODE_NUMBER):
        NEXT_POSE_DICT[i] = []
        for j in range(len(NODE_LIST)):
            x0 = NODE_LIST[i]._x
            y0 = NODE_LIST[i]._y
            x1 = NODE_LIST[j]._x
            y1 = NODE_LIST[j]._y
            #judge whether node j is the neighbor of node i
            if math.sqrt((x0-x1)*(x0-x1)+(y0-y1)*(y0-y1))==1.0:
                NEXT_POSE_DICT[i].append(j)

#g_count = 0 
ALL_TASKS_COMPLETE_BONUS=0.0#set a bonus when all tasks are completed.
'''
def tree_grow(root):
    global HORIZON
    global ALL_TASKS_COMPLETE_BONUS
    path = select(root)#select直接返回root节点到新扩展的叶子节点之间的路径，在select当中包括了expand步骤。
    assert path[-1]._state._distance<=HORIZON#leaf_node:
    if path[-1]._state.check_is_final():#TODO：需要注意新加入一个节点之后对于新加入的节点的state的更新。
        reward=ALL_TASKS_COMPLETE_BONUS
    else:
        reward= rollout(path)#path means the selected nodes from root to leaf, path[-1] means the selected_node
    back_propagate(path, reward)
'''

def tree_grow(root, is_received_other_action_flag):
    global HORIZON
    global ALL_TASKS_COMPLETE_BONUS
    path = select(root)
    assert path[-1]._state._distance<=HORIZON#leaf_node.poses
    #q_value = rollout(path)#path means the selected nodes from root to leaf, path[-1] means the selected_node
    '''
    print "path: "
    for i in range(len(path)):
        print path[i]._state._pose_ID,
    print " "
    '''
    #joint_reward得到的是枝杈上从根节点root到叶子节点leaf(path[-1])的一个整体的reward.
    joint_reward = rollout_for_joint_reward(path, is_received_other_action_flag)#path means the selected nodes from root to leaf, path[-1] means the selected_node
    back_propagate(path, joint_reward)#直接用整条枝杈的joint_reward去控制参数.

#TODO: 等下回来的时候从这里开始写, rollout_for_joint_reward(path, is_received_other_intention_flag)
'''
def select(root):
    global HORIZON
    global GAMMA
    path=[]#node list from root to leaf
    path.append(root)
    current_node = root
    while (( not current_node._state.check_is_final() ) and ( current_node._state._distance<HORIZON) ):
        if(not current_node.check_is_fully_expanded()):
            new_node = expand_with_macro_actions( current_node, HORIZON - current_node._state._distance, GAMMA)
            if(new_node):#not new_node == None
                path.append(new_node)
                assert path[-1]._state._distance<=HORIZON
                return path
            else:
                pass
        else:
            if len(current_node._children)==0:
                break
            else:
                current_node = current_node.best_uct_child()
                path.append(current_node)
    assert path[-1]._state._distance<=HORIZON
    return path
'''
def expand(leaf_node):
    global HORIZON
    #do not expand the fully-expanded node, return leaf_node directly
    if leaf_node.check_is_fully_expanded():
        print "Error! Expand the fully-expanded node."
        sys.exit(1)
    #do not expand the terminal state node, return None
    assert leaf_node._state._distance<=HORIZON
    if leaf_node._state._distance==HORIZON:
        leaf_node._is_fully_expanded = True
        return None
    #if it is not fully_expanded, and is not the terminal state
    else:
        while(not rospy.is_shutdown()):
            child_pose = leaf_node.next_random_pose_to_expand()
            child_distance = leaf_node._state._distance + 1
            '''
            self._pose_ID = pose_ID#pose_ID index of the vertex list
            self._is_terminal_state = False
            self._distance=distance
            #self._is_goal=False
            self._goal_visited_dict = {}
            self._unvisited_goal_number = -1
            self._reward=0.0#a reward of the num_of_steps cost from parent node to self, and plus self goal reward.
            self._num_of_steps=0
            '''
            state = State(child_pose, child_distance)
            state._reward =0.0#如果是整条枝杈的reward用来引导树生长而不是Q Value，那么用不到考虑这么多。
            temp_goal_visited_dict = copy_another_dict(leaf_node._state._goal_visited_dict)
            temp_unvisited_goal_number = leaf_node._state._unvisited_goal_number
            assert temp_unvisited_goal_number>=0
            if( goal_predicate(child_pose) and temp_goal_visited_dict[child_pose]==False  ):
                assert temp_unvisited_goal_number>0
                temp_goal_visited_dict[child_pose]=True
                temp_unvisited_goal_number-=1
            state._goal_visited_dict = temp_goal_visited_dict
            state._unvisited_goal_number = temp_unvisited_goal_number
            state._num_of_steps = 1
            if child_distance < HORIZON:
                pass 
            elif child_distance == HORIZON:
                state._is_terminal_state = True
            else:
                print "Error! child_distance is larger than HORIZON in expand()"
                sys.exit(1)
            child_node = TreeNode(state)
            leaf_node.add_child(child_node)
            if leaf_node.check_is_fully_expanded()==True:
                break
        leaf_node._is_fully_expanded = True
        return random.choice(leaf_node._children)#随机返回其中的一个child


def select(root):
    global HORIZON
    global GAMMA
    path=[]#node list from root to leaf
    path.append(root)
    current_node = root
    while (( not current_node._state.check_is_final() ) and ( current_node._state._distance<HORIZON) ):#and后面第二个判断条件的应该是<，不是<=，因为HORIZON-1的distance的node还可以在扩展一次。
        if(not current_node.check_is_fully_expanded()):
            #new_node = expand_with_macro_actions( current_node, HORIZON - current_node._state._distance, GAMMA)
            new_node = expand(current_node)#如果expand发现是空的节点， 那么就返回一个空的值。
            if(new_node):#not new_node == None
                path.append(new_node)
                assert path[-1]._state._distance<=HORIZON
                return path
            else:
                pass
        else:
            if len(current_node._children)==0:
                break
            else:
                current_node = current_node.best_uct_child()
                path.append(current_node)
    assert path[-1]._state._distance<=HORIZON
    return path

'''
def init_parameters():
    global THRESHOLD
    global ACTION_COVERAGE
    global TOLERATED_ERROR
    THRESHOLD=threshold(ACTION_COVERAGE, TOLERATED_ERROR)
    print "THRESHOLD: ", THRESHOLD
'''
'''
#to test the shot probability of successor_state evolving to subgoals
def test_evolve():
    global GAMMA
    build_the_environment()
    for i in range(len(NODE_LIST)):
        print "pose_ID: ", i, " (x,y) is: ",
        print NODE_LIST[i]._x, NODE_LIST[i]._y
    
    #sys.exit(1)
    print "GOAL_HISTORY_VISITED_DICT: ", GOAL_HISTORY_VISITED_DICT
    print "DOOR_DICT: ", DOOR_DICT
    #sys.exit(1)
    init_parameters()
    
    root_state = State(pose_ID=0, distance=0)#pose 188 means the Graph vertex of (10,0)
    root=TreeNode(root_state)
    root._state._goal_visited_dict = GOAL_HISTORY_VISITED_DICT
    root._state._unvisited_goal_number = UNVISITED_GOAL_NUMBER

    shot_count = 0.0
    total_count=0.0
    min_distance=99999
    final_pose = -1
    
    while(not rospy.is_shutdown()):
        temp_successor_state = copy_another_state(root_state)
        #evolve(max_steps, gamma)
        temp_successor_state.evolve(100, 1.0)
        print "temp_successor_state._pose_ID: ", temp_successor_state._pose_ID
        print "x: ", NODE_LIST[temp_successor_state._pose_ID]._x,
        print " y: ", NODE_LIST[temp_successor_state._pose_ID]._y
        print "temp_successor_state._distance: ", temp_successor_state._distance
        print "subgoal: ", subgoal_predicate(temp_successor_state._pose_ID,0)
        if subgoal_predicate(temp_successor_state._pose_ID,0) and (not temp_successor_state._pose_ID==root_state._pose_ID)==True:
            shot_count+=1
            if min_distance>temp_successor_state._distance:
                min_distance= temp_successor_state._distance
                final_pose=temp_successor_state._pose_ID
        total_count+=1
        if total_count==20:
            print "shot probability: ", shot_count/total_count
            print "min_distance: ", min_distance
            print "final_pose: ", final_pose
            print "x: ", NODE_LIST[final_pose]._x,
            print " y: ", NODE_LIST[final_pose]._y
            break
'''

'''
def test_parallel_expand():
    pass
'''

'''
def test_expand():
    global GAMMA
    global NODE_LIST
    build_the_environment()
    for i in range(len(NODE_LIST)):
        print "pose_ID: ", i, " (x,y) is: ",
        print NODE_LIST[i]._x, NODE_LIST[i]._y
    #sys.exit(1)
    print "GOAL_HISTORY_VISITED_DICT: ", GOAL_HISTORY_VISITED_DICT
    print "DOOR_DICT: ", DOOR_DICT
    #sys.exit(1)
    init_parameters()
    #to test the expand_with_macro_actions()
    sample_state = State(pose_ID=0, distance=0)#pose 188 means the Graph vertex of (10,0)
    sample_node = TreeNode(sample_state)
    sample_node._state._goal_visited_dict = GOAL_HISTORY_VISITED_DICT
    sample_node._unvisited_goal_number = UNVISITED_GOAL_NUMBER
    child_dict={}#key is child pose, value is num_of_steps
    for i in range(1000):#continuously add node to the sample node
        sample_node._state._goal_visited_dict = GOAL_HISTORY_VISITED_DICT
        sample_node._state._unvisited_goal_number = UNVISITED_GOAL_NUMBER
        if sample_node.check_is_fully_expanded():
            break
        new_node = expand_with_macro_actions(sample_node, HORIZON-sample_node._state._distance, GAMMA)
        print "sample_node.check_is_fully_expanded():", sample_node.check_is_fully_expanded()
        #print "new_node: ", new_node
        for i in range(len(sample_node._children)):
            print "cooridnate: (",NODE_LIST[sample_node._children[i]._state._pose_ID]._x,",",NODE_LIST[sample_node._children[i]._state._pose_ID]._y,"): num_of_steps: ",
            print sample_node._children[i]._state._num_of_steps
    sys.exit(1)
'''

def test_expand():
    #TODO：to be completed.
    #在这段代码里， 只有expand这一个需要被修改的代码。
    pass    
    
#to accelerate it, instead of copy.deepcopy()
def copy_another_dict(old_dict):
    new_dict={}
    for i in old_dict:
        new_dict[i]=old_dict[i]
    return new_dict

#to accelerate it, instead of copy.deepcopy()
def copy_another_state(state):
    new_state = State(-1,-1)
    new_state._pose_ID = state._pose_ID
    new_state._is_terminal_state = state._is_terminal_state
    new_state._distance = state._distance
    new_state._goal_visited_dict=copy_another_dict(state._goal_visited_dict)
    new_state._unvisited_goal_number=state._unvisited_goal_number
    new_state._reward=state._reward
    new_state._num_of_steps = state._num_of_steps
    return new_state

'''
def start():
    global GAMMA
    build_the_environment()
    for i in range(len(NODE_LIST)):
        print "pose_ID: ", i, " (x,y) is: ",
        print NODE_LIST[i]._x, NODE_LIST[i]._y

    print "GOAL_HISTORY_VISITED_DICT: ", GOAL_HISTORY_VISITED_DICT
    print "DOOR_DICT: ", DOOR_DICT
    #init_parameters()

    root_state = State(pose_ID=0, distance=0)#pose 188 means the Graph vertex of (10,0)
    root=TreeNode(root_state)
    root._state._goal_visited_dict = GOAL_HISTORY_VISITED_DICT
    root._state._unvisited_goal_number = UNVISITED_GOAL_NUMBER
    
    loop_count = 0
    while(loop_count < BUDGET):
        tree_grow(root)
        loop_count+=1
        print "loop_count: ", loop_count
    
    #a reward of -0.01 is given at every time step, reaching the goal gives a reward of 1 and terminates the episode.
    #to print the track and output the total reward
    total_reward = 0.0
    temp_poses = []
    temp_node = root
    temp_unvisited_goal_number = GOAL_NUMBER
    temp_goal_history_visited_dict = copy_another_dict(GOAL_HISTORY_VISITED_DICT)
    
    while(not rospy.is_shutdown()):
        if (len(temp_node._children)) == 0:
            if goal_predicate(temp_node._state._pose_ID) and temp_goal_history_visited_dict[temp_node._state._pose_ID]==False:
                temp_unvisited_goal_number-=1
                print "temp_node._state._pose_ID: ", temp_node._state._pose_ID
                print "temp_node._state._num_of_steps: ", temp_node._state._num_of_steps
                print "temp_node._state._distance: ", temp_node._state._distance
                print "x: ", NODE_LIST[temp_node._state._pose_ID]._x,
                print "; y: ", NODE_LIST[temp_node._state._pose_ID]._y
                total_reward+=POSITIVE_REWARD
                total_reward+=temp_node._state._num_of_steps*NEGATIVE_REWARD
            else:
                print "temp_node._state._pose_ID: ", temp_node._state._pose_ID
                print "temp_node._state._num_of_steps: ", temp_node._state._num_of_steps
                print "temp_node._state._distance: ", temp_node._state._distance
                print "x: ", NODE_LIST[temp_node._state._pose_ID]._x,
                print "; y: ", NODE_LIST[temp_node._state._pose_ID]._y
                total_reward+=temp_node._state._num_of_steps*NEGATIVE_REWARD
            print "break when len(temp_node._children)==0"
            break
        else:
            temp_poses.append(temp_node._state._pose_ID)
            print "temp_node._state._pose_ID: ", temp_node._state._pose_ID
            print "temp_node._state._num_of_steps: ", temp_node._state._num_of_steps
            print "temp_node._state._distance: ", temp_node._state._distance
            print "x: ", NODE_LIST[temp_node._state._pose_ID]._x,
            print "; y: ", NODE_LIST[temp_node._state._pose_ID]._y
            #assert temp_node._state._pose_ID in temp_goal_history_visited_dict
            if NODE_LIST[temp_node._state._pose_ID]._is_goal==True and temp_goal_history_visited_dict[temp_node._state._pose_ID]==False:
                total_reward+=POSITIVE_REWARD
                total_reward+=NEGATIVE_REWARD
                temp_goal_history_visited_dict[temp_node._state._pose_ID]=True
                temp_unvisited_goal_number-=1
                assert temp_unvisited_goal_number>=0
                if temp_unvisited_goal_number==0:
                    #print "break when temp_unvisited_goal_number is 0."
                    break
            else:
                total_reward+=temp_node._state._num_of_steps*NEGATIVE_REWARD
            temp_node = temp_node.most_visits_child()

    print "len(temp_poses): ", len(temp_poses)
    print "total_reward: ", total_reward
'''

def decide_callback(decide_msg):
    #print "decide callback start!"
    global ACTION_SEQUENCE_PROBABILITY_DICT
    global ACTION_SEQUENCE_PROBABILITY_TIMESTEP
    action_sequence_probability_list = []
    for i in range(len(decide_msg.action_sequence_probability_list)):
        action_sequence = list(decide_msg.action_sequence_probability_list[i].action_sequence)
        #print "action_sequence in decide_callback: ", action_sequence
        probability = decide_msg.action_sequence_probability_list[i].probability
        #print "probability in decide callback: ", probability 
        #is_cycle = decide_msg.action_sequence_probability_list[i].is_cycle
        #print "is_cycle in decide callback: ", is_cycle
        action_sequence_probability_list.append([action_sequence, probability])
    time_step = decide_msg.time_step
    #print "The received action_sequence_probability_list time_step is: ", time_step
    robot_ID = decide_msg.robot_ID
    ACTION_SEQUENCE_PROBABILITY_DICT[robot_ID] = action_sequence_probability_list
    #print "ACTION_SEQUENCE_PROBABILITY_DICT[robot_ID] in decide_callback: ", ACTION_SEQUENCE_PROBABILITY_DICT[robot_ID]
    #print "the decide callback wall time in 325: ", rospy.get_time(), "ACTION_SEQUENCE_PROBABILITY_TIMESTEP is: ", ACTION_SEQUENCE_PROBABILITY_TIMESTEP
    ACTION_SEQUENCE_PROBABILITY_TIMESTEP[robot_ID] = time_step

def pub_decide_result(action_sequence_probability_list, robot_ID, time_step):
    pub = rospy.Publisher('/decide_result', Decide, queue_size=10)
    decided_result = Decide()
    for i in range(len(action_sequence_probability_list)):
        action_sequence_probability = ActionSequenceProbability()
        action_sequence_probability.action_sequence= action_sequence_probability_list[i][0]
        action_sequence_probability.probability = action_sequence_probability_list[i][1]
        #action_sequence_probability.is_cycle = action_sequence_probability_list[i][2]
        decided_result.action_sequence_probability_list.append(action_sequence_probability)
    #decided_result.action_sequence_probability_list = action_sequence_probability_list
    decided_result.robot_ID = robot_ID
    decided_result.time_step = time_step
    pub.publish(decided_result)

def check_received_other_action_flag(time_step):
    #print "check_received_other_action_flag"
    global ACTION_SEQUENCE_PROBABILITY_TIMESTEP#key:robot_ID, value: time_step
    #ACTION_SEQUENCE_PROBABILITY_DICT
    global ROBOT_NUMBER
    flag = True
    for i in range(ROBOT_NUMBER):
        if not i in ACTION_SEQUENCE_PROBABILITY_TIMESTEP:
            flag = False
            break
        elif ACTION_SEQUENCE_PROBABILITY_TIMESTEP[i]==-1 or ACTION_SEQUENCE_PROBABILITY_TIMESTEP[i] < time_step:
            flag = False
            break
        elif ACTION_SEQUENCE_PROBABILITY_TIMESTEP[i] > time_step:
            #print "the error wall time in 881", rospy.get_time()
            print "Error! ACTION_SEQUENCE_PROBABILITY_TIMESTEP[i] is ", ACTION_SEQUENCE_PROBABILITY_TIMESTEP[i]
            print "Error! time_step is ", time_step
            print "Error! ACTION_SEQUENCE_PROBABILITY_TIMESTEP[i] > time_step"
            sys.exit(1)#TODO:很有可能是ACTION_SEQUENCE_PROBABILITY_TIMESTEP
    return flag

#TODO:to imitate the above
def policy_callback(policy_msg):
    global ALL_ROBOT_POLICIES_DICT
    global ALL_ROBOT_POLICIES_TIMESTEP
    global SELF_ROBOT_ID
    print "policy_callback start in robot ", SELF_ROBOT_ID, "!"
    ALL_ROBOT_POLICIES_DICT[policy_msg.robot_ID] = policy_msg.action_sequence
    ALL_ROBOT_POLICIES_TIMESTEP[policy_msg.robot_ID] = policy_msg.time_step

def pub_policy(action_sequence, time_step):
    global SELF_ROBOT_ID
    policy_msg = Policy()
    policy_msg.time_step = time_step
    policy_msg.action_sequence = action_sequence
    policy_msg.robot_ID = SELF_ROBOT_ID
    pub_policy = rospy.Publisher("/generated_policy", Policy, queue_size=10) 
    pub_policy.publish(policy_msg)
    print "The generated policy was published successfully in robot ", SELF_ROBOT_ID, "!"

#random pick an item from a list according to the probability of this item.
def random_pick(some_list,probabilities):
    x=random.uniform(0,1)
    cumulative_probability=0.0
    for item,item_probability in zip(some_list,probabilities):
        cumulative_probability+=item_probability
        if x < cumulative_probability:
            return item

#random pick an action sequence from a list according to the probability of this action sequence.
def random_pick_from_action_sequence_probability_list(probability_list):
    action_sequences = []
    probabilities = []
    for i in range(len(probability_list)):
        action_sequences.append( probability_list[i][0] )#probability_list[i][0]在这里本身就是一个action sequence
        probabilities.append(probability_list[i][1])
    return random_pick(action_sequences, probabilities)

#input: robot_ID
#output: other robots' poses
def sample_from_shared_intention(robot_ID):
    global ACTION_SEQUENCE_PROBABILITY_DICT
    global HORIZON
    action_sequence_probability_list = ACTION_SEQUENCE_PROBABILITY_DICT[robot_ID]
    selected_intention = random_pick_from_action_sequence_probability_list( action_sequence_probability_list )
    robot_poses = []
    #print "selected_intention: ", selected_intention
    robot_poses = selected_intention#selected_intention表示的是一个ActionSequence
    '''
    print "is cycle in sample_from_shared_intention():", selected_intention[1]
    print "len(robot_poses) in sample_from_shared_intention(): ", len(robot_poses)
    print "robot_poses in sample_from_shared_intention(): ", robot_poses
    sys.exit(1)
    '''
    temp_distance=len(robot_poses)-1
    returned_poses=robot_poses
    assert temp_distance<=HORIZON
    if temp_distance==HORIZON:
        pass
    else:
        while(not rospy.is_shutdown()):
            returned_poses.append(next_random_pose(returned_poses[-1]))
            temp_distance+=1
            if temp_distance==HORIZON:
                break
    return returned_poses#返回的就是selected_intention[0], 就是一个ActionSequence

def take_second(elem):
    return elem[1]

#input: root node 
#       intercept depth, the depth of the actions,
#       intercept_length, the length of the action sequence list
#output: action sequence probability list, sorted by the probability in the reverse order, [[[action_sequence], probability]].
#TODO: 可以加入一个策略，截取前百分之多少的动作。
#TODO: 这部分暂时采用reward来控制.
def get_action_sequence_probability_list(root, intercept_depth, intercept_length=None):
    list_parent = []#用于搜索出所有的动作序列的list_parent，是临时的父亲节点变量
    list_children = []#用于搜索出所有的动作序列的list_children，是临时的孩子节点变量
    if intercept_depth == 1:
        for i in range(len(root._children)):
            list_children.append(root._children[i])#先加入第一层的孩子节点
    else:
        for i in range(len(root._children)):
            list_parent.append(root._children[i])#先加入第一层的孩子节点
        #计算树中长度为intercept_depth的所有的枝杈末端的节点。
        for i in range(int(intercept_depth)-1):
            list_children = []
            while(not len(list_parent)==0):
                for i in range(len(list_parent[0]._children)):
                    list_children.append(list_parent[0]._children[i])
                del(list_parent[0])
            list_parent = list_children    

    #print "len(list_children): ", len(list_children)
    reward_list = []
    for i in range(len(list_children)):
        reward_list.append(list_children[i]._reward)
    if len(reward_list)==0:
        print "Error! len(reward_list) is 0 in get_action_sequence_probability_list!"
        sys.exit(1)
    reward_list_min = min(reward_list)
    for i in range(len(reward_list)):
        reward_list[i] = reward_list[i]+reward_list_min
    #so, the reward_list[i] will be >= 0, and the sum(reward_list) will be >= 0
    reward_list_sum = sum(reward_list)
    
    if reward_list_sum > 0:
        action_sequence_probability = reward_list/reward_list_sum
    else:#all of the reward_list[i] is 0, equal probability.
        action_sequence_probability = ones(len(reward_list))/float((len(reward_list)))
    
    action_sequence_list = []
    for i in range(len(list_children)):
        temp_pose_ID_list= []
        temp_node = list_children[i]
        while(not temp_node==None):
            temp_pose_ID_list.append(temp_node._state._pose_ID)
            temp_node=temp_node._parent
        temp_pose_ID_list.reverse()
        action_sequence_list.append(temp_pose_ID_list)
    action_sequence_probability_list = []
    for i in range(len(action_sequence_list)):
        action_sequence_probability_list.append([action_sequence_list[i], action_sequence_probability[i]])
    action_sequence_probability_list.sort(key=take_second, reverse=True)
    #print action_sequence_probability_list

    #print action_sequence_probability_list
    #print "the 0th element is: ", action_sequence_probability_list[0]
    '''
    action_sequence_probability_list = action_sequence_probability_list[0:20]#TODO：只截取十个发送
    temp_reward_sum = 0.0
    for i in range(len(action_sequence_probability_list)):
        temp_reward_sum+=action_sequence_probability_list[i][1]
    for i in range(len(action_sequence_probability_list)):
        action_sequence_probability_list[i][1] = action_sequence_probability_list[i][1]/temp_reward_sum
    '''
    return action_sequence_probability_list

def output_robot_track(root):
    #a reward of -0.01 is given at every time step, reaching the goal gives a reward of 1 and terminates the episode.
    #to print the track and output the total reward
    global GOAL_NUMBER
    global GOAL_HISTORY_VISITED_DICT
    global SELF_ROBOT_ID
    total_reward = 0.0
    temp_poses = []
    temp_node = root
    temp_unvisited_goal_number = GOAL_NUMBER
    temp_goal_history_visited_dict = copy_another_dict(GOAL_HISTORY_VISITED_DICT)
    self_robot_poses=[]
    print "the poses of robot ", SELF_ROBOT_ID, " is: "
    while(not rospy.is_shutdown()):
        if (len(temp_node._children)) == 0:
            if goal_predicate(temp_node._state._pose_ID) and temp_goal_history_visited_dict[temp_node._state._pose_ID]==False:
                temp_unvisited_goal_number-=1
                print "temp_node._state._pose_ID: ", temp_node._state._pose_ID
                print "temp_node._state._num_of_steps: ", temp_node._state._num_of_steps
                print "temp_node._state._distance: ", temp_node._state._distance
                print "x: ", NODE_LIST[temp_node._state._pose_ID]._x,
                print "; y: ", NODE_LIST[temp_node._state._pose_ID]._y
                self_robot_poses.append(temp_node._state._pose_ID)
                total_reward+=POSITIVE_REWARD
                total_reward+=temp_node._state._num_of_steps*NEGATIVE_REWARD
            else:
                print "temp_node._state._pose_ID: ", temp_node._state._pose_ID
                print "temp_node._state._num_of_steps: ", temp_node._state._num_of_steps
                print "temp_node._state._distance: ", temp_node._state._distance
                print "x: ", NODE_LIST[temp_node._state._pose_ID]._x,
                print "; y: ", NODE_LIST[temp_node._state._pose_ID]._y
                self_robot_poses.append(temp_node._state._pose_ID)
                total_reward+=temp_node._state._num_of_steps*NEGATIVE_REWARD
            #print "break when len(temp_node._children)==0"
            break
        else:
            temp_poses.append(temp_node._state._pose_ID)
            print "temp_node._state._pose_ID: ", temp_node._state._pose_ID
            print "temp_node._state._num_of_steps: ", temp_node._state._num_of_steps
            print "temp_node._state._distance: ", temp_node._state._distance
            print "x: ", NODE_LIST[temp_node._state._pose_ID]._x,
            print "; y: ", NODE_LIST[temp_node._state._pose_ID]._y
            self_robot_poses.append(temp_node._state._pose_ID)
            #assert temp_node._state._pose_ID in temp_goal_history_visited_dict
            if NODE_LIST[temp_node._state._pose_ID]._is_goal==True and temp_goal_history_visited_dict[temp_node._state._pose_ID]==False:
                total_reward+=POSITIVE_REWARD
                total_reward+=NEGATIVE_REWARD
                temp_goal_history_visited_dict[temp_node._state._pose_ID]=True
                temp_unvisited_goal_number-=1
                assert temp_unvisited_goal_number>=0
                if temp_unvisited_goal_number==0:
                    #print "break when temp_unvisited_goal_number is 0."
                    break
            else:
                total_reward+=temp_node._state._num_of_steps*NEGATIVE_REWARD
            temp_node = temp_node.most_visits_child()
    
    print "len(temp_poses): ", len(temp_poses)
    print "local reward for a single robot is: ", total_reward
    return self_robot_poses

def pub_policy(action_sequence, time_step):
    global SELF_ROBOT_ID
    policy_msg = Policy()
    policy_msg.time_step = time_step
    policy_msg.action_sequence = action_sequence
    policy_msg.robot_ID = SELF_ROBOT_ID
    pub_policy = rospy.Publisher("/generated_policy", Policy, queue_size=10) 
    pub_policy.publish(policy_msg)
    print "The generated policy was published successfully in robot ", SELF_ROBOT_ID, "!"

#TODO:to imitate the above
def policy_callback(policy_msg):
    global ALL_ROBOT_POLICIES_DICT
    global ALL_ROBOT_POLICIES_TIMESTEP
    global SELF_ROBOT_ID
    print "policy_callback start in robot ", SELF_ROBOT_ID, "!"
    ALL_ROBOT_POLICIES_DICT[policy_msg.robot_ID] = policy_msg.action_sequence
    ALL_ROBOT_POLICIES_TIMESTEP[policy_msg.robot_ID] = policy_msg.time_step
    
def check_received_other_policy_flag(time_step):
    print "check_received_other_action_flag() start"
    global ALL_ROBOT_POLICIES_TIMESTEP
    #ACTION_SEQUENCE_PROBABILITY_DICT
    global ROBOT_NUMBER
    flag = True
    for i in range(ROBOT_NUMBER):
        if not i in ALL_ROBOT_POLICIES_TIMESTEP:
            print "flag 1"
            flag = False
            break
        elif ALL_ROBOT_POLICIES_TIMESTEP[i]==-1 or ALL_ROBOT_POLICIES_TIMESTEP[i] < time_step:
            print "flag 2"
            flag = False
            break
        elif ALL_ROBOT_POLICIES_TIMESTEP[i] > time_step:
            print "Error! ALL_ROBOT_POLICIES_TIMESTEP[i] is ", ALL_ROBOT_POLICIES_TIMESTEP[i]
            print "Error! time_step is ", time_step
            print "Error! ALL_ROBOT_POLICIES_TIMESTEP[i] > time_step"
            sys.exit(1)
    print "flag in check_received_other_action_flag(): ", flag
    return flag

def init_parameters():
    global ACTION_SEQUENCE_PROBABILITY_DICT
    global ACTION_SEQUENCE_PROBABILITY_TIMESTEP
    global ALL_ROBOT_POLICIES_DICT
    global ALL_ROBOT_POLICIES_TIMESTEP
    global NEXT_POSE_DICT
    global NODE_LIST
    global GOAL_HISTORY_VISITED_DICT
    global DOOR_DICT
    global UNVISITED_GOAL_NUMBER
    ACTION_SEQUENCE_PROBABILITY_DICT = {}#订阅到的其他机器人的规划好的动作序列概率分布, {key:robot_ID, value: macro_action_sequence_probability_list}
    ACTION_SEQUENCE_PROBABILITY_TIMESTEP = {}#保存订阅到的其他机器人动作序列概率分布所对应的时间步，{key:robot_ID, value: time_step}
    ALL_ROBOT_POLICIES_DICT={}
    ALL_ROBOT_POLICIES_TIMESTEP={}
    NEXT_POSE_DICT={}#key:pose_ID; value:[next_pose_0, next_pose_1]
    #the Graph of the environment is represented by the adjacent list
    #i.e. NEXT_POSE_DICT
    #ALL_ROBOT_POLICIES_DICT={}#用于输出最后决策结果Policy的字典.
    #ALL_ROBOT_POLICIES_TIMESTEP={}#用于输出最后决策时间步的TIMESTEP,同步收到了所有人的决策结果.
    NODE_LIST = []#the nodes of the environment
    #HORIZON = 150#the time for the robot finish its intrusion
    #NOTICE: HORIZON should not be set too large, you should set a suitable HORIZON
    #FORCE_TREE_EXPLORATION_PROBABILITY = 0.1#probability to force tree exploration
    GOAL_HISTORY_VISITED_DICT = {}#key: pose_ID; value: visited.
    DOOR_DICT={}#key:door pose_ID; value: any
    UNVISITED_GOAL_NUMBER=0
    
def start_dec_mcts():
    global SELF_ROBOT_ID
    global ROBOT_NUMBER
    global INITIAL_ROBOT_POSES
    global GOAL_HISTORY_VISITED_DICT
    global DOOR_DICT
    global UNVISITED_GOAL_NUMBER
    global BUDGET_OUTSIDE
    global BUDGET_INSIDE
    global INTERCEPT_DEPTH
    global ALL_ROBOT_POLICIES_DICT
    global ALL_ROBOT_POLICIES_TIMESTEP
    #TODO: 还需要修改的地方:
    #(1)有可能把初始位置很分散地分布在了地图当中之后, 这种subgoal机制本来就具有邻域发现的特点, 而使得他们的未来轨迹根本就不会重合.
    #(2)需要实现Dec-MCTS的算法, 同时需要再进行进一步的实验.
    rospy.init_node('dec_mcts', anonymous=True)
    #get the value of the parameters from the parameter server
    SELF_ROBOT_ID = rospy.get_param("~robot_ID")#self_robot_ID is 0 or 1
    ROBOT_NUMBER = rospy.get_param("~robot_number")#ROBOT_NUMBER is 2
    INITIAL_ROBOT_POSES.append(rospy.get_param("~initial_robot_pose_0"))#INITIAL_ROBOT_POSES = [0,187]
    INITIAL_ROBOT_POSES.append(rospy.get_param("~initial_robot_pose_1"))#INITIAL_ROBOT_POSES = [0,187]
    INITIAL_ROBOT_POSES.append(rospy.get_param("~initial_robot_pose_2"))#INITIAL_ROBOT_POSES = [0,187]
    print "SELF_ROBOT_ID: ", SELF_ROBOT_ID
    print "ROBOT_NUMBER: ", ROBOT_NUMBER
    print "INITIAL_ROBOT_POSES", INITIAL_ROBOT_POSES
    #sys.exit(1)
    #start the subscribers
    rospy.Subscriber("/decide_result", Decide, decide_callback)
    #rospy.Subscriber("/decide_result", Decide, decide_callback)
    rospy.Subscriber("/generated_policy", Policy, policy_callback)#TODO: 用来最后评价总体的结果是好是坏
    #build the Environment
    
    
    final_joint_reward_list = []
    final_time_cost_list = []
    
    all_test_rounds = 10
    for test_rounds in range(all_test_rounds):
        init_parameters()
        build_the_environment()
        for i in range(len(NODE_LIST)):
            print "pose_ID: ", i, " (x,y) is: ",
            print NODE_LIST[i]._x, NODE_LIST[i]._y
        print "GOAL_HISTORY_VISITED_DICT: ", GOAL_HISTORY_VISITED_DICT
        print "DOOR_DICT: ", DOOR_DICT
        #init_parameters()
        is_received_other_intention_flag = False
        #sys.exit(1)
        root_state = State(pose_ID=INITIAL_ROBOT_POSES[SELF_ROBOT_ID], distance=0)#pose 188 means the Graph vertex of (10,0)
        root = TreeNode(root_state)
        root._state._goal_visited_dict = GOAL_HISTORY_VISITED_DICT
        root._state._unvisited_goal_number = UNVISITED_GOAL_NUMBER
        #multi-robot cooperation
        #negotiating in a cooperative method by sharing the intentions represented as probablity distributions.
        tree_grow_count = 0
        #intention.msg的消息格式
        is_received_other_action_flag = False
        seconds_0=rospy.get_time()
        for negotiating_cooperatively_index in range(BUDGET_OUTSIDE):
            #thinking independently through Monte Carlo Tree Search guided by the joint rewards. 

            #is_received_other_intention_flag = check_received_other_intention_flag()#只在一开始的时候 check_received_other_action_flag, 所以总体来说，不会影响整个算法属于异步的性质。
            for thinking_independently_index in range(BUDGET_INSIDE):
                #print "tree grow for once, tree grow count is: ", tree_grow_count
                tree_grow_count +=1
                tree_grow(root, is_received_other_action_flag)

            #print "Time cost: ", seconds_1-seconds_0
            print "negotiating_cooperatively_index: ", negotiating_cooperatively_index, "in robot ", SELF_ROBOT_ID
            action_sequence_probability_list = get_action_sequence_probability_list(root=root, intercept_depth=INTERCEPT_DEPTH, intercept_length=None)
            pub_decide_result(action_sequence_probability_list=action_sequence_probability_list, robot_ID=SELF_ROBOT_ID, time_step=0)
            is_received_other_action_flag = check_received_other_action_flag(time_step=0)
        seconds_1=rospy.get_time()

        self_robot_poses = output_robot_track(root)

        #self_robot_poses = get_robot_track(root)
        #print "self_robot_poses: ", self_robot_poses

        pub_policy(action_sequence=self_robot_poses, time_step=0)

        #block here to check whether we receive other robots' policies
        while(not rospy.is_shutdown()):
            if check_received_other_policy_flag(time_step=0):
                break
            else:
                pub_policy(action_sequence=self_robot_poses, time_step=0)
                rospy.sleep(0.1)

        all_robot_poses_list = []
        for i in range(ROBOT_NUMBER):
            if not i in ALL_ROBOT_POLICIES_DICT:
                print "Error! not i in ALL_ROBOT_POLICIES_DICT in main()."
                sys.exit(1)
            else:
                all_robot_poses_list.append(ALL_ROBOT_POLICIES_DICT[i])
        #simulation_number = 1000000
        #print "all_robot_poses_list: ", all_robot_poses_list
        final_joint_reward = calculate_final_joint_reward(all_robot_poses_list)
        #print "type(all_robot_poses_list[1]):", type(all_robot_poses_list[1])
        #This is a function to evaluate the capture probability, each robot has a this kind of fuction.
        '''
        capture_probability = simulate_joint_capture(all_robot_poses_list, intruder, simulation_number)
        '''
        #print "The global capture probability for all of the robots in the cooperative team: ", capture_probability
        time_cost = seconds_1 - seconds_0
        print "all robot policies are received in robot ", SELF_ROBOT_ID
        print "Time cost for one cooperative decision: ", time_cost, "in robot ", SELF_ROBOT_ID
        print "final joint reward in robot ", SELF_ROBOT_ID, " is: ", final_joint_reward
        final_joint_reward_list.append(final_joint_reward)
        final_time_cost_list.append(time_cost)
    mean_value = np.mean(final_joint_reward_list)
    std_deviation = np.std(final_time_cost_list)
    print "mean value: ", mean_value
    print "std deviation: ", std_deviation
    print "average time cost", np.mean(final_time_cost_list)

    rospy.spin()
    
if __name__ == "__main__":
    #test_expand()
    #test_evolve()
    #start()
    start_dec_mcts()
