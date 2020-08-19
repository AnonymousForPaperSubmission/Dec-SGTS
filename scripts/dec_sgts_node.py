#!/usr/bin/env python
# coding=utf-8

__author__ = 'Minglong Li'
__date__ = 'Dec. 2019'

import numpy as np
import random
import math
import sys
import copy
import rospy
import pdb
#from dec_sgts.my_messages import Decide
from dec_sgts.msg import Decide
from dec_sgts.msg import Intention
from dec_sgts.msg import MacroActionSequenceProbability
from dec_sgts.msg import MacroAction
from dec_sgts.msg import ActionSequenceProbability
#from dec_pomdp.msg import ActionSequence
from dec_sgts.msg import Policy
from dec_sgts.msg import MacroPolicy

class EnvMap():
    def __init__(self):
        #some example environments
        '''
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
        
        #in the environment, 0 free, 1 means obstacle, 2 means door, 3 means goal.
        #the subgoals include the obstacle and the goal
        #in the matrix of the self._env, left corner means (0,0), 
        #the 0th vertical column is x axis, 
        #the 0th horizontal column is y axis.
        
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
        self._joint_reward=0.0#TODO: joint reward is used to guide the tree.
        self._num_of_steps=0

    #a copied state using as successor_state by copy.deepcopy()
    #update self._goal_visited_dict and self._unvisited_goal_number
    #TODO:evolve在这里可以先不改，直接这样，这里面的state._reward其实只是给一个初始值。这里的reward本身也没有太大用。
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

    #predicate whether it is a goal
    def goal_predicate(self):
        global NODE_LIST
        if NODE_LIST[self._pose_ID]._is_goal == True:
            self._is_goal = True
            return True
        else:
            self._is_goal = False
            return False

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
        #type==1, 角落的节点和目标点G，都符合subgoal heuristics
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
        self._q_value = 0.0
        self._reward = 0.0
        self._state = state
        self._children = []
        self._parent = None
        self._is_fully_expanded = False

    #back_propagate(path, joint_reward)#TODO:修改back_propagate，再加上修改uct公式。
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
    
    def uct(self):
        global SCALAR
        visits_parent = self._parent._visits
        visits_child = self._visits
        return self._reward + SCALAR * math.sqrt( (math.log(visits_parent))/visits_child )
        
    def check_is_leaf_node(self):
        return len(self._children)==0

    def check_is_fully_expanded(self):
        return self._is_fully_expanded

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

    #best reward child
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
                temp_reward = best_reward_child._reward
        return best_reward

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

#Action用来指定是宏观动作还是微观动作
class Action():
    def __init__(self, is_macro_action, pose_ID, num_of_steps):
        self._is_macro_action = is_macro_action
        self._pose_ID = pose_ID
        self._num_of_steps = num_of_steps#num_of_steps表示的是从之前的节点到现在的节点的动作。

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
        #具体添加哪一个subgoal节点到当前树节点当中，只取决于state.evolve()函数中的进化迭代次数。
        #具体当前的树节点到下一个subgoal之间的距离，
        terminal_distance_flag=successor_state.evolve(max_steps, gamma)
        #控制到终点时刻的结果。
        ######################################################
        if terminal_distance_flag == True:
            terminal_distance_count+=1
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

#walk randomly to the next adjacent pose
def next_random_pose(pose):
    global NEXT_POSE_DICT#key:pose_ID; value:[next_pose_0, next_pose_1]
    return random.choice(NEXT_POSE_DICT[pose])

#q_value = calculate_q_value(rollout_poses, temp_goal_visited_dict, temp_unvisited_goal_number)
#we assume gamma=1.0
def calculate_q_value(poses, temp_goal_visited_dict, temp_unvisited_goal_number):
    global POSITIVE_REWARD
    global NEGATIVE_REWARD
    q_value = 0.0
    assert temp_unvisited_goal_number>=0
    for i in range(len(poses)):
        #temp_goal_visited_dict is used only once in one rollout, we need not copy.deepcopy()
        if( (goal_predicate(poses[i]) ) and (temp_goal_visited_dict[poses[i]]==False) ):
            assert temp_unvisited_goal_number>0
            temp_unvisited_goal_number-=1
            temp_goal_visited_dict[poses[i]]=True
            q_value+=POSITIVE_REWARD
        else:
            assert temp_unvisited_goal_number>=0
            if temp_unvisited_goal_number==0:#if there is no temp_unvisited_goal_number
                pass#it means q_value+=0
            else:
                q_value+=NEGATIVE_REWARD
    return q_value

def goal_predicate(pose):
    global GOAL_HISTORY_VISITED_DICT
    return pose in GOAL_HISTORY_VISITED_DICT

#path: the tree nodes from root to leaf
def rollout(path):
    global POSITIVE_REWARD
    global NEGATIVE_REWARD
    global HORIZON
    global NODE_LIST
    q_value=0.0
    #temp_reward
    leaf_node=path[-1]
    assert leaf_node._state._distance<=HORIZON
    #path[-1]: leaf
    #randomly rollout from the leafnode
    rollout_poses = []
    leaf_distance = path[-1]._state._distance
    assert leaf_distance<=HORIZON
    #if leaf is the terminal_state
    if leaf_distance==HORIZON:
        q_value = 0
        return q_value

    assert leaf_node._state._distance<=HORIZON
    #add one initial pose of the next pose of leafnode pose
    leaf_node_pose = leaf_node._state._pose_ID
    rollout_poses.append(next_random_pose(leaf_node_pose))
    leaf_distance+=1
    while(not rospy.is_shutdown()):
        if leaf_distance==HORIZON:
            break
        rollout_poses.append(next_random_pose(rollout_poses[-1]))
        leaf_distance+=1
    q_value = calculate_q_value(rollout_poses, copy_another_dict(path[-1]._state._goal_visited_dict), path[-1]._state._unvisited_goal_number)
    return q_value


def back_propagate(path, reward):
    temp_node=path[-1]
    while(not rospy.is_shutdown()):
        temp_node.update(reward)
        temp_node=temp_node._parent
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
#TODO: 如何计算引导树生长的联合奖励？不只取决于自己，还取决于其他人，是从root节点就开始计算，还是从叶子节点开始计算。
#TODO:
#(1)　check_is_final_for_all(leaf_node),这里直接通过全局变量UNVISITED_GOAL_NUMBER来控制所有人的奖励计算。
    #采样逼近决定树中的某些点的邻域点，与联合奖励无关，这其中点的进化，完全由单独的奖励来决定。找邻域在这里问题不大。
#(2)　思考如何把宏观动作的代价和分布式ＭＣＴＳ协商解耦。是否Dec-MCTS需要先通过并行采样，把他计算出来呢，后面完全和普通的MCTS一样。
#(3)　一开始找subgoal_mcts通过q函数的方式更新reward，看是否可以直接通过从root到terminal_state的整条枝杈的方式进行更新。
#(4)  把动作序列表示的动作意图的协商共享写完。
#TODO： *************important
#把宏观动作生成模型的离线策略和在线策略同时表示出来。
#path: a node list, from the root node to leaf node。
#TODO:把rollout_for_joint_reward(),改成从root一直到terminal_state来计算奖励，把他写成一个可以跑通的，然后完全通过这个可以跑通的进行计算。
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
    #add one initial pose of the next pose of leafnode pose
    for i in range(len(path)):
        temp_action = Action(is_macro_action=True, pose_ID=path[i]._state._pose_ID, num_of_steps=path[i]._state._num_of_steps)
        rollout_poses.append(temp_action)
       
    leaf_node_pose = leaf_node._state._pose_ID
    if leaf_distance < HORIZON:
        rollout_poses.append(Action(is_macro_action=False, pose_ID = next_random_pose(leaf_node_pose), num_of_steps=1))
        leaf_distance+=1
        #print "in rollout(), hello 0."
        while(not rospy.is_shutdown()):
            assert leaf_distance <=HORIZON
            if leaf_distance==HORIZON:
                break
            rollout_poses.append(Action(is_macro_action=False, pose_ID =next_random_pose(rollout_poses[-1]._pose_ID), num_of_steps=1))
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
                #TODO: sample_from_shared_intention只跟intercept_depth有关, 而跟后面要获得的其他数据无关.
            else:#(with_shared_intention==False)
                temp_all_robots_poses_dict[i]=get_random_robot_poses(robot_ID=i)
    return calculate_joint_reward(temp_all_robots_poses_dict)

#只要是从树中提取的节点，一定是subgoal节点。
def get_random_robot_poses(robot_ID):
    global INITIAL_ROBOT_POSES
    global HORIZON#这里面的HORIZON主要是对distance的长度有限制。
    returned_poses = []
    initial_robot_pose = INITIAL_ROBOT_POSES[robot_ID]
    returned_poses.append(Action(is_macro_action=False, pose_ID = initial_robot_pose, num_of_steps=1))
    for i in range(HORIZON):
        returned_poses.append( Action(is_macro_action=False, pose_ID =next_random_pose(returned_poses[-1]._pose_ID), num_of_steps=1) )
    return returned_poses

#random pick an item from a list according to the probability of this item.
def random_pick(some_list,probabilities):
    x=random.uniform(0,1)
    cumulative_probability=0.0
    for item,item_probability in zip(some_list,probabilities):
        cumulative_probability+=item_probability
        if x < cumulative_probability:
            return item

#random pick an action sequence from a list according to the probability of this action sequence.
def random_pick_from_macro_action_sequence_probability_list(probability_list):
    macro_action_sequences = []
    probabilities = []
    for i in range(len(probability_list)):
        macro_action_sequences.append( probability_list[i][0])
        probabilities.append(probability_list[i][1])
    return random_pick(macro_action_sequences, probabilities)

#input: robot_ID
#output: other robots' poses
#从当前节点开始，一直到未来的节点，提取随机的动作。
def sample_from_shared_intention(robot_ID):
    global MACRO_ACTION_SEQUENCE_PROBABILITY_DICT
    global HORIZON
    #global TERMINAL_DISTANCE
    #action_sequence_and_useful_cycle_probability_list=ACTION_SEQUENCE_PROBABILITY_DICT[robot_ID]
    macro_action_sequence_probability_list = MACRO_ACTION_SEQUENCE_PROBABILITY_DICT[robot_ID]
    #print "macro_action_sequence_probability_list: ", macro_action_sequence_probability_list
    selected_intention = random_pick_from_macro_action_sequence_probability_list( macro_action_sequence_probability_list )
    #print "selected_intention: ", selected_intention
    robot_poses = []

    robot_poses = selected_intention
    #print "robot_poses: ", robot_poses
    #print "type(robot_poses[0]): ", type(robot_poses[0])
    returned_poses = []
    for i in range( len( robot_poses ) ):
        #returned_poses.append( Action(is_macro_action=robot_poses[i].is_macro_action, pose_ID=robot_poses[i].pose_ID, num_of_steps=robot_poses[i].num_of_steps) )
        returned_poses.append( robot_poses[i] )#intention_callback和publish_intention已经把消息类型的转换写完了，在内部都当做是Action来用就可以。
    #returned_poses要一直加到HORIZON为止.
    temp_distance=0
    for i in range(len(returned_poses)):
        temp_distance+=returned_poses[i]._num_of_steps
    #temp_distance = returned_poses[-1]._state._distance
    assert temp_distance<=HORIZON
    if temp_distance==HORIZON:
        pass
    else:
        while(not rospy.is_shutdown()):
            returned_poses.append( Action(is_macro_action=False, pose_ID =next_random_pose(returned_poses[-1]._pose_ID), num_of_steps=1) )
            temp_distance+=1
            if temp_distance==HORIZON:
                break
    return returned_poses
    
#优先实现可行性。
def spread_all_robots_poses_dict(all_robots_poses_dict):
    global SELF_ROBOT_ID
    global ROBOT_NUMBER
    #initialize spreaded_poses_dict.
    spreaded_poses_dict={}
    
    for i in range(ROBOT_NUMBER):
        spreaded_poses_dict[i]={}#把spreaded_poses_dict初始化成一个二维dict.
    for i in range(ROBOT_NUMBER):
        temp_time_step=0
        for j in range(len(all_robots_poses_dict[i])):
            if all_robots_poses_dict[i][j]._is_macro_action==False:
                #spreaded_poses_dict[i].append( all_robots_poses_dict[i][j]._pose_ID )
                spreaded_poses_dict[i][temp_time_step]=all_robots_poses_dict[i][j]._pose_ID
                temp_time_step+=1
            else:#all_robots_poses_dict[i][j]._is_macro_action==True
                for temp_index in range(all_robots_poses_dict[i][j]._num_of_steps-1):
                    #spreaded_poses_dict[i].append( -1 )#pose_ID=-1表示的是当前不是subgoal节点，只会产生惩罚，不会产生什么奖励。
                    #key_0这个i表示的是机器人robot的编号, key_1表示的是temp_time_step表示的是时间步.
                    #TODO: 下午从这里开始接着写, 这里面的两个key, 机器人和时间序列步的编号有点弄混了.
                    #TODO: (1)还有两个地方要写, 要写一个朴素的dec_mcts. (2)写完了朴素的dec_mcts之后开始写论文.(3)写论文的时候直接写一章Smooth the Break Point (4)一个加入Discounted Uct的dec_mcts (4)再写一个加入Intention Evolving的dec_mcts (5)其他的trick, 关于动态uct的问题.
                    if not i in spreaded_poses_dict:
                        print "i: ", i
                        print "Error! The key i is not in spreaded_poses_dict!"
                        sys.exit(1)
                    spreaded_poses_dict[i][temp_time_step]=-1
                    temp_time_step+=1
                spreaded_poses_dict[i][temp_time_step]=all_robots_poses_dict[i][j]._pose_ID
                temp_time_step+=1
                #spreaded_poses_dict[i].append(all_robots_poses_dict[i][j]._pose_ID)#把当前的pose_ID加进来.
    return spreaded_poses_dict#这里返回的是一个二维dict, spreaded_poses_dict[robot_ID][time_step]=pose_ID, key_0表示的是robot_ID, key_1表示的是time_step.

#TODO: 如果要是用online-planning, 需要把GOAL_HISTORY_VISITED_DICT和UNVISITED_GOAL_NUMBER加入进来.
#TODO: GOAL_HISTORY_VISITED_DICT 和 UNVISITED_GOAL_NUMBER 用来在online-planning当中记录哪些是访问过的节点，哪些是未访问过的节点。
#TODO:
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
    #transformed_poses_dict = transform_poses_dict(spreaded_poses_dict)
    for time_step in range(HORIZON):
        for robot_ID in range(ROBOT_NUMBER):
            #assert time_step in spreaded_poses_dict[robot_ID]
            #对齐数据
            #只有在最后policy阶段加对其数据,在之前的joint reward阶段不加对其数据.
            '''
            if not time_step in spreaded_poses_dict[robot_ID]:
                spreaded_poses_dict[robot_ID][time_step]=-1
            '''
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

def calculate_final_reward(all_robots_poses_dict):
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
    #transformed_poses_dict = transform_poses_dict(spreaded_poses_dict)
    for time_step in range(HORIZON):
        for robot_ID in range(ROBOT_NUMBER):
            #assert time_step in spreaded_poses_dict[robot_ID]
            #对齐数据
            #只有在最后policy阶段加对其数据,在之前的joint reward阶段不加对其数据.
            
            if not time_step in spreaded_poses_dict[robot_ID]:
                spreaded_poses_dict[robot_ID][time_step]=-1
            
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

ALL_TASKS_COMPLETE_BONUS=0.0#set a bonus when all tasks are completed.

'''
def tree_grow(root):
    global HORIZON
    global ALL_TASKS_COMPLETE_BONUS
    path = select(root)
    assert path[-1]._state._distance<=HORIZON#leaf_node.poses
    if check_is_final_for_all(leaf_node = path[-1]):#TODO:找到一个数据结构，用来检测对于全局机器人而言，是否需要check_is_final_for_all
        q_value = ALL_TASKS_COMPLETE_BONUS
    else:
        #q_value = rollout(path)#path means the selected nodes from root to leaf, path[-1] means the selected_node
        q_value = rollout_with_joint_reward(path)#path means the selected nodes from root to leaf, path[-1] means the selected_node
    back_propagate(path, q_value)
'''

def tree_grow(root, is_received_other_intention_flag):
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
    joint_reward = rollout_for_joint_reward(path, is_received_other_intention_flag)#path means the selected nodes from root to leaf, path[-1] means the selected_node
    back_propagate(path, joint_reward)#直接用整条枝杈的joint_reward去控制参数.

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

def init_parameters():
    global THRESHOLD
    global ACTION_COVERAGE
    global TOLERATED_ERROR
    global NEXT_POSE_DICT
    global MACRO_ACTION_SEQUENCE_PROBABILITY_DICT
    global MACRO_ACTION_SEQUENCE_PROBABILITY_TIMESTEP
    global ALL_ROBOT_POLICIES_DICT
    global ALL_ROBOT_POLICIES_TIMESTEP
    global NODE_LIST
    global GOAL_HISTORY_VISITED_DICT
    global DOOR_DICT
    global UNVISITED_GOAL_NUMBER
    ACTION_COVERAGE = 0.95
    TOLERATED_ERROR=0.05
    THRESHOLD=threshold(ACTION_COVERAGE, TOLERATED_ERROR)
    print "THRESHOLD: ", THRESHOLD
    #the Graph of the environment is represented by the adjacent list
    #i.e. NEXT_POSE_DICT
    NEXT_POSE_DICT={}#key:pose_ID; value:[next_pose_0, next_pose_1]
    MACRO_ACTION_SEQUENCE_PROBABILITY_DICT = {}#订阅到的其他机器人的规划好的动作序列概率分布, {key:robot_ID, value: macro_action_sequence_probability_list}
    MACRO_ACTION_SEQUENCE_PROBABILITY_TIMESTEP = {}#保存订阅到的其他机器人动作序列概率分布所对应的时间步，{key:robot_ID, value: time_step}
    ALL_ROBOT_POLICIES_DICT={}#用于输出最后决策结果Policy的字典.
    ALL_ROBOT_POLICIES_TIMESTEP={}#用于输出最后决策时间步的TIMESTEP,同步收到了所有人的决策结果.
    NODE_LIST = []#the nodes of the environment
    #HORIZON = 150#the time for the robot finish its intrusion
    #NOTICE: HORIZON should not be set too large, you should set a suitable HORIZON
    #FORCE_TREE_EXPLORATION_PROBABILITY = 0.1#probability to force tree exploration
    GOAL_HISTORY_VISITED_DICT = {}#key: pose_ID; value: visited.
    DOOR_DICT={}#key:door pose_ID; value: any
    UNVISITED_GOAL_NUMBER=0

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

def test_parallel_expand():
    pass
    
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
def policy_callback(msg):
    global ALL_ROBOT_POLICIES_DICT
    global ALL_ROBOT_POLICIES_TIMESTEP
    global SELF_ROBOT_ID
    print "policy_callback start in robot ", SELF_ROBOT_ID, "!"
    ALL_ROBOT_POLICIES_DICT[policy_msg.robot_ID] = policy_msg.action_sequence
    ALL_ROBOT_POLICIES_TIMESTEP[policy_msg.robot_ID] = policy_msg.time_step
'''

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
    print "the poses of robot ", SELF_ROBOT_ID, " is: "
    macro_action_sequence = []
    '''
    Message: MacroAction
    bool is_macro_action
    int32 pose_ID
    int32 num_of_steps
    '''
    while(not rospy.is_shutdown()):
        if (len(temp_node._children)) == 0:
            if goal_predicate(temp_node._state._pose_ID) and temp_goal_history_visited_dict[temp_node._state._pose_ID]==False:
                temp_unvisited_goal_number-=1
                '''
                print "temp_node._state._pose_ID: ", temp_node._state._pose_ID
                print "temp_node._state._num_of_steps: ", temp_node._state._num_of_steps
                print "temp_node._state._distance: ", temp_node._state._distance
                print "x: ", NODE_LIST[temp_node._state._pose_ID]._x,
                print "; y: ", NODE_LIST[temp_node._state._pose_ID]._y
                '''
                macro_action_sequence.append(MacroAction(is_macro_action=True, pose_ID=temp_node._state._pose_ID, num_of_steps=temp_node._state._num_of_steps))
                total_reward+=POSITIVE_REWARD
                total_reward+=temp_node._state._num_of_steps*NEGATIVE_REWARD
            else:
                '''
                print "temp_node._state._pose_ID: ", temp_node._state._pose_ID
                print "temp_node._state._num_of_steps: ", temp_node._state._num_of_steps
                print "temp_node._state._distance: ", temp_node._state._distance
                print "x: ", NODE_LIST[temp_node._state._pose_ID]._x,
                print "; y: ", NODE_LIST[temp_node._state._pose_ID]._y
                '''
                macro_action_sequence.append(MacroAction(is_macro_action=True, pose_ID=temp_node._state._pose_ID, num_of_steps=temp_node._state._num_of_steps))
                total_reward+=temp_node._state._num_of_steps*NEGATIVE_REWARD
            print "break when len(temp_node._children)==0"
            break
        else:
            temp_poses.append(temp_node._state._pose_ID)
            '''
            print "temp_node._state._pose_ID: ", temp_node._state._pose_ID
            print "temp_node._state._num_of_steps: ", temp_node._state._num_of_steps
            print "temp_node._state._distance: ", temp_node._state._distance
            print "x: ", NODE_LIST[temp_node._state._pose_ID]._x,
            print "; y: ", NODE_LIST[temp_node._state._pose_ID]._y
            '''
            macro_action_sequence.append(MacroAction(is_macro_action=True, pose_ID=temp_node._state._pose_ID, num_of_steps=temp_node._state._num_of_steps))
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

    #print "len(temp_poses): ", len(temp_poses)
    #print "total_reward: ", total_reward
    #macro_action_sequence = [0]
    return macro_action_sequence

def take_second(elem):
    return elem[1]

#TODO: trick: 一开始的时候就让他们协商的时间比较长,争取能找到最优的方式.
def get_macro_action_sequence_probability_list(root, intercept_depth, intercept_length):
    if intercept_depth == None:
        intercept_depth = 5#这个交互的长度是一个很trick的值,需要通过细微的设置才可以.
    if intercept_length == None:
        pass
    #下面的部分，相当于进行一次深度搜索，把搜索树中所有的深度为indept_depth的所有枝杈末端的节点找到。
    list_parent = []#用于搜索出所有的动作序列的list_parent，是临时的父亲节点变量
    list_children = []#用于搜索出所有的动作序列的list_children，是临时的孩子节点变量
    if intercept_depth == 1:
        for i in range(len(root._children)):
            list_children.append(root._children[i])#先加入第一层的孩子节点
    else:
        for i in range(len(root._children)):
            list_parent.append(root._children[i])#先加入第一层的孩子节点
        #计算树中深度为indept_depth的所有的枝杈末端的节点。
        for i in range(int(intercept_depth)-1):
            list_children = []
            while(not len(list_parent)==0):
                for i in range(len(list_parent[0]._children)):
                    list_children.append(list_parent[0]._children[i])
                del(list_parent[0])
            list_parent = list_children    
    visits_list = []
    #因为如果只通过reward控制, reward当中可能带有负数的奖励,不好调节. 所以通过每个节点的访问次数_visits,来控制每个macro_action的比例.
    #TODO: 把整体的过程全部都改成visits.
    for i in range(len(list_children)):
        visits_list.append(list_children[i]._visits)

    if len(visits_list)==0:
        print "Error! len(visits_list) is 0 in get_macro_action_sequence_probability_list!"
        sys.exit(1)
    #print "visits_list: ", visits_list
    '''
    visits_list_min = min(visits_list)
    for i in range(len(visits_list)):
        visits_list[i] = visits_list[i]+visits_list_min
    print "visits_list: ", visits_list    
    #so, the visits_list[i] will be >= 0, and the sum(visits_list) will be >= 0
    '''
    visits_list_sum = sum(visits_list)
    #print "visits_list_sum: ", visits_list_sum
    if visits_list_sum > 0:
        macro_action_sequence_probability = np.array(visits_list)/visits_list_sum#
    elif visits_list_sum == 0:
        print "Error! visits_list_sum=0 in get_macro_action_sequence_probability_list."    
    else:
        print "Error! visits_list_sum<0 in get_macro_action_sequence_probability_list"
    #action_sequence_probability = list(action_sequence_probability)
    #Action list 在这里是不一样的, Action sequence最起码要有num_of_steps这些东西.
    macro_action_sequence_list = []
    for i in range(len(list_children)):
        temp_macro_action_list= []
        temp_node = list_children[i]
        while(not temp_node==None):
            temp_macro_action_list.append( Action(is_macro_action=True, pose_ID=temp_node._state._pose_ID, num_of_steps=temp_node._state._num_of_steps ) )
            temp_node=temp_node._parent
        temp_macro_action_list.reverse()
        macro_action_sequence_list.append(temp_macro_action_list)
    
    macro_action_sequence_probability_list = []
    for i in range(len(macro_action_sequence_list)):
        macro_action_sequence_probability_list.append([macro_action_sequence_list[i], macro_action_sequence_probability[i]])
    macro_action_sequence_probability_list.sort(key=take_second, reverse=True)
    #print "macro_action_sequence_probability_list: ", macro_action_sequence_probability_list 
    return macro_action_sequence_probability_list

    #消息MacroAction的消息格式
    '''
    std_msgs/Header header
      uint32 seq
      time stamp
      string frame_id
    dec_pomdp/MacroActionSequenceProbability[] macro_action_sequence_probability_list
      dec_pomdp/MacroAction[] macro_action_sequence
        bool is_macro_action
        int32 pose_ID
        int32 num_of_steps
      float32 probability
    int32 robot_ID
    int32 time_step
    '''
def intention_callback(intention_msg):
    #print "intention callback!"
    global MACRO_ACTION_SEQUENCE_PROBABILITY_DICT
    global MACRO_ACTION_SEQUENCE_PROBABILITY_TIMESTEP
    macro_action_sequence_probability_list = []
    
    for i in range(len(intention_msg.macro_action_sequence_probability_list)):
        macro_action_sequence=[]
        for j in range( len(intention_msg.macro_action_sequence_probability_list[i].macro_action_sequence) ):
            macro_action_sequence.append( Action( intention_msg.macro_action_sequence_probability_list[i].macro_action_sequence[j].is_macro_action, intention_msg.macro_action_sequence_probability_list[i].macro_action_sequence[j].pose_ID, intention_msg.macro_action_sequence_probability_list[i].macro_action_sequence[j].num_of_steps ) )
        #macro_action_sequence = intention_msg.macro_action_sequence_probability_list[i].macro_action_sequence
        #print "macro_action_sequence in decide_callback: ", action_sequence
        probability = intention_msg.macro_action_sequence_probability_list[i].probability
        macro_action_sequence_probability_list.append([macro_action_sequence, probability])
    time_step = intention_msg.time_step
    #print "The received macro_action_sequence_probability_list time_step is: ", time_step
    robot_ID = intention_msg.robot_ID
    #macro_action_sequence_probability_list由macro_action_sequence和probability组成
    #macro_action_sequence是一个存储了MacroAction的列表, MacroAction是消息类型, 不是代码中的类Action. 
    #probability是一个float32类型的变量, 具体表示每个宏观动作序列的概率.
    #print "macro_action_sequence_probability_list: ", macro_action_sequence_probability_list
    MACRO_ACTION_SEQUENCE_PROBABILITY_DICT[robot_ID] = macro_action_sequence_probability_list
    MACRO_ACTION_SEQUENCE_PROBABILITY_TIMESTEP[robot_ID] = time_step

#trick: 这里面publish的intention序列的长度, 对应着树的枝杈的深度, 正好是我能搜索到的intention的深度.
def pub_intention_result(macro_action_sequence_probability_list, robot_ID, time_step):
    pub = rospy.Publisher('/intention_result', Intention, queue_size=10)
    intention_result = Intention()
    for i in range(len(macro_action_sequence_probability_list)):
        macro_action_sequence_probability = MacroActionSequenceProbability()
        for j in range( len(macro_action_sequence_probability_list[i][0]) ):
            macro_action_sequence_probability.macro_action_sequence.append(MacroAction(macro_action_sequence_probability_list[i][0][j]._is_macro_action, macro_action_sequence_probability_list[i][0][j]._pose_ID, macro_action_sequence_probability_list[i][0][j]._num_of_steps))
        #macro_action_sequence_probability.macro_action_sequence= macro_action_sequence_probability_list[i][0]
        macro_action_sequence_probability.probability = macro_action_sequence_probability_list[i][1]
        intention_result.macro_action_sequence_probability_list.append(macro_action_sequence_probability)
    #decided_result.action_sequence_probability_list = action_sequence_probability_list
    intention_result.robot_ID = robot_ID
    intention_result.time_step = time_step
    pub.publish(intention_result)
    
#只有第一次接收到其他机器人的消息的时候, 需要调用这个函数, 判断是否接收到intention. 后面无所谓, 由于算法的异步性, 不用过多考虑接收到的intention是否是最新的.
def check_received_other_intention_flag(time_step):
    #print "check_received_other_action_flag"
    global MACRO_ACTION_SEQUENCE_PROBABILITY_TIMESTEP#key:robot_ID, value: time_step
    #ACTION_SEQUENCE_PROBABILITY_DICT
    global ROBOT_NUMBER
    flag = True
    for i in range(ROBOT_NUMBER):
        if not i in MACRO_ACTION_SEQUENCE_PROBABILITY_TIMESTEP:
            flag = False
            break
        elif MACRO_ACTION_SEQUENCE_PROBABILITY_TIMESTEP[i]==-1 or MACRO_ACTION_SEQUENCE_PROBABILITY_TIMESTEP[i] < time_step:
            flag = False
            break
        elif MACRO_ACTION_SEQUENCE_PROBABILITY_TIMESTEP[i] > time_step:
            #print "the error wall time in 881", rospy.get_time()
            print "Error! ACTION_SEQUENCE_PROBABILITY_TIMESTEP[i] is ", MACRO_ACTION_SEQUENCE_PROBABILITY_TIMESTEP[i]
            print "Error! time_step is ", time_step
            print "Error! ACTION_SEQUENCE_PROBABILITY_TIMESTEP[i] > time_step"
            sys.exit(1)#TODO:很有可能是ACTION_SEQUENCE_PROBABILITY_TIMESTEP
    return flag

#这里统一使用消息类型MacroAction来组织MacroActionSequence, 不使用当前代码中的MacroAction类.
def pub_macro_policy(macro_action_sequence, time_step):
    global SELF_ROBOT_ID
    policy_msg = MacroPolicy()
    policy_msg.time_step = time_step
    policy_msg.macro_action_sequence = macro_action_sequence
    policy_msg.robot_ID = SELF_ROBOT_ID
    pub_policy = rospy.Publisher("/generated_policy", MacroPolicy, queue_size=10) 
    pub_policy.publish(policy_msg)
    #print "The generated policy was published successfully in robot ", SELF_ROBOT_ID, "!"
    
#用来计算分布式规划的最终结果.
def macro_policy_callback(macro_policy_msg):
    global ALL_ROBOT_POLICIES_DICT
    global ALL_ROBOT_POLICIES_TIMESTEP
    global SELF_ROBOT_ID
    #print "macro_policy_callback start in robot ", SELF_ROBOT_ID, "!"
    ALL_ROBOT_POLICIES_DICT[macro_policy_msg.robot_ID] = macro_policy_msg.macro_action_sequence
    #print "macro_policy_msg.robot_ID: ", macro_policy_msg.robot_ID
    #print "macro_policy_msg.time_step: ",macro_policy_msg.time_step
    ALL_ROBOT_POLICIES_TIMESTEP[macro_policy_msg.robot_ID] = macro_policy_msg.time_step
    #print "ALL_ROBOT_POLICIES_TIMESTEP in macro_policy_callback(): ", ALL_ROBOT_POLICIES_TIMESTEP

#检测是否接收到了总体的消息.
def check_received_other_policy_flag(time_step):
    #print "check_received_other_action_flag"
    global ALL_ROBOT_POLICIES_TIMESTEP
    #ACTION_SEQUENCE_PROBABILITY_DICT
    global ROBOT_NUMBER
    flag = True
    #print "ALL_ROBOT_POLICIES_TIMESTEP in check_received_other_action_flag(): ", ALL_ROBOT_POLICIES_TIMESTEP
    for i in range(ROBOT_NUMBER):
        if not i in ALL_ROBOT_POLICIES_TIMESTEP:
            flag = False
            break
        elif ALL_ROBOT_POLICIES_TIMESTEP[i]==-1 or ALL_ROBOT_POLICIES_TIMESTEP[i] < time_step:
            flag = False
            break
        elif ALL_ROBOT_POLICIES_TIMESTEP[i] > time_step:
            print "Error! ALL_ROBOT_POLICIES_TIMESTEP[i] is ", ALL_ROBOT_POLICIES_TIMESTEP[i]
            print "Error! time_step is ", time_step
            print "Error! ALL_ROBOT_POLICIES_TIMESTEP[i] > time_step"
            sys.exit(1)
    return flag

def start_subgoal_dec_mcts():
    global SELF_ROBOT_ID
    global ROBOT_NUMBER
    global INITIAL_ROBOT_POSES
    global GOAL_HISTORY_VISITED_DICT
    global DOOR_DICT
    global UNVISITED_GOAL_NUMBER
    global BUDGET_OUTSIDE
    global BUDGET_INSIDE
    global INTERCEPT_DEPTH
    global ITERATION
    global TEST_ROUNDS
    #(1)有可能把初始位置很分散地分布在了地图当中之后, 这种subgoal机制本来就具有邻域发现的特点, 而使得他们的未来轨迹根本就不会重合.
    #(2)需要实现Dec-MCTS的算法, 同时需要再进行进一步的实验.
    #(3)无论让机器人从同一个初始位置出发，还是让机器人从多个初始位置出发，都会产生比较好的结果。
    rospy.init_node('subgoal_dec_mcts', anonymous=True)
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
    #sys.exit(1)
    #start the subscribers
    rospy.Subscriber("/intention_result", Intention, intention_callback)
    #rospy.Subscriber("/decide_result", Decide, decide_callback)
    rospy.Subscriber("/generated_policy", MacroPolicy, macro_policy_callback)#TODO: 用来最后评价总体的结果是好是坏
    rospy.sleep(0.5)#防止出现TCP/IP connection failed的错误
    #build the Environment
    all_final_rewards=[]
    all_time_costs=[]
    for test_iteration in range(TEST_ROUNDS):
        final_reward=0
        init_parameters()
        build_the_environment()
        for i in range(len(NODE_LIST)):
            print "pose_ID: ", i, " (x,y) is: ",
            print NODE_LIST[i]._x, NODE_LIST[i]._y
        print "GOAL_HISTORY_VISITED_DICT: ", GOAL_HISTORY_VISITED_DICT
        print "DOOR_DICT: ", DOOR_DICT
        #sys.exit(1)
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
        seconds_0=rospy.get_time()
        for negotiating_cooperatively_index in range(BUDGET_OUTSIDE):
            #thinking independently through Monte Carlo Tree Search guided by the joint rewards. 
            
            #is_received_other_intention_flag = check_received_other_intention_flag()#只在一开始的时候 check_received_other_action_flag, 所以总体来说，不会影响整个算法属于异步的性质。
            for thinking_independently_index in range(BUDGET_INSIDE):
                #print "tree grow for once, tree grow count is: ", tree_grow_count
                tree_grow_count +=1
                tree_grow(root, is_received_other_intention_flag)            
            #print "Time cost: ", seconds_1-seconds_0
            macro_action_sequence_probability_list = get_macro_action_sequence_probability_list(root=root, intercept_depth=INTERCEPT_DEPTH, intercept_length=None)
            pub_intention_result(macro_action_sequence_probability_list=macro_action_sequence_probability_list, robot_ID=SELF_ROBOT_ID, time_step=0)
            is_received_other_intention_flag = check_received_other_intention_flag(time_step=0)
        seconds_1=rospy.get_time()
        
        time_cost = seconds_1-seconds_0
        all_time_costs.append(time_cost)
        print "time_cost: ", time_cost
        macro_action_sequence=output_robot_track(root)
        '''
        self_robot_poses = get_robot_track(root)
        print "self_robot_poses: ", self_robot_poses
        '''
        #pub_policy(action_sequence=self_robot_poses, time_step=0)
        #block here to check whether we receive other robots' policies
        while(not rospy.is_shutdown()):
            if check_received_other_policy_flag(time_step=0):
                break
            else:
                pub_macro_policy(macro_action_sequence=macro_action_sequence, time_step=0)
                rospy.sleep(0.1)
        
        #print "All robot policies were received."
        #print "ALL_ROBOT_POLICIES_DICT: ", ALL_ROBOT_POLICIES_DICT
        #print "ALL_ROBOT_POLICIES_TIMESTEP: ", ALL_ROBOT_POLICIES_TIMESTEP
        #有可能出现key_error 38
        for i in range(ROBOT_NUMBER):
            for j in range(len(ALL_ROBOT_POLICIES_DICT[i])):
                ALL_ROBOT_POLICIES_DICT[i][j]=Action(ALL_ROBOT_POLICIES_DICT[i][j].is_macro_action,ALL_ROBOT_POLICIES_DICT[i][j].pose_ID,ALL_ROBOT_POLICIES_DICT[i][j].num_of_steps)
        #此时的 ALL_ROBOT_POLICIES_DICT 毕竟是从树中截取的一些节点,这些节点可能总体的深度达不到HORIZON
        
        #spread_all_robots_poses_dict(ALL_ROBOT_POLICIES_DICT)
        final_reward = calculate_final_reward(ALL_ROBOT_POLICIES_DICT)
        print "final_reward is: ", final_reward
        #print "final_reward for test iteration ", test_iteration, " is: " final_reward
        all_final_rewards.append(final_reward)
        del root_state
        del root
    print all_final_rewards
    mean_value = np.mean(all_final_rewards)
    std_deviation = np.std(all_final_rewards, ddof=1)
    print "mean value: ", mean_value
    print "std deviation: ", std_deviation
    print "average time cost", np.mean(all_time_costs)
    '''
    #下面的这部分把整体的奖励整理出来.
    all_robot_poses_list = []
    for i in range(ROBOT_NUMBER):
        if not i in ALL_ROBOT_POLICIES_DICT:
            print "Error! not i in ALL_ROBOT_POLICIES_DICT in main()."
            sys.exit(1)
        else:
            all_robot_poses_list.append(ALL_ROBOT_POLICIES_DICT[i])

    '''
    ''''
    #下面的这部分把最后的结果计算出来.
    simulation_number = 1000000
    print "all_robot_poses_list: ", all_robot_poses_list
    #print "type(all_robot_poses_list[1]):", type(all_robot_poses_list[1])
    #This is a function to evaluate the capture probability, each robot has a this kind of fuction.
    capture_probability = simulate_joint_capture(all_robot_poses_list, intruder, simulation_number)
    print "The global capture probability for all of the robots in the cooperative team: ", capture_probability
    '''
    rospy.spin()

if __name__ == "__main__":
    #global variables
    ###############################################################################
    SCALAR = math.sqrt(2.0)#balance the exploration and exploitation
    #通过调节INTERCEPT_DEPTH(意图共享的长度),B
    INTERCEPT_DEPTH= 5 #INTERCEPT_DEPTH在这里是一个可以调节的参数
    BUDGET_OUTSIDE =150
    BUDGET_INSIDE = 1000#iterative budget for the tree growth
    NODE_NUMBER = 0#number of the environment locations
    #the Graph of the environment is represented by the adjacent list
    #i.e. NEXT_POSE_DICT
    NEXT_POSE_DICT={}#key:pose_ID; value:[next_pose_0, next_pose_1]
    MACRO_ACTION_SEQUENCE_PROBABILITY_DICT = {}#订阅到的其他机器人的规划好的动作序列概率分布, {key:robot_ID, value: macro_action_sequence_probability_list}
    MACRO_ACTION_SEQUENCE_PROBABILITY_TIMESTEP = {}#保存订阅到的其他机器人动作序列概率分布所对应的时间步，{key:robot_ID, value: time_step}
    ALL_ROBOT_POLICIES_DICT={}#用于输出最后决策结果Policy的字典.
    ALL_ROBOT_POLICIES_TIMESTEP={}#用于输出最后决策时间步的TIMESTEP,同步收到了所有人的决策结果.
    NODE_LIST = []#the nodes of the environment
    #HORIZON = 150#the time for the robot finish its intrusion
    #NOTICE: HORIZON should not be set too large, you should set a suitable HORIZON
    HORIZON = 40#the terminal distance for the robot because the battery limit or something else
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
    SELF_ROBOT_ID = -1
    ROBOT_NUMBER=-1
    INITIAL_ROBOT_POSES=[]

    TEST_ROUNDS=10
    ################################################################################
    start_subgoal_dec_mcts()
