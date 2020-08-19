#!/usr/bin/env python
# coding=utf-8

import numpy as np
import random
import math
import sys
import copy
import rospy
import pdb
SCALAR = math.sqrt(2.0)#balance the exploration and exploitation
BUDGET = 40000#iterative budget for the tree growth
NODE_NUMBER = 0#number of the environment locations
#the Graph of the environment is represented by the adjacent list
#i.e. NEXT_POSE_DICT
NEXT_POSE_DICT={}#key:pose_ID; value:[next_pose_0, next_pose_1]
NODE_LIST = []#the nodes of the environment
#HORIZON = 150#the time for the robot finish its intrusion
#NOTICE: HORIZON should not be set too large, you should set a suitable HORIZON
HORIZON = 60#the time for the robot finish its intrusion
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
                     [0,0,0,0,0,2,0,0,0,0,0,1,0,0,0,0,0,2,0,0,0,0,0]]

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
    
if __name__ == "__main__":
    #test_expand()
    #test_evolve()
    start()
