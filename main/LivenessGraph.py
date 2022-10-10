import numpy as np
from copy import deepcopy
import pickle
from math import sqrt
import time

from SafetyReducedGraph import *

class LivenessGraph(object):
    def __init__(self):
        """
        Liveness graph is a sub-graph of safety graph by removing nodes that do not have a path to the goal,
        i.e., the nodes in liveness graph correspond to the set of safe initial states that our algorithm provides safety guarantee. 

        To satisfy the liveness property, each abstract state in the liveness graph is assigned with a controller partition 
        and a next_state, which are used to collect training data.
        """
        pass

    def construct(self):
        """ Construct the liveness graph """
        # Find all states that can be reached from the goal by BFS.
        self.reverse_edges()
        self.bfs()
        self.reverse_edges()

        # Liveness graph only contains nodes that can be reached from the goal.
        remove_keys = [name for name, node in self.node_dict.items() if node.parent is None and node.identity != 1]
        for k in remove_keys:
            del self.node_dict[k]
            self.num_nodes -= 1
        print('States safe but not reachable (total %d):' % len(remove_keys))
        print(remove_keys)

    def reverse_edges(self):
        """ Reverse direction of all edges """
        reversed_dict = deepcopy(self.node_dict)
        for node in reversed_dict.values():
            node.adjacent = {}
        for frm_name, frm_node in self.node_dict.items():
            for to_name, label in frm_node.adjacent.items():
                reversed_dict[to_name].adjacent[frm_name] = label     
        self.node_dict = reversed_dict

    def bfs(self):
        """ Breadth-first search """
        for name, node in self.node_dict.items():
            if node.identity == 1:
                goal_name = name
                node.color = 1
                node.d = 0
                node.parent = None
                #node.label = None
            else:
                node.color = 0
                node.d = None
                node.parent = None      
                #node.label = None
        queue = [goal_name]
        while queue:
            u_name = queue.pop(0)
            u_node = self.node_dict[u_name]
            for v_name, u2v_label in u_node.adjacent.items():
                v_node = self.node_dict[v_name]
                if v_node.color == 0:
                    v_node.color = 1
                    v_node.d = u_node.d + 1
                    v_node.parent = u_name
                    #v_node.label = u2v_label
                    queue.append(v_name)
            u_node.color = 2

    def assign_next_be_parent(self, specified_states):
        """ 
        Assign next_state be the parent of each node from BFS.
        NOTE: The shortest path is not unique, and BFS just returns one of them.
        """
        print('\nAssign next_state be parents in BFS')
        for q_idx in specified_states:
            node = self.node_dict[q_idx]
            assert node.identity == 0, 'Not assign next_state for the goal'
            node.next_state = node.parent
            node.tried_next_states.append(node.next_state)
            #print('\nName:', q_idx)
            #print('adjacent nodes:', node.adjacent.keys())
            #print('parent:', node.parent)

    def assign_next_close_parent(self, specified_states):
        """ 
        Assign next_state be the one close to the parent (can be the parent itself) in x, y dimensions, 
        and close to the current state in higher order dimensions. 
        """
        print('\nAssign next_state be close to parents')
        # Change theta range of the goal to [0, pi] so its center used in computing distance is pi/2.
        #for node in self.node_dict.values():
        #    if node.identity == 1:
        #        node.q_low[2] = 0
        #        node.q_up[2]  = np.pi
        # Assign next_state based on the distance metric.      
        for q_idx in specified_states:
            node = self.node_dict[q_idx]
            old_next = node.next_state
            assert node.identity == 0, 'Not assign next_state for the goal'
            # Find node in the adjacent list of the current node that has the shortest distance measure. 
            incumbent_dist = const.inf
            for adjacent_name in node.adjacent.keys():
                if adjacent_name in node.tried_next_states:
                    continue
                adjacent_dist = self.compute_assign_state_dist(q_idx, node.parent, adjacent_name)
                if adjacent_dist < incumbent_dist:
                    incumbent_dist = adjacent_dist
                    node.next_state = adjacent_name
            node.tried_next_states.append(node.next_state)
            print('\nName:', q_idx)
            print('adjacent nodes:', node.adjacent.keys())
            print('parent:', node.parent)
            print('tried_next_states:', node.tried_next_states)
            print('new next_state:', node.next_state)
            # TODO: Handle this assert. 
            assert node.next_state != old_next, 'All nodes in adjacent list of current node has been tried.'

    def compute_assign_state_dist(self, q_idx, parent_name, adjacent_name):
        """ 
        Distance measure is the weighted sum of distance between the adjacent node and the parent in x, y dimensions,
        and distance between the adjacent node and the current node in higher order dimensions. 
        """
        # If want to penalize change of angle, set a large w_high (if penalize too much, the trajectory is not able to make a turn).
        w_xy   = 1000
        w_high = 1
        current_center = [(low+up)/2 for low, up in zip(self.node_dict[q_idx].q_low, self.node_dict[q_idx].q_up)]
        parent_center = [(low+up)/2 for low, up in zip(self.node_dict[parent_name].q_low, self.node_dict[parent_name].q_up)]
        adjacent_center = [(low+up)/2 for low, up in zip(self.node_dict[adjacent_name].q_low, self.node_dict[adjacent_name].q_up)]
        dist_xy_in_dims = [(adjacent_center[i]-parent_center[i])**2 for i in range(2)]
        dist_high_in_dims = [(adjacent_center[i]-current_center[i])**2 for i in range(2, len(current_center))]
        # Distance between theta need to take wrap-around into account. 
        if self.has_theta:
            current_theta = (current_center[2] + 100*np.pi) % (2*np.pi) 
            adjacent_theta = (adjacent_center[2] + 100*np.pi) % (2*np.pi) 
            # Distance between two theta should be less than pi (it is free to choose clockwise or counterclockwise direction).
            theta_diff = abs(current_theta - adjacent_theta)
            theta_diff = theta_diff if theta_diff < np.pi else 2*np.pi-theta_diff
            dist_high_in_dims[0] = theta_diff**2
        # Weighted sum of distances in xy and higher order dimensions. 
        dist = w_xy * sqrt(sum(dist_xy_in_dims)) + w_high * sqrt(sum(dist_high_in_dims))
        return dist

    def assign_next_direction(self, specified_states, direction=None):
        """ Assign next_state based on the expected moving direction """
        print('\nAssign next_state based on the expected moving direction')
        for q_idx in specified_states:
            node = self.node_dict[q_idx]
            old_next = node.next_state
            assert node.identity == 0, 'Not assign next_state for the goal'
            # Find node in the adjacent list of the current node that biggest move towards the expected direction. 
            incumbent_move = -const.inf
            for adjacent_name in node.adjacent.keys():
                if adjacent_name in node.tried_next_states:
                    continue
                adjacent_move = self.compute_expected_move(adjacent_name, direction=direction)
                if adjacent_move > incumbent_move:
                    incumbent_move = adjacent_move
                    node.next_state = adjacent_name
            node.tried_next_states.append(node.next_state)
            print('\nName:', q_idx)
            print('adjacent nodes:', node.adjacent.keys())
            print('parent:', node.parent)
            print('tried_next_states:', node.tried_next_states)
            print('new next_state:', node.next_state)
            # TODO: Handle this assert. 
            assert node.next_state != old_next, 'All nodes in adjacent list of current node has been tried.'

    def compute_expected_move(self, adjacent_name, direction=None):
        """ 
        Move is measured as the weighted sum of coordinate and distance between theta and the expected direction.
        TODO: Currently, only consider x, y coordinates are all positive values.
        TODO: Need to wrap theta when compute distance in theta?
        """
        assert self.has_theta, 'The measure of move need to be re-defined without theta'
        w_xy = 1. 
        w_theta = 1000.
        node = self.node_dict[adjacent_name]
        if direction=='up':
            y_center = (node.q_low[1] + node.q_up[1])/2
            theta_center = abs((node.q_low[2] + node.q_up[2])/2 - 3*np.pi/2)
            expected_move = w_xy * y_center + w_theta * theta_center
        elif direction=='right':
            x_center = (node.q_low[0] + node.q_up[0])/2
            theta_center = abs((node.q_low[2] + node.q_up[2])/2 - np.pi)
            expected_move = w_xy * x_center + w_theta * theta_center
        elif direction=='right_up':
            x_center = (node.q_low[0] + node.q_up[0])/2
            theta_center = abs((node.q_low[2] + node.q_up[2])/2 - 5*np.pi/4)
            expected_move = w_xy * x_center + w_theta * theta_center
        else:
            assert False, 'Direction is undefined'
        return expected_move

    def assign_partition_dist(self, specified_states):
        """ 
        Among all partitions in the label associated to the transition to next_state (next_state may not be parent), 
        assign the one corresponding to the posterior that is closest to next_state.

        NOTE: The distance only takes higher order dimensions into consideration. Posteriors of the same abstract state under 
              different partitions should be only different in higher dimensions, since x, y are not affected by input in one step.

        TODO: When the next state is the goal, does it make sense to consider distance to its center pi (goal theta range [0, 2*pi))?

        TODO: Distance measure could be the weighted sum of distance between posterior and next_state
              and distance between posterior and current state (in order to penalize change of angle). 
        """ 
        print('\nAssign partition based on distance between posterior and next_state')
        # Change theta range of the goal to [0, pi] so its center used in computing distance is pi/2.
        for node in self.node_dict.values():
            if node.identity == 1:
                node.q_low[2] = 0
                node.q_up[2]  = np.pi
        # Assign controller partition based on distance between posterior and the next state. 
        # NOTE: By the convention of adding nodes to the graph, q_idx in keys (q_idx, p_idx) of post_dict is also name of the corresponding node in node_dict.
        for q_idx in specified_states:
            node = self.node_dict[q_idx]
            assert node.identity == 0, 'Not assign partition for the goal'
            incumbent_dist = const.inf
            for p_idx in node.adjacent[node.next_state]:  
                succ_low, succ_up = self.post_dict[(q_idx, p_idx)]
                q_succ_dist = self.compute_state_post_dist(node.next_state, succ_low, succ_up) 
                if q_succ_dist < incumbent_dist:
                    incumbent_dist = q_succ_dist
                    node.contr_partition = p_idx

    def compute_state_post_dist(self, name, succ_low, succ_up):
        """
        Compute Euclidean distance between center of an abstract state and center of a posterior, 
        where only higher order dimensions are considered for the distance.
        """
        q_low = self.node_dict[name].q_low
        q_up  = self.node_dict[name].q_up
        x_dim = len(q_low)
        q_center = [(q_low[i]+q_up[i])/2 for i in range(2, x_dim)]
        succ_center = [(succ_low[i]+succ_up[i])/2 for i in range(2, x_dim)]
        dist_in_dims = [(q_dim-succ_dim)**2 for q_dim, succ_dim in zip(q_center, succ_center)]
        # Distance between theta need to take wrap-around into account. 
        if self.has_theta:
            q_theta = (q_center[0] + 100*np.pi) % (2*np.pi) 
            succ_theta = (succ_center[0] + 100*np.pi) % (2*np.pi) 
            # Distance between two theta should be less than pi (it is free to choose clockwise or counterclockwise direction).
            theta_diff = abs(succ_theta - q_theta)
            theta_diff = theta_diff if theta_diff < np.pi else 2*np.pi-theta_diff
            dist_in_dims[0] = theta_diff**2
        dist = sqrt(sum(dist_in_dims))
        #print('abst state name:', name)
        #print('q_low:', q_low)
        #print('q_up:', q_up)
        #print('succ_low:', succ_low)
        #print('succ_up:', succ_up)
        #print('q_center:', q_center)
        #print('succ_center:', succ_center)
        #print('dist_in_dims:', dist_in_dims)
        #print('dist:', dist)
        return dist


    def assign_next_state_manual(self):     
        """ Assign next state for some nodes manually """
        frm_list = [290,350, 341, 293]
        to_list  = [292,398, 330, 281]
        for frm_name, to_name in zip(frm_list, to_list):
            self.node_dict[frm_name].next_state = to_name

    def assign_partition_manual(self):
        for node in self.node_dict.values():
            if node.identity == 1:
                continue
            node.next_state = node.parent      
        #self.node_dict[23].next_state = 92
        #self.node_dict[33].next_state = 41
        #self.node_dict[41].next_state = 92
        #self.node_dict[53].next_state = 92
        #self.node_dict[75].next_state = 92
    
        self.node_dict[23].contr_partition = 1
        self.node_dict[33].contr_partition = 1
        self.node_dict[41].contr_partition = 1
        self.node_dict[53].contr_partition = 1
        self.node_dict[75].contr_partition = 1


if __name__ == "__main__":

    root = './data/'

    #root = './data_16angles_40/'

    config = int(sys.argv[1])
    print('config:', config, end='\n\n')

    load_safe_graph = True
    if load_safe_graph:
        fin_name = root + 'safe_graph_c' + str(config)
        with open(fin_name, 'rb') as fin:
            safe_graph = pickle.load(fin)
        fin.close()        

    live_graph = LivenessGraph()
    live_graph.__dict__.update(safe_graph.__dict__)
    t0 = time.time() 
    live_graph.construct()    

    # Assign one controller partition and a next state to each node in liveness graph.
    # NOTE: Current theta range of the goal is [0, 2*pi), which may need be modified based on metrics used in assigning partitions.
    specified_states = set([name for name, node in live_graph.node_dict.items() if node.identity==0])
    live_graph.assign_next_be_parent(specified_states)
    live_graph.assign_partition_dist(specified_states)
    t1 = time.time()  
    print('Liveness graph elapsed time:', t1-t0) 

    for name, node in live_graph.node_dict.items():
        break
        print('\nNode name:', name)
        #print('q_low:', node.q_low)
        #print('q_up:',  node.q_up)
        #print('Number of adjacent nodes = ', len(node.adjacent))
        #print('adjacent nodes:', node.adjacent.keys())   
        #print('adjacent:', node.adjacent) 
        #print('parent:', node.parent)
        #print('label:', node.label)
        #print('tried_next_states:', node.tried_next_states)
        #print('next_state:', node.next_state)
        #print('contr_partition:', node.contr_partition)
    print('\nLiveness graph # of nodes = ', live_graph.num_nodes)

    # Print the trajectory of nodes from the given node to the goal. 
    #name = 108
    #node = live_graph.node_dict[name]
    #while node.next_state is not None:
    #    break
    #    print(node.next_state)
    #    node = live_graph.node_dict[node.next_state]

    save_live_graph = True
    if save_live_graph:        
        fout_name = root + 'live_graph_c' + str(config)
        with open(fout_name, 'wb') as fout:
            pickle.dump(live_graph, fout)
        fout.close()