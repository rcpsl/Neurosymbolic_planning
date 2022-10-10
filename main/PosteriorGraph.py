import numpy as np
from scipy.io import loadmat
import pickle
import polytope as pc
import time

import constant as const
import utility as utility


class Node(object):
    """ Each node corresponds to an abstract state, including states in free space, obstacles, and the goal """
    def __init__(self, q_low, q_up, identity, is_safe):
        # Each abstract state is a hyper-rectangle, which is completely determined by a list of lower bounds (q_low) and upper bounds (q_up) in all dimensions.
        self.q_low = q_low 
        self.q_up  = q_up
        self.adjacent = {} # Nodes that can be reached in one step from the current node based on posterior.
        self.connected = [] # Nodes that can be connected to the current node by a straight line.
        self.identity = identity # The abstract state belongs to <-1: obstacles>, <0: free space>, or <1: goal>.
        self.is_safe = is_safe # Initially only obstacles are unsafe, more states would be unsafe during the backtracking.    
        
        # Polytope representation of each abstract state (in free state space).
        self.polytope = None

        # Select a subset of transitions leaving this node (in ReducedGraph).
        self.reduced_adjacent = {}

        # The following properties are used in liveness graph, for BFS and assigning controller partitions.          
        self.color = None # 0: white, 1: grey, 2: black
        self.d = None
        self.parent = None 
        #self.label = None # Label associated to the transition from its parent to this node.
        self.contr_partition = None # Assigned controller partition.
        self.next_state = None # Assigned state to which this node transit under the assigned controller partition. 
        
        # next_state that have been tried, but not good maybe due to no data can be collected. 
        self.tried_next_states = [] 

        
class PosteriorGraph(object):
    def __init__(self):
        """
        Nodes in posterior graph correspond to abstract states, labels are controller partitions, and transitions are determined based on posteriors.
        Some conventions:
        - When build post_dict, only consider the case where the number of controller partitions at every abstract state is same.
          Furthermore, posteriors saved by TIRA should be in the order that starts with all posteriors (under different partitions) of state 1, then state 2, and so on. 

        - Add nodes corresponding to free space states before adding obstacles and goal. In this way, indices of states in states_low/up
          are names of the corresponding nodes in node_dict. 

        Notice that posteriors may be out of the state space defined by x_span in some dimensions, including higher order states, 
        but this is fine for the purpose of checking intersection.

        NOTE: After adding nodes to node_dict, their name should never be changed, 
        # despite adding/removing nodes in ReducedGraph, SafetyReducedGraph, LivenessGraph, and so on.
        """
        self.node_dict = {}
        self.num_nodes = 0

    def load_files(self, load_cells_file, load_post_file):
        """ Load files and set up properties """
        # Load file of workspace setup and partitioned cells.
        cells_file          = loadmat(load_cells_file)
        states_low          = cells_file['states_low']
        states_up           = cells_file['states_up']
        self.partitions_low = cells_file['partitions_low']
        self.partitions_up  = cells_file['partitions_up']
        self.wksp           = cells_file['wksp']
        self.has_theta      = cells_file['has_theta']
        self.x_span         = cells_file['x_span']
        self.x_num          = cells_file['x_num']
        self.goal           = cells_file['goal']
        self.obstacles      = cells_file['obstacles']
        self.system_dict    = cells_file['system_dict']
        self.disturb_bound  = cells_file['disturb_bound']
        # Load file of posteriors.
        post_file           = loadmat(load_post_file)
        posteriors_low      = post_file['posteriors_low']
        posteriors_up       = post_file['posteriors_up']
        print('states shape:', states_low.shape)
        print('partitions shape:', self.partitions_low.shape)
        print('posteriors shape:', posteriors_low.shape)
        return states_low, states_up, posteriors_low, posteriors_up

    def construct(self, states_low, states_up, posteriors_low, posteriors_up):
        """ Construct the posterior graph """
        self.sys_choice = self.system_dict['sys_choice']
        if self.sys_choice == 2:
            self.x_dim = 3
            self.u_dim = 1
        else:
            assert False, 'System choice is not defined'
        assert self.x_dim == len(self.x_span[0]), 'x_dim mismatch'
        print('\nstate dim: ', self.x_dim)
        print('action dim:', self.u_dim)
        
        # Inflate posteriors to take disturbance into account. 
        print('disturb low:', self.disturb_bound[0])
        print('disturb up: ', self.disturb_bound[1])
        for succ_low, succ_up in zip(posteriors_low, posteriors_up):
            for i in range(self.x_dim):
                succ_low[i] = succ_low[i] - self.disturb_bound[0][i] + const.err
                succ_up[i]  = succ_up[i]  + self.disturb_bound[1][i] - const.err

        # Normalize theta and add nodes. 
        if self.has_theta:
            states_low, states_up, posteriors_low, posteriors_up = self.normalize_theta(states_low, states_up, posteriors_low, posteriors_up)
            self.theta_partitions = self.find_theta_partitions()
            print('theta_partitions:', self.theta_partitions)
        self.add_all_nodes(states_low, states_up)
        print('Done adding nodes')
        
        # TODO: Whether two abstract states are connectable only depends on x, y dimension, 
        # no need redundantly check when only higher order states are different.
        obst_bdry = []
        for o in self.obstacles:
            p1 = [o[0][0]-const.err, o[0][1]-const.err]
            p2 = [o[0][0]-const.err, o[1][1]+const.err]
            p3 = [o[1][0]+const.err, o[1][1]+const.err]
            p4 = [o[1][0]+const.err, o[0][1]-const.err]
            obst_bdry.append([p1, p2])
            obst_bdry.append([p2, p3])
            obst_bdry.append([p3, p4])
            obst_bdry.append([p1, p4])
        count = 0     
        for i in range(self.num_nodes):
            count += 1
            if count % 2000 == 0:
                print('Done %d check connectivity' % count)
            # If do not want self-loops, just let every node to be not connectable to itself.
            # self.node_dict[i].connected.append(i)
            for j in range(i+1, self.num_nodes):
                if self.node_dict[i].identity == -1 or self.node_dict[j].identity == -1:
                    # Every node is connectable to all obstacles, including workspace boundaries.
                    self.node_dict[i].connected.append(j)
                    self.node_dict[j].connected.append(i) 
                    continue
                if self.is_connectable(obst_bdry, self.node_dict[i], self.node_dict[j]):
                    self.node_dict[i].connected.append(j)
                    self.node_dict[j].connected.append(i) 
        for name, node in self.node_dict.items():
            # No need fill connected property for goal and obstacles. 
            if node.identity != 0:
                node.connected = []
            #print('name:', name)
            #print(node.connected, end='\n\n')
        print('Done checking connectivity')

        # post_dict: Key is a tuple (q_idx, p_idx), where q_idx and p_idx are indices of abstract states and controller partitions
        #            in states_low/up and partitions_low/up, respectively.
        #            Value is the corresponding posterior taken from posteriors_low/up.
        # Notice that by the convention of adding nodes to the graph, q_idx is also name of the corresponding node in node_dict. 
        # Notice that states_low/up only contain abstract states in free space, i.e., not including obstacles and the goal.
        self.post_dict = {}
        num_states     = len(states_low)
        num_partitions = len(self.partitions_low)
        num_posteriors = len(posteriors_low)
        assert num_states * num_partitions == num_posteriors, 'Number of states, partitions, and posteriors does not match'
        for succ_idx in range(num_posteriors):
            q_idx = succ_idx // num_partitions
            p_idx = succ_idx % num_partitions
            self.post_dict[(q_idx, p_idx)] = [posteriors_low[succ_idx], posteriors_up[succ_idx]]
        #for idx, succ in self.post_dict.items():
        #    print(idx, end=': ')
        #    print(succ)
        
        #q1_low = np.array([0.5, 0.5, 0]) 
        #q1_up  = np.array([1.0, 1.0, np.pi/4])
        #q2_low = np.array([0.99, 0.5, np.pi/4]) 
        #q2_up  = np.array([1.6, 1.0, 2*np.pi])
        #temp = self.is_intersect(q1_low, q1_up, q2_low, q2_up)
        #print(temp)

        t0 = time.time()  
        # Add transitions based on intersection between abstract states and posteriors.
        for node in self.node_dict.values():
            if node.identity == 0:
                for i in range(self.num_nodes):
                    node.adjacent[i] = set()
        count = 0             
        for idx, succ in self.post_dict.items():
            count += 1
            if count % 1000 == 0:
                print('Done %d posteriors' % count)
            q_idx, p_idx = idx[0], idx[1]
            assert self.node_dict[q_idx].identity == 0, 'obstacles and goal should not appear in post_dict'
            for q_next_idx in self.node_dict[q_idx].connected:
                q_next_low = self.node_dict[q_next_idx].q_low
                q_next_up  = self.node_dict[q_next_idx].q_up
                if self.is_intersect(q_next_low, q_next_up, succ[0], succ[1]):
                    self.node_dict[q_idx].adjacent[q_next_idx].add(p_idx)
        for name, node in self.node_dict.items():
            utility.dict_remove_empty(node.adjacent)
        t1 = time.time()  
        print('Posterior Graph elapsed time:', t1-t0)      

        # Setup polytope representation for each node (except the goal and obstacles), 
        # which is used for checking intersection with linear regions when project NN weights. 
        for node in self.node_dict.values():
            if node.identity == 0:
                box = [[node.q_dim_low, node.q_dim_up] for node.q_dim_low, node.q_dim_up in zip(node.q_low, node.q_up)]
                node.polytope = pc.box2poly(box)
                #print(pc.extreme(node.polytope))

    def normalize_theta(self, states_low, states_up, posteriors_low, posteriors_up):
        """ 
        For the range of theta in both abstract states and posteriors, bring the lower bound to be within [0, 2*pi),
        while the upper bound could be greater than 2*pi in order to be greater than the lower bound. 

        Notice that after normalization, even theta partitions in abstract states may not be within the theta span defined in x_span.
        E.g. suppose the theta span [13*pi/8, 19*pi/8] is partitioned into [13*pi/8, 15*pi/8], [15*pi/8, 17*pi/8], and [17*pi/8, 19*pi/8], 
             then after normalization the last partition is [pi/8, 3*pi/8], which is not within the theta span.
        """
        # Normalize theta partitions in abstract states.
        for q_low, q_up in zip(states_low, states_up):
            # Bring theta into [0, 2*pi).
            q_low[2] = (q_low[2] + 100*np.pi) % (2*np.pi) 
            q_up[2]  = (q_up[2] + 100*np.pi) % (2*np.pi) 
            # When the partition across the positive x direction, incease the upper bound by 2*pi.
            if q_low[2] > q_up[2]: 
                q_up[2] += 2*np.pi
        for q_low, q_up in zip(states_low, states_up): 
            #print(q_low)
            #print(q_up, end='\n\n')   
            assert 0 <= q_low[2] < 2*np.pi, 'theta lower bound is out of [0, 2*pi)'
            assert q_low[2] < q_up[2]  < q_low[2]+2*np.pi, 'theta upper bound is not as expected' 

        # Normalize the range of theta in posteriors.
        for succ_low, succ_up in zip(posteriors_low, posteriors_up):
            # When the range of theta spans more than 2*pi, let it be [0, 2*pi)
            if succ_up[2] - succ_low[2] >= 2*np.pi:
                succ_low[2] = 0
                succ_up[2]  = 2*np.pi - const.err
                continue
            # Bring theta into [0, 2*pi).     
            succ_low[2] = (succ_low[2] + 100*np.pi) % (2*np.pi) 
            succ_up[2]  = (succ_up[2] + 100*np.pi) % (2*np.pi)  
            # When the range of theta across the positive x direction, incease the upper bound by 2*pi.
            if succ_low[2] > succ_up[2]: 
                succ_up[2] += 2*np.pi
        for succ_low, succ_up in zip(posteriors_low, posteriors_up):
            assert 0 <= succ_low[2] < 2*np.pi, 'theta lower bound is out of [0, 2*pi)'
            assert succ_low[2] < succ_up[2] < succ_low[2]+2*np.pi, 'theta upper bound is not as expected'
        return states_low, states_up, posteriors_low, posteriors_up
    
    def find_theta_partitions(self):
        """ Return a list of all theta partitions that appear in abstract states """
        # TODO: Need to be customized if the range of theta is partitioned non-uniformly. 
        theta_intervals = np.linspace(self.x_span[0][2], self.x_span[1][2], self.x_num[0][2]+1).tolist()
        theta_partitions = []
        for i in range(len(theta_intervals)-1):
            p = [theta_intervals[i], theta_intervals[i+1]]
            theta_partitions.append(p)
        # Normalize theta partition such that the lower bound is in [0, 2*pi),
        # while the upper bound could be greater than 2*pi in order to be greater than the lower bound.  
        for p in theta_partitions:
            p[0] = (p[0] + 100*np.pi) % (2*np.pi) 
            p[1] = (p[1] + 100*np.pi) % (2*np.pi)
            if p[0] > p[1]:
                p[1] += 2*np.pi
        for p in theta_partitions:
            assert 0 <= p[0] < 2*np.pi and p[0] < p[1] < p[0] + 2*np.pi, 'theta partition is out of bound'
        #print(theta_partitions)
        return theta_partitions

    def is_intersect(self, q1_low, q1_up, q2_low, q2_up):
        """ 
        Check whether two hyper-rectangles (e.g. one abstract state and one posterior) intersect.
        Two hyper-rectangles intersect if and only if they have overlap in every dimension. 
        """
        x_dim = len(self.x_span[0])
        is_isect = True
        if not self.has_theta:
            for q1_dim_low, q1_dim_up, q2_dim_low, q2_dim_up in zip(q1_low, q1_up, q2_low, q2_up):
                if not self.is_interval_overlap(q1_dim_low, q1_dim_up, q2_dim_low, q2_dim_up):
                    is_isect = False
                    break
        else:
            indices = list(range(x_dim))
            indices.remove(2)
            for i in indices:
                if not self.is_interval_overlap(q1_low[i], q1_up[i], q2_low[i], q2_up[i]):
                    is_isect = False
                    return is_isect
            if not self.is_angle_overlap(q1_low[2], q1_up[2], q2_low[2], q2_up[2]):
                is_isect = False
        return is_isect

    def is_angle_overlap(self, t1_low, t1_up, t2_low, t2_up):
        """
        Check whether two angles t1 and t2 have overlap.

        After normalization, lower bound of theta should be within [0, 2*pi), while upper bound could be greater than 2*pi.
        When check overlap, bring the range of theta within [0, 2*pi] by splitting the range if necessary.
        """
        assert 0 <= t1_low < 2*np.pi and t1_low < t1_up < t1_low + 2*np.pi, 'theta should first be normalized'
        assert 0 <= t2_low < 2*np.pi and t2_low < t2_up < t2_low + 2*np.pi, 'theta should first be normalized'
        is_overlap = False
        if t1_up >= 2*np.pi and t2_up >= 2*np.pi:
            is_overlap = True
            return is_overlap
        if t1_up > 2*np.pi+const.err: # Should be strictly greater than 2*pi since do not want [0, 0] be a sub-range.
            # Split the range t1 into two sub-ranges.
            t1_sub1 = [t1_low, 2*np.pi]
            t1_sub2 = [0, t1_up-2*np.pi]
            if self.is_interval_overlap(t1_sub1[0], t1_sub1[1], t2_low, t2_up) or self.is_interval_overlap(t1_sub2[0], t1_sub2[1], t2_low, t2_up):
                is_overlap = True
        elif t2_up > 2*np.pi+const.err: 
            # Split the range t2 into two sub-ranges.
            t2_sub1 = [t2_low, 2*np.pi]
            t2_sub2 = [0, t2_up-2*np.pi]
            if self.is_interval_overlap(t1_low, t1_up, t2_sub1[0], t2_sub1[1]) or self.is_interval_overlap(t1_low, t1_up, t2_sub2[0], t2_sub2[1]):
                is_overlap = True
        else:
            if self.is_interval_overlap(t1_low, t1_up, t2_low, t2_up):
                is_overlap = True
        return is_overlap

    def is_interval_overlap(self, s1_low, s1_up, s2_low, s2_up):
        """ Check whether two intervals s1 and s2 have overlap """
        assert s1_up - s1_low >= const.err and s2_up - s2_low >= const.err, 'Really want an interval to be a point?'
        is_overlap = False
        if s2_low-const.err < s1_low < s2_up-const.err or s1_low-const.err < s2_low < s1_up-const.err:
            is_overlap = True
        return is_overlap

    def is_connectable(self, obst_bdry, n1, n2):
        """ Check if two abstract states can be connected by a straight line that does not intersect obstacles """
        # Instead of considering all straight lines connecting every pair of vertices, only consider the straight line connecting centers.
        is_connect = True
        n1_center = [(n1.q_low[0]+n1.q_up[0])/2, (n1.q_low[1]+n1.q_up[1])/2]
        n2_center = [(n2.q_low[0]+n2.q_up[0])/2, (n2.q_low[1]+n2.q_up[1])/2]
        for bdry in obst_bdry:
            if utility.is_seg_intersect(n1_center, n2_center, bdry[0], bdry[1]):
                is_connect = False
                break
        return is_connect

    def add_all_nodes(self, states_low, states_up):
        """ Add nodes corresponding to abstract states in free space, obstacles, and the goal """
        x_dim = len(self.x_span[0])
        # Add nodes corresponding to abstract states in free space. 
        for q_low, q_up in zip(states_low, states_up):
            self.add_node(q_low, q_up, identity=0, is_safe=True)

        # Add nodes corresponding to obstacles, including workspace boundaries. 
        # For given obstacles in [x, y] dimensions, fill higher order dimensions with the entire spans.
        for o in self.obstacles:
            self.add_node(o[0], o[1], identity=-1, is_safe=False)
        
        # Add four nodes corresponding to workspace boundaries, where higher order states are unbounded.
        """
        c = 10.
        # Left
        q_low = [self.x_span[0][0]-c, self.x_span[0][1]-c] + [-const.inf] * (x_dim-2)  
        q_up  = [self.x_span[0][0],   self.x_span[1][1]+c] + [ const.inf] * (x_dim-2) 
        self.add_node(q_low, q_up, identity=-1, is_safe=False)
        # Right
        q_low = [self.x_span[1][0],   self.x_span[0][1]-c] + [-const.inf] * (x_dim-2)
        q_up  = [self.x_span[1][0]+c, self.x_span[1][1]+c] + [ const.inf] * (x_dim-2)
        self.add_node(q_low, q_up, identity=-1, is_safe=False)
        # Bottom
        q_low = [self.x_span[0][0], self.x_span[0][1]-c] + [-const.inf] * (x_dim-2) 
        q_up  = [self.x_span[1][0], self.x_span[0][1]  ] + [ const.inf] * (x_dim-2)
        self.add_node(q_low, q_up, identity=-1, is_safe=False)
        # Up
        q_low = [self.x_span[0][0], self.x_span[1][1]  ] + [-const.inf] * (x_dim-2)
        q_up  = [self.x_span[1][0], self.x_span[1][1]+c] + [ const.inf] * (x_dim-2)
        self.add_node(q_low, q_up, identity=-1, is_safe=False)
        """

        # Add node corresponding to the goal.
        self.add_node(self.goal[0], self.goal[1], identity=1, is_safe=True)

        # Instead of [-inf, inf], set theta range be [0, 2*pi) for obstacles and the goal in order to pass some sanity checks (no functional reason).
        if self.has_theta:
            for node in self.node_dict.values():
                if node.identity != 0:
                    node.q_low[2] = 0
                    node.q_up[2]  = 2*np.pi - const.err

        print('Number of nodes:', self.num_nodes)
        #for name, node in self.node_dict.items():
        #    print(name)
        #    print(node.q_low)
        #    print(node.q_up)
        #    print('identity:', node.identity)
        #    print('is_safe:', node.is_safe, end='\n\n')

    def add_node(self, q_low, q_up, identity, is_safe):
        """ Create a node and add it to node_dict """
        name = self.num_nodes
        new_node = Node(q_low, q_up, identity, is_safe)
        self.node_dict[name] = new_node
        self.num_nodes += 1
        return name



if __name__ == "__main__":
    
    root = './data/'

    #root = './cav_backup/scalability_grid/data_w1_552_320/' 
    #root = './cav_backup/scalability_linear/data_n10/' 
    load_cells_file = root+'cells.mat'
    load_post_file = root+'posteriors.mat'
    post_graph = PosteriorGraph()
    states_low, states_up, posteriors_low, posteriors_up = post_graph.load_files(load_cells_file, load_post_file)
    post_graph.construct(states_low, states_up, posteriors_low, posteriors_up)
    
    for name, node in post_graph.node_dict.items():
        break
        print('\nNode name:', name)
        print('q_low:', node.q_low)
        print('q_up: ', node.q_up)
        print('identity:', node.identity)
        print('is_safe:', node.is_safe)
        #print('connected:', node.connected)
        print('Number of adjacent nodes = ', len(node.adjacent))
        print('adjacent nodes:', node.adjacent.keys())   
        #print('adjacent:', node.adjacent) 
    print('\nPost graph # of nodes = ', post_graph.num_nodes)

    save_post_graph = True
    if save_post_graph:        
        fout_name = root+'post_graph'
        with open(fout_name, 'wb') as fout:
            pickle.dump(post_graph, fout)
        fout.close()
        
