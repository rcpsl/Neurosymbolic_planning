import numpy as np
import pickle
import random
from math import sqrt
import time

from PosteriorGraph import *


class ReducedGraph(object):
    def __init__(self):
        """
        Create reduced_adjacent for each node to represent a sub-graph of PosteriorGraph by selecting a subset of transitions,
        which refers to a subset of adjacent nodes and one (a subset of) controller partition(s) associated to each transition.
        ReducedGraph is used for reducing the number NNs need to be trained (each transition corresponds to a NN). 
        NOTE: Should not modify node.adjacent in this class. 

        TODO: Combine this class with SafetyReducedGraph to avoid replicate backtrack().  

        - If goal is in adjacent, should always include it in reduced_adjacent.
        """
        pass

    def construct(self):
        # First exclude unsafe states and partitions with respect to obstacles that are known before runtime. 
        # TODO: Currently I think it is same to backtrack all obstacles together and to backtrack partitioned subsets of obstacles separately, 
        # even when have unsafe states in multiple steps. Need to make sure on this.  
        # NOTE: In this backtracking, not decrease self.num_nodes when delete nodes from node_dict (at the end of backtrack), 
        # in order for runtime obstacles to obtain the correct names (self.num_nodes) when adding to node_dict in SafetyReducedGraph. 
        print('\nTotal number of nodes:', self.num_nodes)
        self.backtrack()
        print('\nNumber of safe nodes: %d\n' % len(self.node_dict))

        for frm_name, frm_node in self.node_dict.items():
            frm_node.reduced_adjacent = {}
            self.reduction_trivial(frm_name)
            #for i in range(5):
            #    self.reduction_one_partition(frm_name)
 
    def reduction_trivial(self, frm_name):
        """ No reduction, just copy adjacent to reduced_adjacent """
        frm_node = self.node_dict[frm_name]
        for to_name, label in frm_node.adjacent.items():
            frm_node.reduced_adjacent[to_name] = label.copy()

    def reduction_one_partition(self, frm_name):
        """ 
        Keep all to nodes in adjacent, but select only one controller partition associated to each transition.
        The selected partition should have not been selected before for the transition. 
        Try to select different partitions associated to different transitions leaving the same node. 

        - When random select one partition, not consider label_used.
        """
        frm_node = self.node_dict[frm_name]
        label_used = set()
        for to_name, label in frm_node.adjacent.items():
            if frm_node.reduced_adjacent.get(to_name) is None:
                frm_node.reduced_adjacent[to_name] = set()
            # All partitions associated to this transition have been selected. 
            label_left = label - frm_node.reduced_adjacent[to_name] 
            if not label_left:
                continue

            # Try to select different partitions for different transitions. 
            # If no more different partitions available, undo the step of restricting to different partitions. 
            label_left = label_left - label_used
            if not label_left:
                label_left = label - frm_node.reduced_adjacent[to_name]
                # TODO: Should reset label_used=set()?
            
            #selected_pid = self.select_partition_dist(frm_name, to_name, label_left)
            selected_pid = self.select_partition_random(label_left)
            frm_node.reduced_adjacent[to_name].add(selected_pid)
            label_used.add(selected_pid)
            
    def select_partition_random(self, label):
        """ Select a partition randomly """
        selected_pid = random.sample(label, 1)[0]
        return selected_pid 

    def select_partition_dist(self, frm_name, to_name, label):
        """
        Among all partitions in label, select the one corresponding to the posterior that is closest to the node transit to. 

        NOTE: The distance only takes higher order dimensions into consideration. Posteriors of the same abstract state under 
              different partitions should be only different in higher dimensions, since x, y are not affected by input in one step.
        """
        # Change theta range of the goal to [0, pi] so its center used in computing distance is pi/2.
        to_node = self.node_dict[to_name]
        if to_node.identity == 1:
            to_node.q_low[2] = 0
            to_node.q_up[2]  = np.pi
        # Select partition by comparing distance. 
        selected_pid = None
        incumbent_dist = const.inf
        for pid in label:  
            succ_low, succ_up = self.post_dict[(frm_name, pid)]
            next_succ_dist = self.compute_state_post_dist(to_node.q_low, to_node.q_up, succ_low, succ_up) 
            if next_succ_dist < incumbent_dist:
                incumbent_dist = next_succ_dist
                selected_pid = pid
        return selected_pid

    def compute_state_post_dist(self, q_low, q_up, succ_low, succ_up):
        """
        Compute Euclidean distance between center of an abstract state and center of a posterior, 
        where only higher order dimensions are considered for the distance.
        """
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
        return dist

    def backtrack(self):
        """ Backtrack to find safe states and partitions """
        # NOTE: When backtracking, should use adjacent instead of reduced_adjacent. 
        # Initially, unsafe states should only be obstacles.
        new_unsafe_states = [name for name, node in self.node_dict.items() if node.is_safe == False]
        print('\nStep 0 new unsafe states (total %d):' % len(new_unsafe_states))
        print(new_unsafe_states)
        # Backtracking until reach the fix point that there is no new unsafe states.
        for i in range(1000000):
            new_unsafe_states = self.backtrack_one_step(new_unsafe_states) 
            if not new_unsafe_states:
                break
            print('\nStep %d new unsafe states: (total %d)' % (i+1, len(new_unsafe_states)))
            print(new_unsafe_states)
            #for state in new_unsafe_states:
            #    print('\nState: ', state)
            #    print(self.node_dict[state].q_low)
            #    print(self.node_dict[state].q_up)
        
        # Only keep safe states in safety graph.
        remove_keys = []
        for name, node in self.node_dict.items():
            if not node.is_safe:
                remove_keys.append(name)
        for name in remove_keys:
            del self.node_dict[name]       
            #self.num_nodes -= 1  

    def backtrack_one_step(self, previous_unsafe_states):
        """ Backtrack one step to find new unsafe states """
        new_unsafe_states = []
        for name, node in self.node_dict.items():
            # A node with empty adjacent list could be an unsafe state that has been discovered, goal, or safe state without outgoing edge.  
            #if not node.is_safe or node.identity == 1:
            if len(node.adjacent) == 0:
                continue
            # A controller partition is unsafe at the current node if there is a transition 
            # from the current node to an unsafe node and has the partition in the label. 
            unsafe_partitions = set()
            for state in previous_unsafe_states:
                label = node.adjacent.get(state)
                if label:
                    unsafe_partitions = unsafe_partitions.union(label)
            # Remove unsafe partitions from labels associated to all transitions leaving the current node. 
            for next_node, label in node.adjacent.items():
                node.adjacent[next_node] = label - unsafe_partitions
            # Remove transitions associated with empty label.
            utility.dict_remove_empty(node.adjacent)
            # A node is unsafe if there is no transition leaves the node after removing unsafe transitions.
            if len(node.adjacent) == 0:
                node.is_safe = False
                new_unsafe_states.append(name)
        return new_unsafe_states



if __name__ == "__main__":
    random.seed(0)

    root = './data/'

    #root = './data_16angles_40/'

    load_post_graph = True
    if load_post_graph:
        fin_name = root + 'post_graph'
        with open(fin_name, 'rb') as fin:
            post_graph = pickle.load(fin)
        fin.close()

    reduced_graph = ReducedGraph()
    reduced_graph.__dict__.update(post_graph.__dict__)        

    reduced_graph.construct()

    #for frm_name, frm_node in reduced_graph.node_dict.items():
    #    print('frm_name %d, reduce # adjacent nodes: %d -> %d' % (frm_name, len(frm_node.adjacent), len(frm_node.reduced_adjacent)))

    count_transitions = 0
    for frm_name, frm_node in post_graph.node_dict.items():
        for to_name, label in frm_node.reduced_adjacent.items():
            count_transitions += len(label)
        #print('\nNode name:', name)
        #print('reduced_adjacent nodes:', node.reduced_adjacent.keys())   
        #print('reduced_adjacent:', node.reduced_adjacent) 
    print('\nTotoal number of transitions:', count_transitions)

    save_reduced_graph = True
    if save_reduced_graph:        
        fout_name = root + 'reduced_graph'
        with open(fout_name, 'wb') as fout:
            pickle.dump(reduced_graph, fout)
        fout.close()  