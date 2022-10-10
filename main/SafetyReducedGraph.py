import numpy as np
import pickle
import sys
import time

from PosteriorGraph import *
from ReducedGraph import *


class SafetyReducedGraph(PosteriorGraph):
    def __init__(self):
        """
        ReducedSafetyGraph first add runtime obstacles to PosteriorGraph, then find unsafe states and partitons by backtracking,
        and only keep transitions selected by ReducedGraph. 
        """
        pass        

    def set_config(self, config):
        """ Add obstacles that are specific in each configuration during test (runtime obstacles) """
        #print('wks:', self.wksp[0][0])
        print('config:', config, end='\n\n')
        x_dim = len(self.x_span[0])

        if self.wksp==1 and config==0:
            o2_low = [0, 2] + [-const.inf] * (x_dim-2)
            o2_up  = [1, 3] + [ const.inf] * (x_dim-2)
            o2 = [o2_low, o2_up]
            o3_low = [2.5, 2] + [-const.inf] * (x_dim-2)
            o3_up  = [5.0, 3] + [ const.inf] * (x_dim-2)
            o3 = [o3_low, o3_up]
            self.runtime_obstacles = [o2, o3]

        elif self.wksp==1 and config==1:
            o2_low = [0.0, 2] + [-const.inf] * (x_dim-2)
            o2_up  = [0.5, 3] + [ const.inf] * (x_dim-2)
            o2 = [o2_low, o2_up]
            o3_low = [2.0, 2] + [-const.inf] * (x_dim-2)
            o3_up  = [5.0, 3] + [ const.inf] * (x_dim-2)
            o3 = [o3_low, o3_up]
            self.runtime_obstacles = [o2, o3]

        elif self.wksp==1 and config==2:
            o2_low = [0.0, 2] + [-const.inf] * (x_dim-2)
            o2_up  = [3.0, 3] + [ const.inf] * (x_dim-2)
            o2 = [o2_low, o2_up]
            o3_low = [4.5, 2] + [-const.inf] * (x_dim-2)
            o3_up  = [5.0, 3] + [ const.inf] * (x_dim-2)
            o3 = [o3_low, o3_up]
            self.runtime_obstacles = [o2, o3]

        elif self.wksp==1 and config==3:
            o2_low = [1.0, 2] + [-const.inf] * (x_dim-2)
            o2_up  = [3.5, 3] + [ const.inf] * (x_dim-2)
            o2 = [o2_low, o2_up]
            self.runtime_obstacles = [o2]

        elif self.wksp==1 and config==4:
            o2_low = [1.0, 2] + [-const.inf] * (x_dim-2)
            o2_up  = [3.5, 3] + [ const.inf] * (x_dim-2)
            o2 = [o2_low, o2_up]
            o3_low = [3.0, 3.0] + [-const.inf] * (x_dim-2)
            o3_up  = [3.5, 4.5] + [ const.inf] * (x_dim-2)
            o3 = [o3_low, o3_up]
            o4_low = [5.0, 2] + [-const.inf] * (x_dim-2)
            o4_up  = [5.5, 3] + [ const.inf] * (x_dim-2)
            o4 = [o4_low, o4_up]
            self.runtime_obstacles = [o2, o3, o4]     

        else:
            assert False, 'undefined wksp or config'
 
    def construct(self):
        """ 
        Construct safety graph by:
        1. Adding runtime obstacles to node_dict;
        2. Removing states that are covered by runtime obstacles (in x, y dimensions);
        3. Adding transitions to runtime obstacles. 
        NOTE: All updates should be done to adjacent, instead of reduced_adjacent. 
        """
        print('Length of node_dict at the beginning:', len(self.node_dict)) # At this point, self.num_nodes is not the actual number of nodes. 
        # 1. Add runtime obstacles to node_dict. 
        # Do this before removing nodes in order for obstacles to obtain the correct names (self.num_nodes). 
        runtime_obst_names = []
        for o in self.runtime_obstacles:
            name = self.add_node(o[0], o[1], identity=-1, is_safe=False)
            runtime_obst_names.append(name)
        self.num_nodes = len(self.node_dict)
        print('Length of node_dict after adding runtime obstacles:', self.num_nodes)
        print('Name of runtime obstacles in node_dict:', runtime_obst_names)
        #self.obstacles = self.obstacles.tolist()
        #self.obstacles.extend(self.runtime_obstacles)
        #print('Total # of obstacles:', len(self.obstacles))
        
        # 2. From node_dict, remove nodes whose x, y dimensions are covered by runtime obstacles, and remove all transitions to these nodes.
        nodes_to_remove = []
        for o in self.runtime_obstacles:
            nodes_to_remove.extend(self.find_states_in_range(o[0], o[1]))
        print('# nodes to remove:', len(nodes_to_remove))
        print('nodes_to_remove:', nodes_to_remove)
        for name in nodes_to_remove:
            del self.node_dict[name]
            self.num_nodes -= 1
        print('Length of node_dict after removing states covered obstacles:', self.num_nodes)

        # Remove transitions to nodes that are covered obstacles. 
        # NOTE: After this, some adjacent may become empty (i.e., no transition leaves the node). 
        # Only remove transitions from adjacent, no need to remove transitions from reduced_adjacent. 
        for frm_name, frm_node in self.node_dict.items():
            #print('\nfrm_name:', frm_name)
            #print('Before remove adjacent:', frm_node.adjacent.keys())
            for name in nodes_to_remove:
                try:
                    del frm_node.adjacent[name]
                except KeyError:
                    continue
            #print('After remove adjacent:', frm_node.adjacent.keys())

        # 3. Add transitions to runtime obstacles by checking intersection between them with posteriors of all free space states under all controller partitions. 
        # Instead of [-inf, inf], set theta range be [0, 2*pi) for obstacles and the goal in order to pass some sanity checks (no functional reason).
        if self.has_theta:
            for node in self.node_dict.values():
                if node.identity != 0:
                    node.q_low[2] = 0
                    node.q_up[2]  = 2*np.pi - const.err

        # NOTE: By the convention of adding nodes to the graph, q_idx in keys (q_idx, p_idx) of post_dict is also name of the corresponding node in node_dict.
        # Only add transitions to adjacent, no need to add transitions to reduced_adjacent. 
        for idx, succ in self.post_dict.items():
            q_idx, p_idx = idx[0], idx[1]
            # Should not add transition from nodes that have already been removed from node_dict 
            # (including both nodes covered by runtime obstacles and unsafe nodes in backtracking). 
            # NOTE: Some partitions have been removed during backtracking in ReducedGraph. Nevertheless, there is no need to exclude 
            # these partitions when adding transitions to runtime obstacles, since these partitions will just be removed again. 
            if q_idx not in self.node_dict.keys():
                continue
            for obst_name in runtime_obst_names:
                obst_low = self.node_dict[obst_name].q_low
                obst_up  = self.node_dict[obst_name].q_up
                if self.is_intersect(obst_low, obst_up, succ[0], succ[1]):
                    try:
                        self.node_dict[q_idx].adjacent[obst_name].add(p_idx)
                    except KeyError:
                        self.node_dict[q_idx].adjacent[obst_name] = {p_idx}

    def find_states_in_range(self, range_low, range_up):
        """ Find FREE states in node_dict that are contained in the given range """
        states_in_range = []
        x_dim = len(range_low)
        for name, node in self.node_dict.items():
            if node.identity != 0:
                continue
            if all([range_low[i]-const.err <= node.q_low[i] and node.q_up[i] <= range_up[i]+const.err for i in range(x_dim)]):
                states_in_range.append(name)
        return states_in_range

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
            self.num_nodes -= 1  
        #self.num_nodes = len(self.node_dict)

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

    def reduction(self):
        """ Reduce transitions in node_dict to only include those selected by ReducedGraph """ 
        # First remove keys in adjacent but not in reduced_adjacent
        # NOTE: After this, some adjacent may become empty (i.e., no transition leaves the node). 
        for frm_name, frm_node in self.node_dict.items():
            keys_to_remove = [k for k in frm_node.adjacent.keys() if k not in frm_node.reduced_adjacent.keys()] 
            for k in keys_to_remove:
                del frm_node.adjacent[k]
        
        # For each transition, remove controller partitions allowed by adjacent but not by reduced_adjacent. 
        for frm_name, frm_node in self.node_dict.items():
            for to_name, label in frm_node.adjacent.items():
                reduced_label = frm_node.reduced_adjacent[to_name]
                frm_node.adjacent[to_name] = label.intersection(reduced_label)
            # reduced_adjacent will not be further used. 
            frm_node.reduced_adjacent = None
        # Remove elements in adjacent if the corresponding labels are empty. 
        for node in self.node_dict.values():
            utility.dict_remove_empty(node.adjacent)


if __name__ == "__main__":

    root = './data/'
    
    #root = './data_16angles_40/'

    config = int(sys.argv[1])

    load_reduced_graph = True
    if load_reduced_graph:
        fin_name = root + 'reduced_graph'
        with open(fin_name, 'rb') as fin:
            reduced_graph = pickle.load(fin)
        fin.close()    

    safe_graph = SafetyReducedGraph()
    safe_graph.__dict__.update(reduced_graph.__dict__)
    #t0 = time.time()
    
    safe_graph.set_config(config)
    safe_graph.construct()
    safe_graph.backtrack()
    print('\nAfter backtracking, # safe nodes = ', safe_graph.num_nodes)

    safe_graph.reduction()
    print('After reduction, # safe nodes = ', safe_graph.num_nodes)

    #t1 = time.time()  

    for name, node in safe_graph.node_dict.items():
        break
        print('\nNode name:', name)
        print('q_low:', node.q_low)
        print('q_up:',  node.q_up)
        print('identity:', node.identity)
        print('is_safe:', node.is_safe)
        print('Number of adjacent nodes = ', len(node.adjacent))
        print('adjacent nodes:', node.adjacent.keys())   
        #print('adjacent:', node.adjacent) 
    #print('Safety graph elapsed time:', t1-t0)  

    save_safe_graph = True
    if save_safe_graph:        
        fout_name = root + 'safe_graph_c' + str(config) 
        with open(fout_name, 'wb') as fout:
            pickle.dump(safe_graph, fout)
        fout.close()    


         