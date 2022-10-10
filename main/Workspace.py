import numpy as np
from copy import deepcopy
from scipy.io import savemat

import constant as const


class Workspace(object):
    def __init__(self):
        # Whether heading angle theta is a state.
        self.has_theta = False 

    def set_config(self):
        """
        Set up system parameters, states space, controller space. 
 
        Some conventions on state space:

        - Let x, y correspond to the first two entries in all data structures related to states. 
        - If there is heading angle theta, let it be the third entry (index 2) in all data structures related to states.
        - Make sure the order of entries in states and controllers is consistent with that in TIRA and FORCES Pro.

        - It would be good to set the theta span and number of partitions in a way such that when uniformly partition the theta span, 
          directions along x, y axes are at center of some theta partitions. 

        - Let the lower bound of theta span be within [0, 2*pi), while the upper bound could be greater than 2*pi in order to be greater than the lower bound.
          This is only used as an assumption in Workspace.reconcile_theta().

        - Workspace is a rectangle in x, y dimensions and edges are aligned with x, y axes (this is used when automatically add boundaries as obstacles).
        
        - Choose partition parameters such that each abstract state is a subset of either obstacles or free space. 

        - If higher order states of the goal is bounded, or there is obstacles in higher order state space, the following functions need to be modified:
            is_goal_obstacles() wrap-around theta, PosteriorGraph.add_all_nodes(), MotionPlanner.check_theta().

        TODO: If the span of theta is less than 2*pi, then should add higher order obstacles to avoid theta be in the range where no abstract state can cover,
              otherwise, during test, when theta is not in any abstract state, there is no trained NN.

        Some conventions on controller space:

        - Entries in K_span_low/up is in row-first order, and b is attached at the end,  i.e.:
            K_11, ..., K_1n, K_21, ..., K_2n, ..., K_m1, ..., K_mn, b_1, ..., b_m
          This is consistent with that in TIRA and FORCES Pro, and is used as an assumption in check_gain_in_partition(), add_ap_constraints().
        
        - For K span, avoid setting lower and upper bounds for some entry be same, e.g. both 0, which causes infeasibility when call FORCES Pro. 

        TODO: Adaptive / non-uniform partition of state and controller space. 

        Meta-learning setting:

        - When define obstacles, only include those remain unchanged in all configurations. 
        """
        self.wksp = 1

        if self.wksp == 0:
            # NOTE: sys_choice should be consistent with system_choice in TIRA (System_description.m and UP_Jacobian_Bounds.m).
            sys_choice = 2 # Unicycle with constant speed (K,b as control inputs).
            dt = 0.1
            v = 1.
            self.system_dict = {'sys_choice': sys_choice, 'dt': dt, 'v': v}
            self.has_theta = True

            # Lower/upper bound of state space and number of partitions in each dimension.
            # State: [x, y, theta]
            x_span_low = [0, 0, 0]
            x_span_up  = [1.5, 1.5, 2*np.pi]
            self.x_num = [3, 3, 8]
            self.x_span = [x_span_low, x_span_up]
            x_dim = len(x_span_low)

            # Lower/upper bound of controller space and number of partitions in each dimension.
            # Control gain: [K1, K2, K3, b]
            K_span_low = [-4., -4., -0.2, -2.]
            K_span_up  = [ 4.,  4.,  0.2,  2.] 
            self.K_num = [4, 4, 1, 4]
            self.K_span = [K_span_low, K_span_up]
            
            # Higher order states of the goal could be unbounded (not even bounded by x_span), 
            # i.e., no requirement on higher order states in terms of reaching the goal.  
            goal_low = [1.0, 1.0] + [-const.inf] * (x_dim-2)
            goal_up  = [1.5, 1.5] + [ const.inf] * (x_dim-2)
            self.goal = [goal_low, goal_up]
            
            # Higher order states of obstacles should be unbounded (unsafe as long as hit obstacles in x, y dimension).
            # If needed, may add obstacles in higher order state space separately.
            o1_low = [0.5, 0.5] + [-const.inf] * (x_dim-2)
            o1_up  = [1.0, 1.0] + [ const.inf] * (x_dim-2)
            o1 = [o1_low, o1_up]
            self.obstacles = [o1]

        elif self.wksp == 1:
            # NOTE: sys_choice should be consistent with system_choice in TIRA (System_description.m and UP_Jacobian_Bounds.m).
            sys_choice = 2 # Unicycle with constant speed (K,b as control inputs).
            dt = 0.1
            
            v = 3.
            #v = 3.9 
            
            self.system_dict = {'sys_choice': sys_choice, 'dt': dt, 'v': v}
            self.has_theta = True

            # Bound on disturbance that captures unmodeled dynamics.
            d_low = [0, 0, 0]
            
            #d_up  = [0.07, 0.07, 0]
            d_up  = [0.1, 0.1, 0]
            #d_up  = [0.15, 0.15, 0]
            #d_up  = [0.2, 0.2, 0]

            self.disturb_bound = [d_low, d_up]

            # Lower/upper bound of state space and number of partitions in each dimension.
            # State: [x, y, theta]
            # CAV trajectory (552 states).
            #x_span_low = [0, 0, 0]
            #x_span_up  = [5, 5, 2*np.pi]
            #self.x_num = [10, 10, 8]
            #self.x_span = [x_span_low, x_span_up]
            #x_dim = len(x_span_low)

            x_span_low = [0, 0, np.pi/16]
            x_span_up  = [5, 5, np.pi/16+2*np.pi]
            self.x_num = [10, 10, 16]
            self.x_span = [x_span_low, x_span_up]
            x_dim = len(x_span_low)

            # Lower/upper bound of controller space and number of partitions in each dimension.
            # Control gain: [K1, K2, K3, b]
            # CAV trajectory (160 partitions).
            #K_span_low = [-2, -2, -0.5, -20]
            #K_span_up  = [ 2,  2,  0.5,  20] 
            #self.K_num = [2, 2, 2, 20]
            #self.K_span = [K_span_low, K_span_up]





            #K_span_low = [-0.1, -0.1, -0.1, -30]
            #K_span_up  = [ 0.1,  0.1,  0.1,  30] 
            #self.K_num = [2, 2, 2, 30]
            #self.K_span = [K_span_low, K_span_up]

            K_span_low = [-0.1, -0.1, -0.1, -40]
            K_span_up  = [ 0.1,  0.1,  0.1,  40] 
            self.K_num = [2, 2, 2, 40]
            self.K_span = [K_span_low, K_span_up]


            


            ###### 16 angles ######
            #K_span_low = [-0.1, -0.1, -0.1, -20]
            #K_span_up  = [ 0.1,  0.1,  0.1,  20] 
            #self.K_num = [2, 2, 2, 40]
            #self.K_span = [K_span_low, K_span_up]

            #K_span_low = [-0.1, -0.1, -0.1, -40]
            #K_span_up  = [ 0.1,  0.1,  0.1,  40] 
            #self.K_num = [2, 2, 2, 40]
            #self.K_span = [K_span_low, K_span_up]


            ###### 12 angles ######
            #K_span_low = [-0.1, -0.1, -0.1, -30]
            #K_span_up  = [ 0.1,  0.1,  0.1,  30]
            #self.K_num = [2, 2, 2, 90]
            #self.K_span = [K_span_low, K_span_up]
            
            # Higher order states of the goal could be unbounded (not even bounded by x_span), 
            # i.e., no requirement on higher order states in terms of reaching the goal.  
            goal_low = [4, 4.5] + [-const.inf] * (x_dim-2)
            goal_up  = [5, 5] + [ const.inf] * (x_dim-2)
            self.goal = [goal_low, goal_up]
            
            # Higher order states of obstacles should be unbounded (unsafe as long as hit obstacles in x, y dimension).
            # If needed, may add obstacles in higher order state space separately.
            o1_low = [0, 0.0] + [-const.inf] * (x_dim-2)
            o1_up  = [5, 0.5] + [ const.inf] * (x_dim-2)
            o1 = [o1_low, o1_up]
            #o2_low = [0, 2] + [-const.inf] * (x_dim-2)
            #o2_up  = [1, 3] + [ const.inf] * (x_dim-2)
            #o2 = [o2_low, o2_up]
            #o3_low = [2.5, 2] + [-const.inf] * (x_dim-2)
            #o3_up  = [5.0, 3] + [ const.inf] * (x_dim-2)
            #o3 = [o3_low, o3_up]
            o4_low = [1.0, 4.5] + [-const.inf] * (x_dim-2)
            o4_up  = [3.5, 5.0] + [ const.inf] * (x_dim-2)
            o4 = [o4_low, o4_up]
            self.obstacles = [o1, o4]

        else:
            assert False, 'workspace is undefined'
        
        if self.has_theta:
            assert 0 <= self.x_span[0][2] < 2*np.pi, 'theta lower bound is out of [0, 2*pi)'
            assert self.x_span[0][2] < self.x_span[1][2] <= self.x_span[0][2]+2*np.pi, 'theta upper bound is not as expected'
           
    def partition_space(self):
        """ Partition state and controller space """
        # Partition state space in each dimension. 
        # NOTE: If theta dimension is partitioned non-uniformly, need also customize find_theta_partitions() in PosteriorGraph. 
        x_dim = len(self.x_span[0])
        x_intervals = []
        for i in range(x_dim):
            if self.wksp==2 and i==0: # Non-uniform partition x dimension.
                dim_intervals = [0.0, 0.5, 0.75, 1.25, 1.5, 2.0, 2.25, 2.75, 3.0, 3.5, 3.75, 4.25, 4.5, 5.0]
            elif self.wksp==2 and i==1: # Non-uniform partition y dimension.
                dim_intervals = [0.0, 0.5, 1.0, 1.25, 1.75, 2.0, 2.5, 3.0, 3.25, 3.75, 4.0, 4.5, 5.0]
            else:
                dim_intervals = np.linspace(self.x_span[0][i], self.x_span[1][i], self.x_num[i]+1).tolist()
            x_intervals.append(dim_intervals)
            print('x%d: ' % i, end='')
            print(dim_intervals)
        # A list consists of lower bound of each abstract state. 
        states_low = [[x_intervals[0][j]] for j in range(self.x_num[0])]
        for i in range(1, x_dim):
            states_low_new = [q_low + [x_intervals[i][j]] for q_low in states_low for j in range(self.x_num[i])]
            states_low = deepcopy(states_low_new)      
        # A list consists of upper bound of each abstract state.
        states_up = [[x_intervals[0][j]] for j in range(1, self.x_num[0]+1)]
        for i in range(1, x_dim):
            states_up_new = [q_up + [x_intervals[i][j]] for q_up in states_up for j in range(1, self.x_num[i]+1)]
            states_up = deepcopy(states_up_new)     
        # Remove abstract states corresponding to goal and obstacles.
        remove_low, remove_up = [], []
        for q_low, q_up in zip(states_low, states_up):
            if self.is_goal_obstacles(q_low, q_up):
                remove_low.append(q_low)
                remove_up.append(q_up)
        for q_low, q_up in zip(remove_low, remove_up):
            states_low.remove(q_low)
            states_up.remove(q_up)
        # Reconcile heading angle if needed.
        if self.has_theta:
            states_low, states_up = self.reconcile_theta(states_low, states_up)
        print('Number of abstract states: ', len(states_low))
        #count = 0
        #for q_low, q_up in zip(states_low, states_up):
        #    print(count)
        #    print(q_low)
        #    print(q_up, end='\n\n')
        #    count += 1

        # Partition controller space in each dimension.
        K_dim = len(self.K_span[0])
        K_intervals = []
        for i in range(K_dim):
            dim_intervals = np.linspace(self.K_span[0][i], self.K_span[1][i], self.K_num[i]+1)
            K_intervals.append(dim_intervals)
            print('K%d: ' % i, end='')
            print(dim_intervals)
        # A list consists of lower bound of each controller partition. 
        partitions_low = [[K_intervals[0][j]] for j in range(self.K_num[0])]
        for i in range(1, K_dim):
            partitions_low_new = [p_low + [K_intervals[i][j]] for p_low in partitions_low for j in range(self.K_num[i])]
            partitions_low = deepcopy(partitions_low_new) 
        # A list consists of upper bound of each controller partition. 
        partitions_up = [[K_intervals[0][j]] for j in range(1, self.K_num[0]+1)]
        for i in range(1, K_dim):
            partitions_up_new = [p_up + [K_intervals[i][j]] for p_up in partitions_up for j in range(1, self.K_num[i]+1)]
            partitions_up = deepcopy(partitions_up_new)      
        print('Number of controller partitions: ', len(partitions_low))
        #count = 0
        #for p_low, p_up in zip(partitions_low, partitions_up):
        #    print(count)
        #    print(p_low)
        #    print(p_up, end='\n\n')
        #    count += 1
        return states_low, states_up, partitions_low, partitions_up
        
    def reconcile_theta(self, states_low, states_up):
        """ 
        Bring theta "roughly" into range [-pi, pi].
        The reason to do this is TIRA seems to return tighter over-approximation when absolute value of theta is smaller. 
        E.g. [3*pi/4, 5*pi/4] --> [3*pi/4, 5*pi/4] (No change though the upper bound exceeds pi)
             [5*pi/4, 7*pi/4] --> [-3*pi/4, -pi/4]
             [7*pi/4, 9*pi/4] --> [-pi/4, pi/4]
        """
        for q_low, q_up in zip(states_low, states_up):
            if q_low[2] >= np.pi-const.err: 
                q_low[2] -= 2*np.pi
                q_up[2]  -= 2*np.pi
        return states_low, states_up     

    def is_goal_obstacles(self, q_low, q_up):
        """ Check if the given abstract state is a subset of goal or obstacles """        
        is_goal_obst = False
        goal_obstacles = deepcopy(self.obstacles)
        goal_obstacles.append(self.goal)
        for o in goal_obstacles:
            if all([o_dim_low-const.err <= q_dim_low and q_dim_up <= o_dim_up+const.err 
                    for o_dim_low, o_dim_up, q_dim_low, q_dim_up in zip(o[0], o[1], q_low, q_up)]):
                is_goal_obst = True
                break
        return is_goal_obst          

    def save_cells(self, save_file, states_low, states_up, partitions_low, partitions_up):
        """ Save workspace configuration, abstract states, and controller partitions """
        cells = {'states_low':    states_low, 
                'states_up':      states_up, 
                'partitions_low': partitions_low, 
                'partitions_up':  partitions_up,
                'wksp':           self.wksp,
                'has_theta':      self.has_theta,
                'x_span':         self.x_span,
                'x_num':          self.x_num,
                'goal':           self.goal,
                'obstacles':      self.obstacles,
                'system_dict':    self.system_dict,
                'disturb_bound':  self.disturb_bound
                }
        savemat(save_file, cells)
        print('Cells are saved, please call TIRA to compute posteriors')


if __name__ == "__main__":
    # Partition state and controller space, then save the resulting cells. 
    
    save_file = './data/cells.mat'

    wks = Workspace()
    wks.set_config()
    states_low, states_up, partitions_low, partitions_up = wks.partition_space()
    wks.save_cells(save_file, states_low, states_up, partitions_low, partitions_up)


              
