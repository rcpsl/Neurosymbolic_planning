import numpy as np
import pickle
from math import sqrt

class Environment(object):
    def __init__(self, system_dict, has_theta, theta_partitions, disturb_bound):
        self.sys_choice = system_dict['sys_choice']
        if self.sys_choice == 2:
            self.dt = system_dict['dt'][0][0][0][0]
            self.v  = system_dict['v'][0][0][0][0]
            self.x_dim  = 3
        else:
            assert False, 'System choice is not defined'
        self.has_theta = has_theta
        self.theta_partitions = theta_partitions          
        self.disturb_bound = disturb_bound

    def load_gp(self):
        """ Load GP for residual dynamics """
        if self.sys_choice == 2:
            fin_name = './data/gpr'
            with open(fin_name, 'rb') as fin:
                self.gpr = pickle.load(fin)
            fin.close() 

    def setup(self, frm_node, to_node, partition_low, partition_up):     
        """ Setup transition that the NNs trained for """
        self.frm_node = frm_node
        self.to_node = to_node
        self.partition_low = partition_low
        self.partition_up  = partition_up
        assert self.x_dim == len(to_node.q_low), 'System dimension does not match'

    def reset(self):
        """ Sample initial states in frm_node """
        state = [np.random.uniform(low, up) for low, up in zip(self.frm_node.q_low, self.frm_node.q_up)]
        return state

    def step(self, state, action, verbose):
        """ Update state, compute reward, and decide if episode is done"""
        new_state = self.update_state_gp(state, action)
        reward, done = self.compute_reward(new_state, action, verbose)
        return new_state, reward, done

    def update_state_perfect(self, state, action):
        """ Update state using perfect dynamics """
        if self.sys_choice == 2:
            x0, y0, theta0 = state[0], state[1], state[2]
            x1 = x0 + self.dt * self.v * np.cos(theta0) #+ 0.01 #+ 0.006 * y0 #+ 0.02 * self.dt * x0 * y0 #0.03
            y1 = y0 + self.dt * self.v * np.sin(theta0) #+ 0.1 #* y0 #+ 0.04 * self.dt * x0 * y0 #0.03
            theta1 = theta0 + self.dt * action[0]
            new_state = [x1, y1, theta1]
            new_state = self.wrap_theta2abst_state(new_state)
        return new_state

    def update_state_nominal(self, state, action):
        """ Update state using nominal dynamics """
        if self.sys_choice == 2:
            x0, y0, theta0 = state[0], state[1], state[2]
            x1 = x0 + self.dt * self.v * np.cos(theta0)
            y1 = y0 + self.dt * self.v * np.sin(theta0)
            theta1 = theta0 + self.dt * action[0]
            new_state = [x1, y1, theta1]
            new_state = self.wrap_theta2abst_state(new_state)
        return new_state

    def update_state_gp(self, state, action):
        """ Update state using estimated dynamics, i.e. nominal plus GP """
        if self.sys_choice == 2:
            mu, sigma = self.gpr.predict(np.array([state]), return_std=True)
            residual = min(max(self.disturb_bound[0][0], mu[0]), self.disturb_bound[1][0])
            #print('state:', state)
            #print('mu:', mu)
            #print('residual:', residual)
            x0, y0, theta0 = state[0], state[1], state[2]
            x1 = x0 + self.dt * self.v * np.cos(theta0) + residual
            y1 = y0 + self.dt * self.v * np.sin(theta0) + residual
            theta1 = theta0 + self.dt * action[0]
            new_state = [x1, y1, theta1]
            new_state = self.wrap_theta2abst_state(new_state)
        return new_state

    def wrap_theta2abst_state(self, state):
        """ 
        Wrap theta such that it is in some theta ranges that appears in abstract states.
        Notice that not every theta range in abstract states is within the theta span defined in x_span 
        (see the example in PosteriorGraph.normalize_theta).
        """
        state[2] = (state[2] + 100*np.pi) % (2*np.pi)
        for p in self.theta_partitions:
            if p[0] <= state[2] <= p[1]:
                return state
        state[2] += 2*np.pi
        for p in self.theta_partitions:
            if p[0] <= state[2] <= p[1]:
                return state
        print('Theta in current state (after wrap into [0, 2*pi)):', (state[2] + 100*np.pi) % (2*np.pi))
        assert False, 'Theta in current state is not in any abstract state. Try to define theta span that covers 2*pi.' 

    def compute_reward(self, new_state, action, verbose):
        """ Compute reward and decide if episode is done """
        # TODO: state_dist_thld and action_dist_thld should depend on grid size in discretization.
        # TODO: Instead of using threshold distances, check whether state and action are contained in the deflated to_node and partition, respectively. 
        #state_dist_thld  = 0.2
        #action_dist_thld = 1.

        # Episode is done if reached to_node.
        done = self.inside_to_node(new_state)

        # Reward based on distance between new_state and assigned next_state. 
        state_dist = self.compute_state_distance(new_state)
        #if state_dist > state_dist_thld:
        if not done: 
            reward_state = -state_dist
        else:      
            reward_state = 10.

        # Reward based on distance between action and assigned controller partition. 
        action_dist = self.compute_action_distance(action, new_state)
        #if action_dist > action_dist_thld:
        reward_action = -action_dist
        #else:
        #reward_action = 10.

        # Should give higher priority on reward_state or reward_action? 
        # TODO: May first scale reward_state and reward_action to the same order of magnitude. 
        reward = 10 * reward_state + 1 * reward_action
        if verbose:
            print('reward_state:', reward_state)
            print('reward_action:', reward_action)
        return reward, done

    def inside_to_node(self, state):
        """ Whether state is located inside to_node """
        # TODO: Deflate to_node.
        # Note theta range of the goal should be superset of theta range of any abstract state.  
        is_inside = True
        for q_dim_low, q_dim_up, state_dim in zip(self.to_node.q_low, self.to_node.q_up, state):
            if not (q_dim_low <= state_dim <= q_dim_up):
                is_inside = False
                break
        return is_inside

    def compute_state_distance(self, state):
        """
        Compute distance between state and the center of to_node.
        The distance is weighted sum of distance in xy dimesion and that in high dimension. 
        """
        w_xy = 1.
        w_high = 1. 
        to_low = self.to_node.q_low.copy()
        to_up  = self.to_node.q_up.copy()
        # Change theta range of the goal to [0, pi] so its center used in computing distance is pi/2.
        if self.has_theta and self.to_node.identity==1:
            to_low[2] = 0.
            to_up[2]  = np.pi 
        to_center = [(l+u)/2 for l, u in zip(to_low, to_up)]
        dist_in_dims = [(s-c)**2 for s, c in zip(state, to_center)]
        # Distance between two theta should be less than pi (it is free to choose clockwise or counterclockwise direction).
        if self.has_theta:
            state_theta = (state[2] + 100*np.pi) % (2*np.pi) 
            to_theta    = (to_center[2] + 100*np.pi) % (2*np.pi) 
            theta_diff = abs(state_theta - to_theta)
            theta_diff = theta_diff if theta_diff < np.pi else 2*np.pi-theta_diff
            dist_in_dims[2] = theta_diff**2
            assert 0 <= theta_diff <= np.pi, 'Difference in theta should be in [0, pi]'
        # Weighted sum of distances in xy and higher order dimensions. 
        dist = sqrt(w_xy * sum(dist_in_dims[:2]) + w_high * sum(dist_in_dims[2:]))
        return dist

    def compute_action_distance(self, action, state):
        """ compute distance between action and Kc(state), where Kc is center of assigned partition """
        Kc = [(l+u)/2 for l, u in zip(self.partition_low, self.partition_up)]  
        if self.sys_choice == 2:
            uc = Kc[0]*state[0] + Kc[1]*state[1] + Kc[2]*state[2] + Kc[3]
            dist = abs(uc-action[0])
        return dist





