import numpy as np
import scipy.stats as st
from copy import deepcopy
from functools import reduce
from operator import mul
import pickle

from NNController import NNController

class NNSynth(object):
    def __init__(self, states, inputs, specs, noise, dynamics):
        self.states = states
        self.inputs = inputs
        self.specs = specs
        self.noise = noise
        self.dynamics = dynamics
        self.half_cutoff_num = None
        self.nx, self.nu = None, None
        self.abstr_states, self.abstr_inputs = None, None
        self.x_grid_basis, self.u_grid_basis = None, None
        self.v_info = []
        # Store transition matrix T of dimension (nx*nu, nx) as a dictionary with keys (x_id, u_id).
        # Only entries of T suggested by the NN will be computed. 
        # Entries of T do not change across iterations once they are computed.
        self.T = {}
        self.visited_u_spaces = None
        self.prob_dict = {}
        self.goal_ids      = set()
        self.goal_obst_ids = set()
        self.T_goal = {}

    def main(self, params, load_nn, save_nn, save_symbolic, ex_id):
        """ Main synthesis loop """
        self.x_dim = self.states['dim']
        self.u_dim = self.inputs['dim']
        self.H = self.specs['time_steps']
        # Limit the number of post states depend on the cutting-off probability.
        if self.noise['pdf_truncation'] == 'cutting_probability':
            p = 1 - self.noise['cutting_probability']
            z = st.norm.ppf(p)
            self.half_cutoff_num = [int(sigma*z/eta)+1 for sigma, eta in zip(self.noise['std_deviation'], self.states['eta'])]
        elif self.noise['pdf_truncation'] == 'fixed_truncation':
            self.half_cutoff_num = [int(r/eta)+1 for r, eta in zip(self.noise['cutting_radius'], self.states['eta'])]
        else:
            assert False, 'Please specify how to truncate probability'     
        #print('half_cutoff_num:', self.half_cutoff_num)
        # Partition state and input spaces (not compute transitions). 
        self.build_abstraction()
        self.nx = len(self.abstr_states)
        self.nu = len(self.abstr_inputs)
        print('\nExperiment #%d' % int(ex_id))
        print('Spec type: ', self.specs['type'])
        print('Number of abstract states |X|:', self.nx)
        print('Number of abstract inputs |U|:', self.nu)
        print('Problem complexity |X x U|:', self.nx * self.nu)
        print('Number of local actions to be added at each state: %d' % reduce(mul, self.inputs['local_u_num'], 1))
        if self.specs['type'] != 'safe':
            # Identify x_id corresponding to goal and obstacles
            self.goal_ids = self.identify_contained_states(self.specs['goal'])
            self.goal_obst_ids = self.goal_ids.copy()
            if self.specs['type'] == 'reachavoid':
                #for obst in self.specs['obst']:
                obst_ids = self.identify_contained_states(self.specs['obst'])
                self.goal_obst_ids.update(obst_ids)
            #print('Size of goal_obst_ids:', len(self.goal_obst_ids))
        # self.visited_u_spaces tracks which local actions u have been visitied, i.e. T at (x,u) has been computed.
        # Since computed entries of T do not change across iterations, self.visited_u_spaces does not change. 
        # Alternatively, can directly check existence of (x,u) in T, which might be slower.
        self.visited_u_spaces = [set() for i in range(self.nx)] 
        if abs(ex_id-4.1) < 1e-5:
            data_dir = './data-robot2d/'
        elif abs(ex_id-4.2) < 1e-5:     
            data_dir = './data-roomtemp5d/'
        else:     
            data_dir = './data/'

        # Iteration 0: train NN with expert data, then value iteration. 
        # TODO: In different iterations, re-initialize the NN model instead of building a new one.        
        print('\n=================== Iteration 1/%d ==================\n' % params['max_iter'])
        #if params['training_option'] == 'imitation':
        nn_controller = NNController(params, self.x_dim, self.u_dim)
        model = nn_controller.build_model()
        if load_nn:
            model_dir = data_dir + 'model.h5'
            model.load_weights(model_dir)
        #init_weights = model.get_weights()
        X, y  = nn_controller.load_samples(data_dir) 
        model = nn_controller.train(model, X, y, data_type='expert', if_compile=True)
        #elif params['training_option'] == 'reinforcement':
        #    ppo_agent = PPOAgent(self.H, self.x_dim, self.u_dim)
        #    if load_nn:
        #        model = ppo_agent.build_actor()
        #        model_dir = './data/model'
        #        model.load_weights(model_dir + ".h5")
        #    else:
        #        model = ppo_agent.train(self.dynamics, self.states, self.inputs, self.specs)
        V, Sc = self.value_iteration(model)
        self.get_v_info(V)
        print('Satisfaction probability at the end of this iteration V_avg: ', '{:.2%}'.format(self.v_info[-1][2]))

        for count in range(1, params['max_iter']):
            print('\n=================== Iteration %d/%d ==================\n' % (count+1, params['max_iter']))
            #if params['training_option'] == 'imitation':
            # Re-initialize NN
            #model.set_weights(init_weights)
            model = nn_controller.build_model()
            # Lift by imitation learning of the current symbolic controller. 
            X, y = self.collect_sc_data(Sc)
            model = nn_controller.train(model, X, y, data_type='symbolic', if_compile=True)
            # Further training by expert. 
            X, y  = nn_controller.load_samples(data_dir) 
            model = nn_controller.train(model, X, y, data_type='expert', if_compile=False)
            #elif params['training_option'] == 'reinforcement':
            #    # Make sure NN model and PPO actor have the same size. 
            #    # Still need lift here.
            #    pass
            V, Sc = self.value_iteration(model)
            self.get_v_info(V)
            print('Satisfaction probability at the end of this iteration V_avg: ', '{:.2%}'.format(self.v_info[-1][2]))
    
        if ex_id in [1, 2, 3]:
            if ex_id == 1:
                print('\n--------------------- Gather Information in Table 1 (2-d Robot Column) -----------------------\n')
                print('Benchmark: 2-d Robot') 
            elif ex_id == 2: 
                print('\n--------------------- Gather Information in Table 1 (5-d Room Temp. Column) -----------------------\n')
                print('Benchmark: 5-d Room Temp.') 
            else:
                print('\n--------------------- Gather Information in Table 1 (5-d Traffic Column) -----------------------\n')
                print('Benchmark: 5-d Traffic')    
            print('Spec type: ', self.specs['type'])
            print('Spec horizon T_d: ', self.specs['time_steps'])
            print('Problem complexity |X x U|:', self.nx * self.nu)
            print('Satisfaction probability V_avg: ', '{:.2%}'.format(self.v_info[-1][2]))
        if save_nn:
            model_dir = data_dir + 'model.h5'
            model.save_weights(model_dir)      
        if save_symbolic:
            fout_name = data_dir + 'symbolic'
            symbolic = {
                'states'        : self.states,
                'inputs'        : self.inputs,
                'specs'         : self.specs, 
                'noise'         : self.noise,
                'abstr_states'  : self.abstr_states,
                'abstr_inputs'  : self.abstr_inputs,
                'x_grid_basis'  : self.x_grid_basis,
                'u_grid_basis'  : self.u_grid_basis,
                'goal_ids'      : self.goal_ids,
                'goal_obst_ids' : self.goal_obst_ids,
                'params'        : params,
                'v_info'        : self.v_info,
                'Sc'            : Sc,
                'V'             : V
            }
            with open(fout_name, 'wb') as fout:
                pickle.dump(symbolic, fout)

    def value_iteration(self, model):         
        """ Synthesize symbolic controller by value iteration """   
        V = np.ones(self.nx) if self.specs['type'] == 'safe' else np.zeros(self.nx)
        Sc = {}
        for k in range(self.H, -1, -1): 
            print('Value iteration step: ', k)
            local_u_spaces = self.add_transitions(model, k)
            #print('Finished adding local actions...')
            #print('local_u_spaces:', local_u_spaces)
            if self.specs['type'] == 'safe':
                self.safe_update_transition_matrix(local_u_spaces)
                #Q = self.update_q(V, local_u_spaces)
                #V, Sc_k = self.update_optimal_v(Q, local_u_spaces)
                V, Sc_k = self.safe_update_q_and_v(V, local_u_spaces)
                Sc[k] = Sc_k
                #print('T:', self.T)
                #print('V: ', V)
                #print('Sc_k:', Sc_k)
            else:
                self.reach_update_transition_matrix(local_u_spaces)
                V, Sc_k = self.reach_update_q_and_v(V, local_u_spaces)
                Sc[k] = Sc_k
        return V, Sc

    def reach_update_transition_matrix(self, local_u_spaces):
        """ UPdate transition matrix T (for reach or reach-avoid specs) """
        for x_id in range(self.nx):
            #print(x_id)
            if x_id in self.goal_obst_ids:
                # No transition leaves goal and obstacles.
                continue
            for u_id in local_u_spaces[x_id] - self.visited_u_spaces[x_id]:
                self.visited_u_spaces[x_id].add(u_id)
                x = self.abstr_states[x_id]
                u = self.abstr_inputs[u_id]
                post_mean = self.dynamics(x, u) 
                post_mean_id = self.x2id(post_mean)
                if post_mean_id == -1:
                    # If post_mean is out of the state space, ignore this input.
                    continue
                self.T[(x_id, u_id)] = post_mean_id
                if post_mean_id not in self.prob_dict:
                    post_states, probabilities = self.find_local_states_prob(post_mean_id)
                    # Set T entries to zero if the corresponding post state in the goal and obstacles.
                    # A little bit redundancy: computed probability then set to zero.
                    # This set is not necessary as long as keep V entries be zero for the goal and obstacles.
                    #for i in range(len(probabilities)):
                    #    if post_states[i] in self.goal_obst_ids:
                    #        probabilities[i] = 0.
                    p_goal = self.compute_transition_prob(post_mean_id, self.specs['goal'][0], self.specs['goal'][1])
                    self.prob_dict[post_mean_id] = [post_states, np.array(probabilities), p_goal]
                #self.T_goal[(x_id, u_id)] = self.compute_transition_prob(post_mean, self.specs['goal'][0], self.specs['goal'][1])
                #post_states = self.find_local_states(post_mean)
                #self.T[(x_id, u_id)] = np.zeros(self.nx)
                #for post_x_id in post_states:
                #    if post_x_id in self.goal_obst_ids:
                #        # The corresponding T entry is 0
                #        continue      
                #    else:   
                #       post_x = self.abstr_states[post_x_id]
                #        lbs = [post_x[i] - self.states['eta'][i]/2. for i in range(self.x_dim)]
                #        ubs = [post_x[i] + self.states['eta'][i]/2. for i in range(self.x_dim)]
                #        self.T[(x_id, u_id)][post_x_id] = self.compute_transition_prob(post_mean, lbs, ubs)   
            
    def reach_update_q_and_v(self, V_old, local_u_spaces):
        """ 
        Compute Q and the optimal V function (for reach or reach-avoid specs). 
        Return the optimal value function and the optimal controller at current step.
        """
        V, Sc_k = [], []
        for x_id in range(self.nx):
            if x_id in self.goal_obst_ids:
                V.append(0.)
                Sc_k.append(0) # Add a random action to occupy the entry.
                continue 
            V_max = 0.
            u_max = None
            for u_id in local_u_spaces[x_id]: 
                try:
                    post_mean_id = self.T[(x_id, u_id)]
                    post_states, probabilities, p_goal = self.prob_dict[post_mean_id]
                    Q_xu = p_goal + np.dot(probabilities, V_old[post_states]) 
                    if Q_xu > V_max:
                        V_max = Q_xu
                        u_max = u_id
                except KeyError:
                    # T[(x_id, u_id)] does not exist since post_mean is out of the state space. 
                    continue
            if u_max is None:
                #print('+++++ WARNING: ALL actions lead post mean out of the sate space +++++')
                #print('Better train NN or add more local actions')
                V_max = 0.
                u_max = u_id
            #assert u_max is not None, 'u_max should not be None'
            V.append(V_max)
            Sc_k.append(u_max)
        return np.array(V), Sc_k     

    def safe_update_transition_matrix(self, local_u_spaces):
        """ UPdate transition matrix T (for safety specs) """
        for x_id in range(self.nx):
            for u_id in local_u_spaces[x_id] - self.visited_u_spaces[x_id]:
                self.visited_u_spaces[x_id].add(u_id)
                x = self.abstr_states[x_id]
                u = self.abstr_inputs[u_id]
                post_mean = self.dynamics(x, u) 
                post_mean_id = self.x2id(post_mean)
                if post_mean_id == -1:
                    # If post_mean is out of the state space, ignore this input.
                    continue
                #post_mean = self.abstr_states[post_mean_id]
                #post_states = self.find_local_states(post_mean)
                #print('num post states:', len(post_states))
                #probabilities = []
                #for post_x_id in post_states:
                #    post_x = self.abstr_states[post_x_id]
                #    lbs = [post_x[i] - self.states['eta'][i]/2. for i in range(self.x_dim)]
                #    ubs = [post_x[i] + self.states['eta'][i]/2. for i in range(self.x_dim)]
                #    p = self.compute_transition_prob(post_mean, lbs, ubs)
                #    probabilities.append(p)
                self.T[(x_id, u_id)] = post_mean_id
                if post_mean_id not in self.prob_dict:
                    post_states, probabilities = self.find_local_states_prob(post_mean_id)
                    self.prob_dict[post_mean_id] = [post_states, np.array(probabilities)]

    def safe_update_q_and_v(self, V_old, local_u_spaces):
        """ 
        Compute Q and the optimal V function (for safety specs). 
        Return the optimal value function and the optimal controller at current step.
        """
        V, Sc_k = [], []
        for x_id in range(self.nx):
            V_max = 0.
            u_max = None
            for u_id in local_u_spaces[x_id]: 
                try:
                    post_mean_id = self.T[(x_id, u_id)]
                    post_states, probabilities = self.prob_dict[post_mean_id]
                    Q_xu = np.dot(probabilities, V_old[post_states]) 
                    if Q_xu > V_max:
                        V_max = Q_xu
                        u_max = u_id
                except KeyError:
                #    # T[(x_id, u_id)] does not exist since post_mean is out of the state space. 
                    continue
            if u_max is None:
                #print('+++++ WARNING: ALL actions lead post mean out of the sate space +++++')
                #print('Better train NN or add more local actions\n\n')
                V_max = 0.
                u_max = u_id
            #assert u_max is not None, 'u_max should not be None'
            V.append(V_max)
            Sc_k.append(u_max)
        return np.array(V), Sc_k     

    #def update_q(self, V, local_u_spaces):
    #    # NOTE: Only compute entries corresponding to (x, u) where u is local at x at CURRENT step. 
    #    # Not compute all (x, u) corresponding to non-zero T at (x, u).
    #    Q = np.zeros(self.nx * self.nu)
    #    for x_id in range(self.nx):
    #        for u_id in local_u_spaces[x_id]: 
    #            try:
    #                Q[x_id * self.nu + u_id] = np.dot(self.T[(x_id, u_id)], V) 
    #            except IndexError:
    #                # T[(x_id, u_id)] does not exist since the posterior mean is out of the state space. 
    #                continue
    #    return Q
   
    #def update_optimal_v(self, Q, local_u_spaces):
    #    # Optimal value V(x) is the u maximizes Q(x, u), where u is over local input at x.
    #    # Also, computes the optimal controller at current step Sc_k.
    #    Q_matrix = Q.reshape((self.nx, self.nu))
    #    V = np.amax(Q_matrix, axis=1)
    #    Sc_k = np.argmax(Q_matrix, axis=1)
    #    return V, Sc_k
    
    def find_local_states_prob(self, x_id):
        """ 
        Find states close to the state x_id (up to probability cut-off), and integrate 
        Gaussian distribution with mean at the state x_id over these close states.
        """
        # NOTE: state ids in the returned list is not in increasing order.
        #x = self.abstr_states[self.x2id(x)]
        #mean = x.copy()
        #x_id = self.x2id(x)
        #if x_id != -1:
        # If x is within the state space bound, shift x to a center, which is NECESSARY, 
        # otherwise integral is not accurate due to the integral limit [x-eta, x+eta]
        mean = self.abstr_states[x_id] #shift post_mean to a center
        intervals, probs = [], []
        for i in range(self.x_dim):
            sigma = self.noise['std_deviation'][i]
            eta = self.states['eta'][i]
            #if x_id != -1:
            dim_intervals = [mean[i]]
            dim_probs = [self.compute_prob(mean[i], sigma, mean[i]-eta/2., mean[i]+eta/2.)]
            #else:
            #    dim_intervals, dim_probs = [], []
            #left  = [x[i] - m * self.states['eta'][i] for m in range(1, self.half_cutoff_num+1)]
            #left  = [a for a in left if a > self.states['lb'][i]]
            #right = [x[i] + m * self.states['eta'][i] for m in range(1, self.half_cutoff_num+1)]
            #right = [b for b in right if b < self.states['ub'][i]]
            #dim_intervals.extend(left)
            #dim_intervals.extend(right)
            for m in range(1, self.half_cutoff_num[i]+1):
                a = mean[i] - m * self.states['eta'][i]
                #if self.states['lb'][i] < a < self.states['ub'][i]: 
                if a > self.states['lb'][i]:
                    dim_intervals.append(a)
                    #ct = self.abstr_states[self.x2id(a)]
                    p = self.compute_prob(mean[i], sigma, a-eta/2., a+eta/2.)
                    dim_probs.append(p)
                else:
                    break
            for m in range(1, self.half_cutoff_num[i]+1):
                b = mean[i] + m * self.states['eta'][i]
                #if self.states['lb'][i] < b < self.states['ub'][i]:
                if b < self.states['ub'][i]:
                    dim_intervals.append(b)
                    #ct = self.abstr_states[self.x2id(b)]
                    p = self.compute_prob(mean[i], sigma, b-eta/2., b+eta/2.)
                    dim_probs.append(p)
                else:
                    break 
            intervals.append(dim_intervals)
            probs.append(dim_probs)
        # Combine state components in all dimensions.
        x_space = [[a] for a in intervals[0]]
        for i in range(1, self.x_dim):
            x_space_new = [partial_x + [a] for partial_x in x_space for a in intervals[i]]
            x_space = deepcopy(x_space_new)
        x_space_id = [self.x2id(x) for x in x_space]
        assert -1 not in x_space_id, 'State margin is not a grid'
        # Combine probability components in all dimensions.
        probabilities = [p for p in probs[0]]
        for i in range(1, self.x_dim):
            probabilities_new = [partial_p * p for partial_p in probabilities for p in probs[i]]
            probabilities = probabilities_new.copy()
        #print(len(x_space_id))
        #print(len(probabilities))
        return x_space_id, probabilities

    #def cdf(self, x, mu, sigma):
    #    return (1.0 + erf((x-mu) / (sqrt(2.0) * sigma))) / 2.0

    def compute_prob(self, mu, sigma, lb, ub):
        p = st.norm.cdf((ub-mu)/sigma) - st.norm.cdf((lb-mu)/sigma)
        assert p > -1e-6, 'make sure ub > lb' 
        return p
    
    def compute_transition_prob(self, x_id, lbs, ubs):
        mean = self.abstr_states[x_id] #shift post_mean to a center
        prob = 1.
        for i in range(self.x_dim):
            sigma = self.noise['std_deviation'][i]
            #def gauss(x):
            #    return st.norm.pdf(x, mu, sigma)
            #res, _ = integrate.quad(gauss, l, u)
            #res = self.cdf(u, mu, sigma) - self.cdf(l, mu, sigma)
            #res = st.norm.cdf((u-mu)/sigma) - st.norm.cdf((l-mu)/sigma)
            res = self.compute_prob(mean[i], sigma, lbs[i], ubs[i])
            prob *= res
        return prob

    def identify_contained_states(self, box):
        """ Find all abstract states in the give box """
        lbs, ubs = box
        x = [l + 1e-5 for l in lbs]
        x = self.abstr_states[self.x2id(x)]
        intervals = []
        for i in range(self.x_dim):
            dim_intervals = [x[i]]
            for m in range(1, self.nx):
                a = x[i] + m * self.states['eta'][i]
                if a < ubs[i]:
                    dim_intervals.append(a)
                else:
                    break    
            intervals.append(dim_intervals)
        # Combine state components in all dimensions.
        x_space = [[a] for a in intervals[0]]
        for i in range(1, self.x_dim):
            x_space_new = [partial_x + [a] for partial_x in x_space for a in intervals[i]]
            x_space = deepcopy(x_space_new)
        x_space_id = set([self.x2id(x) for x in x_space])
        assert -1 not in x_space_id, 'State margin is not a grid'
        return x_space_id

    def x2id(self, x):
        """ 
        Given x coordinates (x does not need to be a center), return x_id of the closest center.
        Return -1 if x_id is wrong, which happens because x is out of the state space or state space margin is not a grid.
        """
        x_indices = [int((x[i] - self.states['lb'][i])/self.states['eta'][i]) for i in range(self.x_dim)]
        x_id = sum(a * b for a, b in zip(x_indices[:-1], self.x_grid_basis)) + x_indices[-1]
        try:
            center = self.abstr_states[x_id]
            if not all([abs(x[i] - center[i]) <= self.states['eta'][i]/2. + 1e-5 for i in range(self.x_dim)]):
                x_id = -1
        except IndexError:
            x_id = -1  
        return x_id

    def add_transitions(self, model, k):
        """ Add actions at x that are close to NN(x) """
        # A little bit redundancy: no need to find local inputs at goal and obstacles. 
        xt = [x + [k] for x in self.abstr_states]
        if self.x_dim == 7 or self.x_dim == 8:
            num_dp = len(xt)
            dummy_advantage, dummy_action = np.zeros((num_dp, 1)), np.zeros((num_dp, self.u_dim))
            y = model.predict([np.array(xt).reshape(num_dp, self.x_dim+1), dummy_advantage, dummy_action])
        else:
            y = model.predict(np.array(xt))
        # Project NN outputs to the input space.
        for u in y:
            for i in range(self.u_dim):
                if u[i] < self.inputs['lb'][i]:
                    u[i] = self.inputs['lb'][i] + 1e-4
                elif u[i] > self.inputs['ub'][i]:
                    u[i] = self.inputs['ub'][i] - 1e-4   
        local_u_spaces = [self.find_local_inputs(u) for u in y]
        return local_u_spaces

    def find_local_inputs(self, u):
        """ Return a set of u_id correspond to actions close to u """
        # NOTE: when call this function, u should be inside the input space.
        #u = np.array([-0.5, -0.7])
        #u = np.array([-0.7, 0.9])
        u = self.abstr_inputs[self.u2id(u)]
        intervals = []
        for i in range(self.u_dim):
            dim_intervals = [u[i]]
            half_u_num = int(self.inputs['local_u_num'][i]/2)
            #left  = [u[i] - m * self.inputs['eta'][i] for m in range(1, half_u_num+1)]
            #left  = [a for a in left if a > self.inputs['lb'][i]]
            #right = [u[i] + m * self.inputs['eta'][i] for m in range(1, half_u_num+1)]
            #right = [b for b in right if b < self.inputs['ub'][i]]
            #dim_intervals.extend(left)
            #dim_intervals.extend(right)
            for m in range(1, half_u_num+1):
                a = u[i] - m * self.inputs['eta'][i]
                if a > self.inputs['lb'][i]:
                    dim_intervals.append(a)
                else:
                    break
            for m in range(1, half_u_num+1):
                b = u[i] + m * self.inputs['eta'][i]
                if  b < self.inputs['ub'][i]:
                    dim_intervals.append(b)
                else:
                    break 
            intervals.append(dim_intervals)
            #print(dim_intervals)
        # Combine u components in all dimensions.
        u_space = [[a] for a in intervals[0]]
        for i in range(1, self.u_dim):
            u_space_new = [partial_u + [a] for partial_u in u_space for a in intervals[i]]
            u_space = deepcopy(u_space_new)
        u_space_id = set([self.u2id(u) for u in u_space])
        #print('u_space:', u_space)
        #print('u_space_id:', u_space_id)
        return u_space_id

    def u2id(self, u):
        """ Given u coordinates (u does not need to be a center), return u_id of the closest center """
        # NOTE: when call this function, u should be inside the input space.
        # TODO: add sanity check that input space margin should be a grid. 
        #u = np.array([-0.5, 0.9, 0., 0.99, 2.0])
        u_indices = [int((u[i] - self.inputs['lb'][i])/self.inputs['eta'][i]) for i in range(self.u_dim)]
        u_id = sum(a * b for a, b in zip(u_indices[:-1], self.u_grid_basis)) + u_indices[-1]
        assert all([abs(u[i] - self.abstr_inputs[u_id][i]) <= self.inputs['eta'][i]/2. + 1e-5 for i in range(self.u_dim)]), \
        'u_id is wrong. Possible reason: u is not in any gird, either out of input bound or at input space margin that is not a grid'
        #print('u:', u)
        #print('u_indices:', u_indices)
        #print('u_id', u_id)
        #print('abstact_inputs[u_id]', self.abstr_inputs[u_id])
        return u_id

    def build_abstraction(self):
        """ Partition state and action spaces """
        # Partition state space in each dimension. 
        x_grid_num = [int((self.states['ub'][i] - self.states['lb'][i] + 1e-5)/self.states['eta'][i]) for i in range(self.x_dim)]
        x_intervals = []
        for i in range(self.x_dim):
            dim_intervals = np.linspace(self.states['lb'][i], self.states['ub'][i], x_grid_num[i]+1).tolist()
            dim_intervals = [float('{:.4f}'.format(x + self.states['eta'][i]/2.)) for x in dim_intervals]
            x_intervals.append(dim_intervals)
        # A list consists of coordinates of abstract states (centers of partitions). 
        self.abstr_states = [[x_intervals[0][j]] for j in range(x_grid_num[0])]
        for i in range(1, self.x_dim):
            abstr_states_new = [x + [x_intervals[i][j]] for x in self.abstr_states for j in range(x_grid_num[i])]
            self.abstr_states = deepcopy(abstr_states_new)
        self.x_grid_basis = [reduce(mul, x_grid_num[i:], 1) for i in range(1, self.x_dim)]
        # Partition input space in each dimension. 
        u_grid_num = [int((self.inputs['ub'][i] - self.inputs['lb'][i] + 1e-5)/self.inputs['eta'][i]) for i in range(self.u_dim)]
        u_intervals = []
        for i in range(self.u_dim):
            dim_intervals = np.linspace(self.inputs['lb'][i], self.inputs['ub'][i], u_grid_num[i]+1).tolist()
            dim_intervals = [float('{:.4f}'.format(u + self.inputs['eta'][i]/2.)) for u in dim_intervals]
            u_intervals.append(dim_intervals)
        # A list consists of coordinates of abstract inputs (centers of partitions). 
        self.abstr_inputs = [[u_intervals[0][j]] for j in range(u_grid_num[0])]
        for i in range(1, self.u_dim):
            abstr_inputs_new = [u + [u_intervals[i][j]] for u in self.abstr_inputs for j in range(u_grid_num[i])]
            self.abstr_inputs = deepcopy(abstr_inputs_new)
        self.u_grid_basis = [reduce(mul, u_grid_num[i:], 1) for i in range(1, self.u_dim)]
        #print(self.abstr_states) 
        #print(self.abstr_inputs)
        #print(u_grid_num)
        #print(self.u_grid_basis)

    def get_v_info(self, V):
        """ Get satisfaction probability information """
        # When find min/max/average probability, not consider the goal and obstacles.
        if self.specs['type'] == 'safe':
            v_min, v_max, v_avg = np.amin(V), np.amax(V), np.average(V)
            self.v_info.append([v_min, v_max, v_avg])
        else:
            V_reduced = []
            for x_id, p in zip(range(self.nx), V):
                if x_id not in self.goal_obst_ids:
                    V_reduced.append(p)
            V_reduced = np.array(V_reduced)
            v_min, v_max, v_avg = np.amin(V_reduced), np.amax(V_reduced), np.average(V_reduced)
            self.v_info.append([v_min, v_max, v_avg])
        #print('\nProbabilities at the end of this iteration:')
        #print('v_min: ', self.v_info[-1][0])
        #print('v_max: ', self.v_info[-1][1])
        #print('v_avg: ', self.v_info[-1][2])

    def collect_sc_data(self, Sc):
        """ Run symbolic controller to collect training data """
        # NOTE: since the goal here is just for NN to mimic Sc, we add failed trajectories to training data as well.
        # TODO: put parameters in this function to the user customized file. 
        print('Running symbolic controller to collect data...')
        x_inits = []
        if self.x_dim == 2:
            for x0 in np.linspace(-9.0, 9.0, 20):
                for x1 in np.linspace(-9.0, 9.0, 20):
                    x_inits.append([x0, x1])
        elif self.x_dim == 5:
            for i in range(100):
                x = np.random.uniform(self.states['lb'], self.states['ub']).tolist()
                x_inits.append(x)
        else:
            assert False, 'System dimension does not match when collect Sc data'      
        num_succ = 0
        x_all_trajs, u_all_trajs = [], []
        for xi in x_inits:
            x_traj, u_traj = [], []
            x = xi
            success = True if self.specs['type'] == 'safe' else False
            for k in range(self.H):   
                x_id = self.x2id(x)
                if x_id == -1: # out of space
                    success = False
                    break
                elif x_id in self.goal_obst_ids - self.goal_ids: # hit obstacles
                    success = False 
                    break
                elif x_id in self.goal_ids: # reach goal
                    success = True
                    break
                else:     
                    u_id = Sc[k][x_id]
                    u = self.abstr_inputs[u_id]
                    x_traj.append(x + [k]) 
                    u_traj.append(u)
                    x = self.dynamics(x, u)
            if success:
                num_succ += 1
            if x_traj:
                x_all_trajs.append(x_traj)
                u_all_trajs.append(u_traj)
        num_traj = len(x_all_trajs)
        x_all_trajs = np.array(x_all_trajs)
        u_all_trajs = np.array(u_all_trajs)
        print('%d out of %d symbolic trajectories satisfy spec' % (num_succ, num_traj))     
        #print('x_all_trajs shape: ', x_all_trajs.shape)
        #print('u_all_trajs shape: ', u_all_trajs.shape)
        #print('Example trajectory and inputs:')
        #print(x_all_trajs[0])
        #print(u_all_trajs[0])
        X = x_all_trajs[0]
        y = u_all_trajs[0]
        for i in range(1, num_traj):
            X = np.concatenate((X, x_all_trajs[i]))
            y = np.concatenate((y, u_all_trajs[i]))
        return X, y







