import numpy as np
import pickle
import cplex as cplex
import polytope as pc

from PPOAgent import *

from tensorflow.keras.models import Sequential

class NNControllerProject(PPOAgent):
    def __init__(self):
        """ This class only project weights, training is done by PPOAgent """
        # NOTE: When project weights, make sure layer size consists with actor_layer_size in PPOAgent. 
        self.layer_size = 6
        
    def build_actor_predict(self):
        """ Build NN for actor policy """
        print('\nBuild actor layer size: %d\n' % self.layer_size)
        model = Sequential()
        model.add(Dense(self.layer_size, input_shape=(self.x_dim,), activation='relu'))
        model.add(Dense(self.u_dim))
        #model.summary()
        return model

    def project_all_nns(self, nns_need_project, model_ppo_dir, model_project_dir, save_models_project):
        """ Load NNs trained by PPO and project weights """
        model = self.build_actor_predict()
        # A set consists of indices of NNs that fail to project weights.
        project_infeasible_nns = set()
        for nn_idx in nns_need_project:
            frm_name, to_name, contr_partition = nn_idx
            print('\nTransition: %d -> %d' % (frm_name, to_name), end = ', ')
            print('contr_partition =', contr_partition)

            file_name = model_ppo_dir + 'frm' + str(frm_name) + '_to' + str(to_name) + '_contr' + str(contr_partition) + '.h5'
            model.load_weights(file_name)
            model, feasible = self.check_project_weights(model, frm_name, contr_partition)
            
            if not feasible:
                print('================= WARNING: ALL Attempts of Projecting Weights is Infeasible =================\n\n')
                project_infeasible_nns.add(nn_idx)
                continue
            if save_models_project:
                file_name = model_project_dir + 'frm' + str(frm_name) + '_to' + str(to_name) + '_contr' + str(contr_partition) + '.h5'
                model.save_weights(file_name)
        return project_infeasible_nns

    def check_project_weights(self, model, frm_name, contr_partition, ppo_actor=False):
        """ 
        Project weights if needed such that control gains are within the assigned controller partition.
        
        TODO: Currently only consider the NN has one hidden layer. 
        """
        feasible = True # Either projection is feasible or no need projection.
        weights = [layer.get_weights() for layer in model.layers]
        if ppo_actor:
            W, b = weights[1][0], weights[1][1] # Hidden layer weights
            V, c = weights[4][0], weights[4][1] # Output layer weights
        else:     
            W, b = weights[0][0], weights[0][1] # Hidden layer weights
            V, c = weights[1][0], weights[1][1] # Output layer weights
        Wt, Vt = np.transpose(W), np.transpose(V)
        assert W.shape == (self.x_dim, self.layer_size) and V.shape == (self.layer_size, self.u_dim), 'Dimension of weights does not match layer size'
        # A list consists of hidden layer weights.
        # Make sure the list stores V in column-first order, which is consistent with the real variable indices in CPLEX.
        Vc = np.concatenate((Vt, c), axis=None).tolist()
        #print('W = ')
        #print(W)
        #print('b = ', b)
        #print('V = ')
        #print(V)
        #print('c = ', c)
        #print('Vt = ', Vt)
        #print('Vc = ', Vc)
        #--- A debug example ---#
        #x = [0.49, 1.49, 5.9]
        #h = list(np.array(x) @ W + b)
        #h = [i if i >0 else 0 for i in h]
        #print('h:', h)
        #y = np.array(h) @ V + c
        #print(y)
        #X = np.array([x])
        #print('model prediction:', model.predict(X))

        # Find activation patterns corresponding to linear regions that intersect the given abstract state.
        all_aps = self.get_all_aps()
        isect_aps = self.find_isect_aps(Wt, b, frm_name, all_aps)
        assert isect_aps, 'Each abstract state should intersect at least one linear region'
        #print('Intersected APs:', isect_aps)

        # Check if all the affine functions associated to APs intersecting the abstract state satisfy K in the assigned controller partition.  
        need_project = False
        for ap in isect_aps:
            K, d = self.weights2gain(ap, W, V, b, c)
            #x = np.array(x).reshape((3,1))
            #print('Kx+d = ', K @ x + d)
            if not self.check_gain_in_partition(contr_partition, K, d):
                need_project = True
                break

        # Adjust weights in the output layer (V and c) for control gains (K, d) to be within the assigned controller partition.
        if need_project:
            print('Need Project Weights')
            V_proj, c_proj, feasible, added_constraints = self.projection(isect_aps, W, b, Vc, contr_partition)
            if feasible:
                if ppo_actor:
                    model.layers[4].set_weights([V_proj, c_proj])
                else:    
                    model.layers[1].set_weights([V_proj, c_proj])
                #for ap in isect_aps:
                #    K, d = self.weights2gain(ap, W, V_proj, b, c_proj)
                #    print('After projecting weights, when AP =', ap, end=', ')
                #    print('K, d are:')
                #    print(K)
                #    print('d:', d)
            else:         
                print('========== WARNING: This Attempt of Projecting Weights is Infeasible ==========\n')          
                #assert False, 'Infeasible projection'
                if not added_constraints:
                    # If infeasible due to all neurons are off but the controller partition does not contain zero, 
                    # set bias b to a large positive number to avoid all neurons are off. 
                    b_update = np.array([1.]*self.layer_size)
                    model.layers[0].set_weights([W, b_update])
        return model, feasible

    def get_all_aps(self):
        """ Return all activation patterns (sign combinations of hidden layer neurons), including those not allowed by the NN weights """
        all_aps = [[i] for i in [1, -1]]
        for count in range(1, self.layer_size):
            all_aps_new = [ap + [i] for ap in all_aps for i in [1, -1]]
            all_aps = deepcopy(all_aps_new)
        return all_aps

    def find_isect_aps(self, Wt, b, frm_name, all_aps):    
        """ Among all APs, find those intersect the given abstract state """
        isect_aps = []
        for ap in all_aps:
            lr = self.ap2lr(Wt, b, ap)
            if self.check_lr_isect(lr, frm_name):
                isect_aps.append(ap)
        return isect_aps 

    def ap2lr(self, Wt, b, ap):
        """ 
        Given an activation pattern, return the corresponding NN linear region
        NOTE: NN linear regions may be not full dimension. 
        """ 
        A, B = Wt.copy(), b.copy()
        for i in range(len(ap)):
            if ap[i] == 1:
                # ReLU is on: flip signs of the i-th row of A.
                A[i] = -1 * A[i]
            else:
                # ReLU is off: flip sign of the i-th element of B.
                B[i] = -1 * B[i]
        # Only consider linear regions within some boundary that contrains all abstract states.  
        A = np.concatenate((A, np.identity(self.x_dim), -np.identity(self.x_dim)))
        bdry_low, bdry_up = np.array(self.x_span[0]).copy(), np.array(self.x_span[1]).copy()
        if self.has_theta:
            # Since theta lower bound of each abstract state is in [0, 2*pi), and upper bound is lees than lower bound plus 2*pi, 
            # it is enough to consider linear regions with theta in the range [0, 4*pi].
            # NOTE: Need to set this boundary contain theta ranges in all abstract states, instead of that in x_span 
            # (theta range in abstract states may be not in x_span, see the example in PosteriorGraph.normalize_theta). 
            bdry_low[2] = 0
            bdry_up[2]  = 4*np.pi
        B = np.concatenate((B, bdry_up, -bdry_low))
        #print('ap:', ap) 
        #print('Wt:', )
        #print(Wt)
        #print('A:')
        #print(A)
        #print('b:', b)
        #print('B:', B)
        #print('\n')
        p = pc.Polytope(A, B)
        p = pc.reduce(p)
        #print(pc.extreme(p))
        return p 

    def check_lr_isect(self, lr, frm_name):
        """ Whether linear region lr and abstract state frm_name intersect """
        frm_poly = self.node_dict[frm_name].polytope
        #print(pc.extreme(frm_poly))
        isect = frm_poly.intersect(lr) 
        rp, xp = isect.cheby
        if rp > const.err:
            return True
        else:
            return False   

    def weights2gain(self, ap, W, V, b, c):   
        """ Given NN weights, compute the corresponding K, d in the feedback controller u = Kx + d """
        W_relu, b_relu = self.relu(W, b, ap)
        K = np.transpose(W_relu @ V)
        d = b_relu @ V + c
        d = d.reshape(self.u_dim, 1)
        return K, d

    def relu(self, W, b, ap):
        """ Set the i-th column of W and the i-th element of b to zero if the i-th neuron is off """
        W_relu, b_relu  = W.copy(), b.copy()
        indices = [i for i in range(len(ap)) if ap[i] == -1]
        for i in indices:
            W_relu[:, i] = 0     
            b_relu[i] = 0
        #print('ap:', ap) 
        #print('W:', )
        #print(W)
        #print('W_relu:')
        #print(W_relu)
        #print('b:', b)
        #print('b_relu:', b_relu)
        return W_relu, b_relu     

    def check_gain_in_partition(self, pid, K, d):
        """ Whether K,d is contained in the assigned controller partition """
        in_partition = True
        Kd = np.concatenate((K, d), axis=None) # Note this assumes the order of entries in partitions_low/up is row-first order for K, and d is attached at the end. 
        partition_low, partition_up = self.partitions_low[pid], self.partitions_up[pid]
        #print('pid:', pid)
        #print('Kd:', Kd)
        #print('partition_low:', partition_low)
        #print('partition_up:', partition_up)
        for K_dim, partition_dim_low, partition_dim_up in zip(Kd, partition_low, partition_up):
            if not (partition_dim_low-const.err <= K_dim <= partition_dim_up+const.err):
                in_partition = False
                break
        return in_partition          

    def projection(self, isect_aps, W, b, Vc, contr_partition, verbose=False):
        """ Project weights by solving a QP """
        num_reals = self.u_dim * self.layer_size + self.u_dim
        convex_solver = cplex.Cplex()
        if not verbose:
            convex_solver.set_results_stream(None) 
            convex_solver.set_log_stream(None)
        convex_solver.objective.set_sense(convex_solver.objective.sense.minimize)
        real_vars =  ['x'+str(i) for i in range(num_reals)]
        convex_solver.variables.add(
            names = real_vars,
            ub    = [cplex.infinity] * num_reals,
            lb    = [-cplex.infinity] * num_reals
            )
        # Minimize the Euclidean distance to the current output layer weights. 
        for real, coef in zip(real_vars, Vc):
            convex_solver.objective.set_linear(real, -2*coef)
        params = [2.] * num_reals
        convex_solver.objective.set_quadratic(params)     
        for ap in isect_aps:
            added_constraints = self.add_ap_constraints(convex_solver, real_vars, ap, W, b, contr_partition)
            if not added_constraints:
                return list(), list(), False, added_constraints
        convex_solver.solve()
        # Extract projected weights from solution. 
        if convex_solver.solution.get_status() == 1:
            reals_model = convex_solver.solution.get_values(real_vars)
            V_model = np.array([reals_model[j*self.layer_size: (j+1)*self.layer_size] for j in range(self.u_dim)])
            V_proj = np.transpose(V_model)
            c_proj = np.array(reals_model[self.layer_size*self.u_dim: self.layer_size*self.u_dim+self.u_dim])
            #print('reals model:', reals_model)
            #print('V_proj:')
            #print(V_proj)
            #print('c_proj:', c_proj)
            return V_proj, c_proj, True, added_constraints
        else: 
            return list(), list(), False, added_constraints

    def add_ap_constraints(self, convex_solver, real_vars, ap, W, b, pid, verbose=False):
        """ Add constraints that control gains corresponding to the given AP are contained in the assigned controller partition """
        added_constraints = True
        p_low, p_up = self.partitions_low[pid], self.partitions_up[pid]
        W_relu, b_relu = self.relu(W, b, ap)

        # Each entry of K is determined by the dot product of a row of W_relu (coefficients) and a column of V (optimization variables).
        for i in range(W_relu.shape[0]):
            #print('i:', i)
            W_row = W_relu[i].tolist()
            num_zero_entries = len([entry for entry in W_row if abs(entry) < const.err])
            if num_zero_entries == W_relu.shape[1]:
                # If all entries in a row of W_relu are 0, then the corresponding entries of K can only be zero, independent of V, and no need add constraints. 
                # Indices of entries of K determined by this row of W_relu.
                K_entry_indices = [i + j*self.x_dim for j in range(self.u_dim)]
                #print('K_entry_indices:', K_entry_indices)
                for K_entry_id in K_entry_indices:
                    #assert p_low[K_entry_id]-const.err <= 0 <= p_up[K_entry_id]+const.err, \
                    #'Projection is infeasible since K entry is 0 but 0 is not in the controller partition'
                    if not (p_low[K_entry_id]-const.err <= 0 <= p_up[K_entry_id]+const.err):
                        print('========== WARNING: Projection is infeasible since K entry is 0 but 0 is not in the controller partition ==========')
                        added_constraints = False
                        return added_constraints
            else:
                # Add constraints on all K entries that are determined by this row of W_relu.
                for j in range(self.u_dim):
                    K_entry_id = i + j*self.x_dim
                    reals = [real_vars[s] for s in range(j*self.layer_size, (j+1)*self.layer_size)]
                    #print('K_entry_id:', K_entry_id)
                    #print(reals)
                    ap_constraints = self.LPClause(np.array([W_row]), [p_low[K_entry_id]], reals, sense='G')
                    self.add_convex_constraint(convex_solver, ap_constraints)
                    ap_constraints = self.LPClause(np.array([W_row]), [p_up[K_entry_id]], reals, sense='L')
                    self.add_convex_constraint(convex_solver, ap_constraints)
        
        # Add constraints on d, where d is the bias term in u = Kx+d.
        b_relu = b_relu.tolist()
        b_relu.append(1.)
        for j in range(self.u_dim):
            #print('j:', j)
            K_entry_id = self.x_dim * self.u_dim + j
            reals = [real_vars[s] for s in range(j*self.layer_size, (j+1)*self.layer_size)]
            bias_var_id = self.layer_size * self.u_dim + j
            reals.append(real_vars[bias_var_id])
            #print('K_entry_id:', K_entry_id)
            #print(reals)
            ap_constraints = self.LPClause(np.array([b_relu]), [p_low[K_entry_id]], reals, sense='G')
            self.add_convex_constraint(convex_solver, ap_constraints)
            ap_constraints = self.LPClause(np.array([b_relu]), [p_up[K_entry_id]], reals, sense='L')
            self.add_convex_constraint(convex_solver, ap_constraints)
        return added_constraints

    def add_convex_constraint(self, convex_solver, constraint, name=None):
        # XS: add argument name
        #print constraint['lin_expr']
        if name:
            if constraint['type'] == 'LP':
                names = [name] * len(constraint['senses'])
                convex_solver.linear_constraints.add(
                    lin_expr = constraint['lin_expr'],
                    senses   = constraint['senses'],
                    rhs      = constraint['rhs'],
                    names    = names
                )
            elif constraint['type'] == 'QP':
                convex_solver.quadratic_constraints.add(
                    quad_expr = constraint['quad_expr'],
                    lin_expr  = constraint['lin_expr'],
                    sense     = constraint['sense'],
                    rhs       = constraint['rhs'],
                    names     = names
                )
        else:
            if constraint['type'] == 'LP':
                convex_solver.linear_constraints.add(
                    lin_expr = constraint['lin_expr'],
                    senses   = constraint['senses'],
                    rhs      = constraint['rhs'],
                )
            elif constraint['type'] == 'QP':
                convex_solver.quadratic_constraints.add(
                    quad_expr = constraint['quad_expr'],
                    lin_expr  = constraint['lin_expr'],
                    sense     = constraint['sense'],
                    rhs       = constraint['rhs'],
                )

    def LPClause(self, A, b, rVars, sense = "L"):
        # x = rVars
        # sense = "L", "G", "E"
        #A x {sense} b, A is a matrix and b is a vector with same dimension
        # TODO: add a dimension check
        # Put the constraint in CPLEX format, example below:
        # lin_expr = [cplex.SparsePair(ind = ["x1", "x3"], val = [1.0, -1.0]),\
        #    cplex.SparsePair(ind = ["x1", "x2"], val = [1.0, 1.0]),\
        #    cplex.SparsePair(ind = ["x1", "x2", "x3"], val = [-1.0] * 3),\
        #    cplex.SparsePair(ind = ["x2", "x3"], val = [10.0, -2.0])],\
        # senses = ["E", "L", "G", "R"],\
        # rhs = [0.0, 1.0, -1.0, 2.0],\
        numOfRows       = len(b)
        lin_expr        = list()
        for counter in range(0, numOfRows):
            lin_expr.append(cplex.SparsePair(ind = rVars, val = A[counter,:]))
        rhs             = b
        senses          = [sense] * numOfRows
        constraint  = {'type':'LP', 'lin_expr':lin_expr, 'rhs':rhs, 'x':rVars, 'senses':senses, 'A':A}
        return constraint

    def run(self, state, transition_dict, model_test_dir):
        """ Run the trained NN controller multiple steps """
        run_steps = 100

        model = self.build_actor_predict()
        env = Environment(self.system_dict, self.has_theta, self.theta_partitions, self.disturb_bound)
        env.load_gp()
        traj, traj_abst, traj_input  = [], [], []

        for step in range(run_steps):
            traj.append(state)
            frm_name = self.localization(state, transition_dict)
            traj_abst.append(frm_name)
            if step % 1 == 0:
                print('\nStep:', step)     
                print('Current state:', state)
                print('Inside abstract state:', frm_name)
            if self.node_dict[frm_name].identity == 1:
                print('Reached Goal')
                break
            # Load the corresponding NN.
            to_name, contr_partition = transition_dict[frm_name] 
            file_name = model_test_dir + 'frm' + str(frm_name) + '_to' + str(to_name) + '_contr' + str(contr_partition) + '.h5'
            try:
                model.load_weights(file_name)
            except IOError:
                print(file_name, end='\n\n')
                assert False, 'frm_name is in liveness graph but the corresponding NN is missing'
            X = np.array([state])
            y = model.predict(X)
            u = y[0]
            #print('u:', u)
            traj_input.append(u)
            state = env.update_state(state, u)
        return traj, traj_abst, traj_input

    def localization(self, state, transition_dict):
        """ Decide which abstract state contains the current state """
        abst_name = None
        for frm_name in transition_dict.keys():
            frm_node = self.node_dict[frm_name]
            if self.is_inside_state(frm_node, state):
                abst_name = frm_name
                break
        if abst_name is None:
            print('\nCurrent state:', state) 
            assert False, '===== ERROR: Current abstract state is not EVEN in liveness graph ====='
        return abst_name

    def is_inside_state(self, node, state):
        """ Whether state is in the abstract state represented by node """
        is_inside = True
        for q_dim_low, q_dim_up, state_dim in zip(node.q_low, node.q_up, state):
            if not (q_dim_low <= state_dim <= q_dim_up):
                is_inside = False
                break
        return is_inside


    def run_online(self, state, test_transit_dict, offline_nns, use_online_nns, model_test_dir, save_online_nns):
        """ During test if need NNs that are not trained, then learn policy online """
        run_steps = 37

        # Setup PPO parameters when training online.
        self.actor_layer_size = self.layer_size
        self.critic_num_layers = 2
        self.critic_layer_size = 10

        self.learning_rate = 1e-3
        self.gamma = 0.9
        self.loss_clipping = 0.2
        self.std_deviation = 1.0

        self.num_updates = 10 #150
        self.num_parallel_runners = 8 #16
        self.max_traj_length = 4
        self.fit_epochs = 10
        self.fit_batch_size  = 256

        loaded_model = self.build_actor_predict()
        env = Environment(self.system_dict, self.has_theta, self.theta_partitions, self.disturb_bound)
        env.load_gp()
        traj, traj_abst, traj_input  = [], [], []
        # A buffer saves actor models that are currently training online.  
        # TODO: Also need a buffer for critic models
        model_buffer = {}

        actor = self.build_actor()
        critic = self.build_critic()

        for step in range(run_steps):
            traj.append(state)
            frm_name = self.localization(state, test_transit_dict)
            traj_abst.append(frm_name)
            if step % 1 == 0:
                print('\nStep:', step)     
                print('Current state:', state)
                print('Inside abstract state:', frm_name)
            if self.node_dict[frm_name].identity == 1:
                print('Reached Goal')
                break

            # Load the corresponding NN from model_test_dir. 
            to_name, contr_partition = test_transit_dict[frm_name] 
            nn_idx = (frm_name, to_name, contr_partition)
            file_name = model_test_dir + 'frm' + str(frm_name) + '_to' + str(to_name) + '_contr' + str(contr_partition) + '.h5'
            found_model = True
            if use_online_nns:
                # Any model in model_test_dir can be used, including those trained offline and those trained previously online. 
                try:
                    loaded_model.load_weights(file_name)
                except IOError:
                    found_model = False
                    print('Current NN has not been trained either OFFLINE or ONLINE:', nn_idx)
            else:
                # Only allow to use NNs trained offline.
                if nn_idx in offline_nns:
                    loaded_model.load_weights(file_name)
                else:      
                    found_model = False
                    print('Current NN has not been trained OFFLINE:', nn_idx)

            # Predict if have loaded the model, otherwise, train the model online. 
            if found_model:
                X = np.array([state])
                y = loaded_model.predict(X)
                u = y[0]
            else:
                #actor = model_buffer.get(nn_idx)
                #if actor is not None:
                if False:
                    # TODO: Further train the model, project weights, and update model_buffer. 
                    print('Current NN is in model_buffer, keep training (currently direct use not further train)...')

                else:
                    # Initialize the models, initialize weights from nearby NN, train, project weights, and add to model_buffer.
                    print('Current NN is NOT in model_buffer, initialize a new NN...')
                    #actor = self.build_actor()
                    #critic = self.build_critic()

                    train_online = False
                    if train_online:
                        # Initialize weights by copying closest NNs.
                        closest_actor_idx  = self.find_nearby_nns(frm_name, contr_partition, offline_nns)
                        frm_name_exist, to_name_exist, contr_partition_exist = closest_actor_idx
                        closest_actor_file =  model_test_dir + 'frm' + str(frm_name_exist) + '_to' + str(to_name_exist) + '_contr' + str(contr_partition_exist) + '.h5'
                        actor.load_weights(closest_actor_file)
                        # Further training.
                        partition_low, partition_up = self.partitions_low[contr_partition], self.partitions_up[contr_partition]
                        frm_node, to_node = self.node_dict[frm_name], self.node_dict[to_name]
                        env.setup(frm_node, to_node, partition_low, partition_up)
                        actor = self.learn(actor, critic, env)

                    actor, feasible = self.check_project_weights(actor, frm_name, contr_partition, ppo_actor=True)
                    if not feasible:
                        # TODO: Further train the model and try to project again. 
                        assert False, 'ALL Attempts of Projecting Weights is Infeasible'
                    model_buffer[nn_idx] = actor
                
                # Use online trained actor to predict. 
                dummy_advantage, dummy_action = np.zeros((1, 1)), np.zeros((1, self.u_dim))        
                y = actor.predict([np.array(state).reshape(1, self.x_dim), dummy_advantage, dummy_action])
                u = y[0]

            traj_input.append(u)
            state = env.update_state_perfect(state, u)

        print('\n# NNs trained online: %d\n' % len(model_buffer))

        # Save models in buffer. 
        if save_online_nns:
            print('\nSave %d online trained NNs\n' % len(model_buffer))
            for nn_idx, actor in model_buffer.items():
                frm_name, to_name, contr_partition = nn_idx
                file_name = model_test_dir + 'frm' + str(frm_name) + '_to' + str(to_name) + '_contr' + str(contr_partition) + '.h5'
                actor.save_weights(file_name)

        return traj, traj_abst, traj_input


    def find_nearby_nns(self, frm_name, contr_partition, offline_nns):
        """ Find actor model that is trained OFFLINE to be used for transfer learning """
        # TODO: Search among not only NNs trained offline, but also online.
        # TODO: Save critic as well.
        # TODO: Weighted Distance between states and controller partitions. 
        incumbent_dist = const.inf
        nn_idx_closest = None
        for nn_idx in offline_nns:
            frm_name_exist, to_name_exist, contr_partition_exist = nn_idx
            d = self.compute_dist(frm_name, contr_partition, frm_name_exist, contr_partition_exist)
            if d < incumbent_dist:
                incumbent_dist = d
                nn_idx_closest = nn_idx
        return nn_idx_closest

    def compute_dist(self, frm_name, contr_partition, frm_name_exist, contr_partition_exist):
        frm_center       = [(low+up)/2 for low, up in zip(self.node_dict[frm_name].q_low, self.node_dict[frm_name].q_up)]
        frm_center_exist = [(low+up)/2 for low, up in zip(self.node_dict[frm_name_exist].q_low, self.node_dict[frm_name_exist].q_up)]
        dist_in_dims = [(frm_center[i]-frm_center_exist[i])**2 for i in range(len(frm_center))]
        # Distance between theta need to take wrap-around into account. 
        if self.has_theta:
            frm_theta       = (frm_center[2] + 100*np.pi) % (2*np.pi) 
            frm_theta_exist = (frm_center_exist[2] + 100*np.pi) % (2*np.pi)
            theta_diff = abs(frm_theta - frm_theta_exist)
            theta_diff = theta_diff if theta_diff < np.pi else 2*np.pi-theta_diff
            dist_in_dims[2] = theta_diff**2
        d = sqrt(sum(dist_in_dims))
        return d

    def retrieve_control_gains(self, traj_dict, transition_dict, model_test_dir):
        """ Given a trajectory and trained NNs, find the control gains K, d in the controller u = Kx + d at each time step """
        model = self.build_actor_predict()
        traj_K, traj_d = [], []
        for state, frm_name in zip(traj_dict['traj'], traj_dict['traj_abst']):
            if self.node_dict[frm_name].identity == 1:
                break
            # Load the corresponding NN.
            to_name, contr_partition = transition_dict[frm_name] 
            file_name = model_test_dir + 'frm' + str(frm_name) + '_to' + str(to_name) + '_contr' + str(contr_partition) + '.h5'
            model.load_weights(file_name)
            # Decide which AP (linear region of NN) contains the current state.
            # TODO: Which is more efficient: search over all APs, or first find APs intersect the current abstract state then only search over the intersected APs?
            weights = [layer.get_weights() for layer in model.layers]
            W, b = weights[0][0], weights[0][1] # Hidden layer weights
            V, c = weights[1][0], weights[1][1] # Output layer weights
            Wt = np.transpose(W)
            assert W.shape == (self.x_dim, self.layer_size) and V.shape == (self.layer_size, self.u_dim), 'Dimension of weights does not match layer size'
            all_aps = self.get_all_aps()
            #isect_aps = self.find_isect_aps(Wt, b, frm_name, all_aps)
            current_ap = None
            for ap in all_aps:
                lr = self.ap2lr(Wt, b, ap)
                if not pc.is_fulldim(lr):
                    continue
                if pc.is_inside(lr, state):
                    current_ap = ap
                    break
            assert current_ap is not None, 'State is not in any linear region, which should not happen. \
                                            Make sure the theta bound used in ap2lr() contains theta ranges in all abstract states.'
            K, d = self.weights2gain(current_ap, W, V, b, c)
            traj_K.append(K)
            traj_d.append(d)
        traj_dict['traj_K'], traj_dict['traj_d'] = traj_K, traj_d
        print('\nDone retrieve control gains, length of K traj:', len(traj_K))
        return traj_dict


