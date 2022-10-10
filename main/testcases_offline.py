import numpy as np
import pickle
import time

from NNControllerProject import *
import utility as utility

def find_nns_in_liveness_graphs(root, all_configs):
    """ 
    Find indices of NNs (selected transitions) for workspace configurations in all_configs.
    Index of a NN is a tuple (from node, to node, controller partition) that represents the corresponding transition.
    """
    nns_in_liveness_graphs = set()
    for config in all_configs:
        fin_name = root + 'live_graph_c' + str(config)
        with open(fin_name, 'rb') as fin:
            live_graph = pickle.load(fin)
        fin.close() 
        print('# of nodes in livegraph_c%d: %d' % (config, live_graph.num_nodes))  
        # Add all assigned transitions in this liveness graph. 
        for frm_name, frm_node in live_graph.node_dict.items():
            if frm_node.identity == 1:
                continue
            nn_idx = (frm_name, frm_node.next_state, frm_node.contr_partition)
            nns_in_liveness_graphs.add(nn_idx)
    print('# of different NNs in liveness graphs:', len(nns_in_liveness_graphs))
    #print(nns_in_liveness_graphs)
    return nns_in_liveness_graphs

def find_nns_need_process(nns_in_liveness_graphs, model_dir):
    """ Find NN indices in nns_in_liveness_graphs but the corresponding models are missing in model_dir """
    nns_need_process = set()
    for nn_idx in nns_in_liveness_graphs:
        frm_name, to_name, contr_partition = nn_idx
        try:
            file_name = model_dir + 'frm' + str(frm_name) + '_to' + str(to_name) + '_contr' + str(contr_partition) + '.h5'
            open(file_name, 'rb') 
        except IOError:
            nns_need_process.add(nn_idx)            
    #print('# of NNs in liveness graphs but have not been trained/projected:', len(nns_need_process))
    #print(nns_need_process)
    return nns_need_process

def generate_traj(nn_controller, transition_dict, model_test_dir):
    """ Generate a trajectory by running the trained NN controllers """
    init_abst_state = 108 #8 #517
    q = nn_controller.node_dict[init_abst_state]
    state = [np.random.uniform(low, up) for low, up in zip(q.q_low, q.q_up)]
   
    #state = [4.8, 1.9, 3.15] # data_0214 error

    # config 1 & 2
    #state = [2.4, 0.7, 2.]
    state = [2.4, 0.7, 1.5]

    # config 3 & 4
    #state = [0.4, 2.3, 6.]
    #state = [0.9, 2.3, 4.6]
    #state = [0.4, 2.6, 1.5]

    ################ CAV Workspace 1 ################
    #state = [4.8, 1.9, 3.15]
    #state = [0.2, 1.05, 4.9]
    #state = [0.96, 4.89, 4.7]
    
    ################ CAV Workspace 2 ################
    #state = [1.47, 0.11, 1.58]
    #state = [0.751, 4.86, 3.2]
    #state = [2.73, 4.7, 6.22]

    ################ CAV Workspace 1 Compare Global NN ################
    #state  = [0.35, 1.04, 4.82]
    #state  = [0.48, 1.48, 1.6]
    #state  = [1.98, 2.15, 0.1]

    ################ CAV Workspace 2 Compare Global NN ################
    #state = [2.9, 3.5, 4.7]
    #state = [0.76, 1.37, 3.21]
    #state = [2.26, 0.75, 3.1]

    print('\nInitial state:', state)
    traj, traj_abst, traj_input = nn_controller.run(state, transition_dict, model_test_dir)
    traj_dict = {'traj': traj, 'traj_abst': traj_abst, 'traj_input': traj_input}
    print('Length of traj:', len(traj))
    print('Length of abstract traj:', len(traj_abst))
    print('Length of input traj:', len(traj_input))
    #print('traj:', traj)
    #print('Abstract traj:', traj_abst)
    #print('Input traj:', traj_input)
    return traj_dict

def check_control_gains(nn_controller, traj_dict, transition_dict):
    """ Whether K, d at each step is within the assigned controller partition """
    for frm_name, K, d in zip(traj_dict['traj_abst'], traj_dict['traj_K'], traj_dict['traj_d']):
        if nn_controller.node_dict[frm_name].identity == 1:
            break
        to_name, contr_partition = transition_dict[frm_name] 
        assert nn_controller.check_gain_in_partition(contr_partition, K, d), '\nERROR: Control gain is not in the assigned controller partition\n\n\n\n'      
    print('Passed checking control gains in partitions')


if __name__ == "__main__":
    np.random.seed(0)
    train   = False
    project = True

    train_configs   = [1]
    project_configs = [1]
    test_configs    = [1]

    root = './data/'

    #root = './data_16angles_40/'   
    
    #root = './data_0217/'    
    #root = './data_0218_v1/'
    #root = './data_0218_v2/'
    #root = './data_0220/'          # No reduce; layer size 10; updates 200; runners 16
    
    post_graph_dir    = root + 'post_graph'
    ppo_agent_dir     = root + 'ppo_agent'
    nn_controller_dir = root + 'nn_controller'
    model_ppo_dir     = root + 'models_ppo/'
    model_project_dir = root + 'models_project/'    
    
    model_test_dir = model_project_dir
    traj_dir       = root + 'traj_c' + str(test_configs[0])

    save_ppo_agent      = True
    save_nn_controller  = True
    save_models_ppo     = True
    save_models_project = True
    save_traj           = True


    if train:
        print('\nThis is training\n')
        nns_in_liveness_graphs = find_nns_in_liveness_graphs(root, train_configs)
        nns_need_train         = find_nns_need_process(nns_in_liveness_graphs, model_ppo_dir)


        #for e in nns_need_train:
        #    break
        #nns_need_train = {e}     


        # Since safety and liveness graphs are config specific, we use posterior graph to access properties of nodes. 
        # Names of nodes in node_dict should never be modified, despite adding or removing nodes in safety and liveness graphs. 
        with open(post_graph_dir, 'rb') as fin:
            post_graph = pickle.load(fin)
        fin.close()  
        ppo_agent = PPOAgent()
        ppo_agent.__dict__.update(post_graph.__dict__)

        t0 = time.time()
        ppo_agent.train(nns_need_train, model_ppo_dir, save_models_ppo)
        t1 = time.time()  
        print('Training elapsed time:', t1-t0)
        if save_ppo_agent:        
            with open(ppo_agent_dir, 'wb') as fout:
                pickle.dump(ppo_agent, fout)
            fout.close()


    elif project:
        print('\nThis is projection\n')
        nns_in_liveness_graphs = find_nns_in_liveness_graphs(root, project_configs)
        nns_need_project       = find_nns_need_process(nns_in_liveness_graphs, model_project_dir)
        


        #nns_need_project = {(1222, 1097, 171)}



        # Access node properties using posterior graph. 
        with open(post_graph_dir, 'rb') as fin:
            post_graph = pickle.load(fin)
        fin.close()  
        nn_controller = NNControllerProject()
        nn_controller.__dict__.update(post_graph.__dict__)

        t0 = time.time()
        project_infeasible_nns = nn_controller.project_all_nns(nns_need_project, model_ppo_dir, model_project_dir, save_models_project)
        t1 = time.time()  
        print('project_infeasible_nns (total %d):' % len(project_infeasible_nns))
        print(project_infeasible_nns)
        print('Projection elapsed time:', t1-t0)
        if save_nn_controller:        
            with open(nn_controller_dir, 'wb') as fout:
                pickle.dump(nn_controller, fout)
            fout.close()

    else: 
        print('\nThis is testing\n')
        fin_name = nn_controller_dir
        with open(fin_name, 'rb') as fin:
            nn_controller = pickle.load(fin)
        fin.close()    

        assert len(test_configs) == 1, 'Specify one config for testing'
        nns_in_liveness_graphs = find_nns_in_liveness_graphs(root, test_configs)      
        nns_need_process       = find_nns_need_process(nns_in_liveness_graphs, model_test_dir)
        assert len(nns_need_process) == 0, 'If want to keep testing with NNs in liveness graph but have not been trained/projected, comment this assert'

        # Construct a dictionary that maps each abstract state to the assigned next_state and controller partition. 
        transition_dict = {}
        for nn_idx in nns_in_liveness_graphs:
            frm_name, to_name, contr_partition = nn_idx
            assert frm_name not in transition_dict.keys(), 'Given config, should have only one NN for each abstract state'
            transition_dict[frm_name] = (to_name, contr_partition)
        # Add goal to transition_dict. 
        for name, node in nn_controller.node_dict.items():
            if node.identity == 1:
                transition_dict[name] = (None, None) 

        traj_dict = generate_traj(nn_controller, transition_dict, model_test_dir)
        utility.plot_traj(test_configs[0], traj_dict['traj'])

        #traj_dict = nn_controller.retrieve_control_gains(traj_dict, transition_dict, model_test_dir)
        #check_control_gains(nn_controller, traj_dict, transition_dict)
        if save_traj:
            with open(traj_dir, 'wb') as fout:
                pickle.dump(traj_dict, fout)
            fout.close()