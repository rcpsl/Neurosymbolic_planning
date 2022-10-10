import numpy as np
import pickle
import time
from shutil import copyfile

from NNControllerProject import *
import utility as utility
from testcases_offline import find_nns_in_liveness_graphs, find_nns_need_process


def copy_models(offline_configs, root, model_project_dir, models_exist_dir):
    """ Copy NNs that should have been trained to models_offline_dir """
    offline_nns = find_nns_in_liveness_graphs(root, offline_configs)
    for nn_idx in offline_nns:
        frm_name, to_name, contr_partition = nn_idx
        src = model_project_dir + 'frm' + str(frm_name) + '_to' + str(to_name) + '_contr' + str(contr_partition) + '.h5'
        dst = model_exist_dir   + 'frm' + str(frm_name) + '_to' + str(to_name) + '_contr' + str(contr_partition) + '.h5'
        copyfile(src, dst)

def generate_traj(nn_controller, test_transit_dict, offline_nns, use_online_nns, model_test_dir, save_online_nns, test_configs):
    """ Generate a trajectory by running the trained NN controllers """
    #init_abst_state = 108 #8 #517
    #q = nn_controller.node_dict[init_abst_state]
    #state = [np.random.uniform(low, up) for low, up in zip(q.q_low, q.q_up)]
   
    #state = [4.8, 1.9, 3.15] # data_0214 error

    # config 1 & 2
    #state = [2.4, 0.7, 2.]
    #state = [2.4, 0.7, 1.5]
    #state = [1.2, 1.4, 5.4]

    # config 3 & 4
    #state = [0.4, 2.3, 6.]
    #state = [0.9, 2.3, 4.6]
    #state = [0.4, 2.6, 1.5]
    #state = [0.4, 1.6, 5.2]

    config = test_configs[0]

    if config==1:
        state = [4.2, 1.3, 2.8]

    elif config==2:
        #state = [1.2, 1.4, 5.4]
        state = [0.7, 1.9, 5.6]

    elif config==3:
        #state = [0.4, 2.6, 1.5]
        #state = [0.9, 2.1, 2.2]
        state = [0.9, 1.9, 2.2]
    
    elif config==4:
        #state = [0.4, 1.6, 5.2]
        state = [0.9, 2.3, 4.11]

    else:
        assert False, 'config not defined'


    print('\nInitial state:', state)
    t0 = time.time()
    traj, traj_abst, traj_input = nn_controller.run_online(state, test_transit_dict, offline_nns, use_online_nns, model_test_dir, save_online_nns)
    t1 = time.time()
    traj_dict = {'traj': traj, 'traj_abst': traj_abst, 'traj_input': traj_input}
    print('Length of traj:', len(traj))
    print('Length of abstract traj:', len(traj_abst))
    print('Length of input traj:', len(traj_input))
    print('Exec time to generate traj:', t1-t0)
    #print('traj:', traj)
    #print('Abstract traj:', traj_abst)
    #print('Input traj:', traj_input)
    return traj_dict


if __name__ == "__main__":
    #np.random.seed(0)
    test = True
    # During test, whether use saved NNs that were previously trained online.
    use_online_nns  = True
    
    offline_configs = [1]
    test_configs    = [2]

    root = './data/'

    #root = './data_16angles_40/' 

    #root = './data_0220/'          # No reduce; layer size 10; updates 200; runners 16

    nn_controller_dir = root + 'nn_controller'
    model_project_dir = root + 'models_project/'

    model_exist_dir   = root + 'models_exist/'
    #model_exist_dir   = root + 'c1_models_exist_c4_0p9_2p3_4p6/'

    model_test_dir    = model_exist_dir
    traj_dir          = root + 'traj_c' + str(test_configs[0])

    save_online_nns   = False
    save_traj         = True

    if not test:
        copy_models(offline_configs, root, model_project_dir, model_exist_dir)

    else:
        fin_name = nn_controller_dir
        with open(fin_name, 'rb') as fin:
            nn_controller = pickle.load(fin)
        fin.close() 

        # NOTE: NNs in model_exist_dir include both NNs trained OFFLINE (for the configurations in offline_configs) and those trained ONLINE (during test).
        # NOTE: offline_nns only include those trained OFFLINE, but not those trained online.
        # As a result, offline_nns may not include all NNs that are currently in model_exist_dir, since some of them are trained online. 
        offline_nns = find_nns_in_liveness_graphs(root, offline_configs)

        # test_nns include NNs for the configuration in test_configs.
        assert len(test_configs) == 1, 'Specify one config for testing'
        test_nns            = find_nns_in_liveness_graphs(root, test_configs)      
        missing_offline_nns = test_nns - offline_nns
        missing_nns         = find_nns_need_process(test_nns, model_test_dir)
        print('# of NNs for test config but have not been trained OFFLINE:', len(missing_offline_nns))
        print('# of NNs for test config but have not been trained (either offline or online):', len(missing_nns))

        # Construct a dictionary that maps each abstract state to the assigned next_state and controller partition. 
        test_transit_dict = {}
        for nn_idx in test_nns:
            frm_name, to_name, contr_partition = nn_idx
            assert frm_name not in test_transit_dict.keys(), 'Given config, should have only one NN for each abstract state'
            test_transit_dict[frm_name] = (to_name, contr_partition)
        # Add goal to the dictionary. 
        for name, node in nn_controller.node_dict.items():
            if node.identity == 1:
                test_transit_dict[name] = (None, None) 

        traj_dict = generate_traj(nn_controller, test_transit_dict, offline_nns, use_online_nns, model_test_dir, save_online_nns, test_configs)
        utility.plot_traj(test_configs[0], traj_dict['traj'])   

        #traj_dict = nn_controller.retrieve_control_gains(traj_dict, transition_dict, model_test_dir)
        #check_control_gains(nn_controller, traj_dict, transition_dict)
        if save_traj:
            with open(traj_dir, 'wb') as fout:
                pickle.dump(traj_dict, fout)
            fout.close()          