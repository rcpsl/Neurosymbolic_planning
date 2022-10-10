import numpy as np
from math import cos, sin
import pickle
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.dirname(dir_path)
sys.path.append(dir_path)
from NNSynth import NNSynth
from utils import plot_traj_robot2d
from ex1_NNSynth_time import dynamics

x01 = [-8.0, -8.0]
x02 = [-8.0, 0.0]
x03 = [-8.0, 8.0]
x04 = [-6.0, -2.0]
x05 = [-4.0, 0.0]
x06 = [-3., -1.5]
x07 = [0.0, -8.0]
x08 = [0.0, 4.0]
x09 = [3.5, -8.0]

x_inits = [x01, x02, x03, x04, x05, x06, x07, x08, x09]

def run(nn_synth, H, Sc):
    x_all_trajs = []
    for x0 in x_inits:
        print('x0:', x0, end=', ')
        x_traj, u_traj = [x0], []
        x = x0
        success = False
        for k in range(H):    
            x_id = nn_synth.x2id(x)
            if x_id == -1:
                print('Failed: out of the state space!\n')
                success = False
                break
            if x_id in nn_synth.goal_obst_ids - nn_synth.goal_ids: 
                print('Failed: hit obstacles!\n')
                success = False
                break    
            if x_id in nn_synth.goal_ids:
                print('Success: reached goal!\n')
                success = True
                break
            u_id = Sc[k][x_id]
            u = nn_synth.abstr_inputs[u_id]
            x = dynamics(x, u)
            x_traj.append(x)
            u_traj.append(u)
        #print('x0: ', x_traj[0])
        #print('x_traj length: ', len(x_traj))
        #print('u_traj length: ', len(u_traj))    
        if success:
            x_all_trajs.append(x_traj)
    plot_traj_robot2d(x_all_trajs)


if __name__ == '__main__':
    nn_synth = NNSynth(None, None, None, None, None)         
    fin_name = './data/symbolic'
    with open(fin_name, 'rb') as fin:
        symbolic = pickle.load(fin)
    nn_synth.states = symbolic['states']
    nn_synth.inputs = symbolic['inputs']  
    nn_synth.x_dim = symbolic['states']['dim']
    nn_synth.u_dim = symbolic['inputs']['dim']
    nn_synth.abstr_states = symbolic['abstr_states']
    nn_synth.abstr_inputs = symbolic['abstr_inputs']
    nn_synth.x_grid_basis = symbolic['x_grid_basis']
    nn_synth.u_grid_basis = symbolic['u_grid_basis']
    H = symbolic['specs']['time_steps']
    nn_synth.goal_obst_ids = symbolic['goal_obst_ids']
    nn_synth.goal_ids = symbolic['goal_ids']
    Sc = symbolic['Sc']
    V  = symbolic['V']
    
    run(nn_synth, H, Sc)

    #nx = V.shape[0]
    #print('Number of abstract states: %d\n' % nx)
    #V_reduced  = []
    #x_low_prob, p_low_prob = [], []
    #p_thld = 0.9
    #for x_id, p in zip(range(nx), V):
    #    if x_id not in goal_obst_ids:
            #print(x_id, 'p = ', p)
    #        V_reduced.append(p)
    #        if p < p_thld:
    #            x_low_prob.append(x_id)
    #print('\n\nStates with p < %.2f: ' % p_thld)
    #for x_id in x_low_prob:
    #    x = nn_synth.abstr_states[x_id]
    #    print(x, 'p = ', V[x_id])      
    #V_reduced = np.array(V_reduced)
    #v_min, v_max, v_avg = np.amin(V_reduced), np.amax(V_reduced), np.average(V_reduced)
    #print('\n\nv_min: ', v_min)
    #print('v_max: ', v_max)
    #print('v_avg: ', v_avg)