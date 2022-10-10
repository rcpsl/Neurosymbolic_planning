import numpy as np
from math import cos, sin
import pickle
import matplotlib.pyplot as plt
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.dirname(dir_path)
sys.path.append(dir_path)
from NNSynth import NNSynth
from ex2_NNSynth_time import dynamics

def run(nn_synth, H, Sc):
    count = 0
    x0_traj, x1_traj, x2_traj, x3_traj, x4_traj, x5_traj = [], [], [], [], [], []
    for i in range(10000):
        # Sample initial states
        x = np.random.uniform(nn_synth.states['lb'], nn_synth.states['ub']).tolist()
        x_traj, u_traj = [[0] + x], []
        #x = x0
        for k in range(H):    
            x_id = nn_synth.x2id(x)
            if x_id == -1:
                break
            u_id = Sc[k][x_id]
            u = nn_synth.abstr_inputs[u_id]
            x = dynamics(x, u)
            if all([nn_synth.states['lb'][i] <= x[i] <= nn_synth.states['ub'][i] for i in range(nn_synth.x_dim)]):
                x_traj.append([k+1] + x)
                u_traj.append(u)
        if len(x_traj) == 9:
            for x in x_traj:
                #print(x[0], x[1], x[2], x[3], x[4], x[5])
                x0_traj.append(x[0])
                x1_traj.append(x[1])
                x2_traj.append(x[2])
                x3_traj.append(x[3])
                x4_traj.append(x[4])
                x5_traj.append(x[5])
            count += 1    
            x0_traj.append(None)
            x1_traj.append(None) 
            x2_traj.append(None) 
            x3_traj.append(None) 
            x4_traj.append(None) 
            x5_traj.append(None) 
            #print('')
        if count == 100:
            break
    plot_traj(x0_traj, x1_traj, 1)
    plot_traj(x0_traj, x2_traj, 2)
    plot_traj(x0_traj, x3_traj, 3)
    plot_traj(x0_traj, x4_traj, 4)
    plot_traj(x0_traj, x5_traj, 5)


def plot_traj(x_traj, y_traj, id):
    fig, ax = plt.subplots()

    ax.plot(x_traj, y_traj, '-ok', color='blue', markersize=1)
    ax.set_ylim(18.5, 21.5)
    ax.set_xlim(0, 8)
    plt.axhline(y=18.8, color='r', linestyle='-')
    plt.axhline(y=21.2, color='r', linestyle='-')
    plt.ylabel('x'+str(id))
    plt.xlabel('t')
    plt.show()



def print_v_info(V):
    nx = V.shape[0]
    print('Number of abstract states: %d\n' % nx)
    V_reduced  = []
    x_low_prob, p_low_prob = [], []
    p_thld = 0.9
    for x_id, p in zip(range(nx), V):
        V_reduced.append(p)
        if p < p_thld:
            x_low_prob.append(x_id)
    #print('\n\nStates with p < %.2f: ' % p_thld)
    #for x_id in x_low_prob:
    #    x = nn_synth.abstr_states[x_id]
    #    print(x, 'p = ', V[x_id])      
    V_reduced = np.array(V_reduced)
    v_min, v_max, v_avg = np.amin(V_reduced), np.amax(V_reduced), np.average(V_reduced)
    print('\n\nv_min: ', v_min)
    print('v_max: ', v_max)
    print('v_avg: ', v_avg)

if __name__ == '__main__':
    np.random.seed(0)
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
    goal_obst_ids = symbolic['goal_obst_ids']
    Sc = symbolic['Sc']
    V  = symbolic['V']

    run(nn_synth, H, Sc)

    #print_v_info(V)

