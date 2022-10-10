from math import cos, sin
import time
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.dirname(dir_path)
sys.path.append(dir_path)
from NNSynth import NNSynth

# State space dimension, grid size (eta), lower and upper bounds.
states = {
    'dim': 2,
    'eta': [  0.5,   0.5],
    'lb':  [-10.0, -10.0],
    'ub':  [ 10.0,  10.0]
}

# The number of added local actions is the product of elements in 'local_u_num'.
# Since control inputs to be applied are centers of grids, please inflate the upper/lower bounds by half grid size
# if want to achieve the bounds. 
inputs = {
    'dim': 2,
    'eta': [ 0.1,  0.1],
    'lb':  [-1.05, -1.05],
    'ub':  [ 1.05,  1.05],
    'local_u_num': [7, 7]
}

# safe | reach | reachavoid
specs = {
	'type': 'reachavoid',
    'time_steps': 16,
	'goal': [[ 5.0,  5.0], [7.0, 7.0]],
    'obst': [[-2.0, -2.0], [2.0, 2.0]]
}

# Parameters of Gaussian noise. 
noise = {
    'pdf_truncation': 'cutting_probability',
    'std_deviation': [0.75, 0.75],
	'cutting_probability': 1e-3
}

# When max_iter is 1, only project NN to symbolic controller, but no lift. 
params ={
    'training_option': 'imitation',
    'max_iter': None,
    'expert_epochs': None,  # Set numbers if __main__
    'expert_batch_size': 1000, 
    'symbolic_epochs': 2000,
    'symbolic_batch_size': 1000
}

# System dynamics.
def dynamics(x, u):
    xx0 = x[0] + 2 * u[0] * cos(u[1])
    xx1 = x[1] + 2 * u[0] * sin(u[1])
    xx = [xx0, xx1]
    return xx

load_nn       = False
save_nn       = True
save_symbolic = True


if __name__ == '__main__':
    ex_id = 5
    
    print('\n\n==================== Call NNSynth with max_iter = 1 (250 epochs) ====================')
    params['max_iter'] = 1
    params['expert_epochs'] = 250
    t0 = time.time()
    nn_synth = NNSynth(states, inputs, specs, noise, dynamics)    
    nn_synth.main(params, load_nn, save_nn, save_symbolic, ex_id)
    t1 = time.time()
    v_avgs_single = nn_synth.v_info[-1][2]
    print('\nSatisfaction probability V_avg: ', '{:.2%}'.format(v_avgs_single))
    print('NNSynth total execution time [s]: %.2f' % (t1-t0))
    
    print('\n\n============ Call NNSynth with max_iter = 5 (50 epochs per iteration) ===============')
    params['max_iter'] = 5
    params['expert_epochs'] = 50    
    t0 = time.time()
    nn_synth = NNSynth(states, inputs, specs, noise, dynamics)    
    nn_synth.main(params, load_nn, save_nn, save_symbolic, ex_id)
    t1 = time.time()
    print('\nSatisfaction probability V_avg: ', '{:.2%}'.format(nn_synth.v_info[-1][2]))
    print('NNSynth total execution time [s]: %.2f' % (t1-t0))
        
    print('\n\n--------------------- Gather Information in Table 3 -----------------------\n')
    print('Benchmark: 2-d Robot\n') 
    print('(base case) 1 iteration, 250 epochs, satisfaction probability V_avg: ', '{:.2%}'.format(v_avgs_single))
    print('\nFor the 5-iteration case, satisfaction probability after each iteration is:')
    print('Iteration 1, 50  epochs, satisfaction probability V_avg: ', '{:.2%}'.format(nn_synth.v_info[0][2]))
    print('Iteration 2, 100 epochs, satisfaction probability V_avg: ', '{:.2%}'.format(nn_synth.v_info[1][2]))
    print('Iteration 3, 150 epochs, satisfaction probability V_avg: ', '{:.2%}'.format(nn_synth.v_info[2][2]))
    print('Iteration 4, 200 epochs, satisfaction probability V_avg: ', '{:.2%}'.format(nn_synth.v_info[3][2]))
    print('Iteration 5, 250 epochs, satisfaction probability V_avg: ', '{:.2%}'.format(nn_synth.v_info[4][2]))






     