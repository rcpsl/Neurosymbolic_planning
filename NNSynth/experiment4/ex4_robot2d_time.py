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
    'local_u_num': None # Set numbers if __main__
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
    'max_iter': 1,
    'expert_epochs': 3000,
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
    ex_id = 4.1
    local_u_nums = [2, 4, 7, 10]
    exe_times = []
    v_avgs = [] 

    for num in local_u_nums:
        print('\n\n==================== Call NNSynth with a New Number of Local Actions: %d ====================' % num**2)
        inputs['local_u_num'] = [num, num]
        t0 = time.time()
        nn_synth = NNSynth(states, inputs, specs, noise, dynamics)    
        nn_synth.main(params, load_nn, save_nn, save_symbolic, ex_id)
        t1 = time.time()
        exe_times.append(t1-t0)
        v_avgs.append(nn_synth.v_info[-1][2])
        print('\nSatisfaction probability V_avg: ', '{:.2%}'.format(v_avgs[-1]))
        print('NNSynth total execution time [s]: %.2f' % exe_times[-1])

    print('\n\n--------------------- Gather Information in Table 2 -----------------------\n')
    print('Benchmark: 2-d Robot\n') 
    for i in range(len(local_u_nums)):
        print('Number of local actions: ', local_u_nums[i]**2)
        print('Satisfaction probability V_avg: ', '{:.2%}'.format(v_avgs[i]))
        print('NNSynth total execution time [s]: %.2f\n' % exe_times[i])
