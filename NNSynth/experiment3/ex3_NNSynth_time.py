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
    'dim': 5,
    'eta': [1., 1., 2., 2., 2.],
    'lb':  [0., 0., 0., 0., 0.],
    'ub':  [10., 10., 10., 10., 10.]
}

# The number of added local actions is the product of elements in 'local_u_num'.
# Since control inputs to be applied are centers of grids, please inflate the upper/lower bounds by half grid size
# if want to achieve the bounds.
inputs = {
    'dim': 2,
    'eta': [0.01, 0.01],
    'lb':  [0., 0.],
    'ub':  [1., 1.],
    'local_u_num': [4, 4]
}

# safe | reach | reachavoid
specs = {
	'type': 'safe',
    'time_steps': 7
}

# Parameters of Gaussian noise. 
noise = {
    'pdf_truncation': 'cutting_probability',
    'std_deviation': [0.7, 0.7, 0.7, 0.7, 0.7],
    'cutting_probability': 1e-4
}

# When max_iter is 1, only project NN to symbolic controller, but no lift. 
params ={
    'training_option': 'imitation',
    'max_iter': 1,
    'expert_epochs': 3000,
    'expert_batch_size': 1000, 
    'symbolic_epochs': 3000,
    'symbolic_batch_size': 1000
}

# System dynamics.
def dynamics(x, u):
    T = 0.0018
    v = 100
    L = 0.5
    q = 0.25
    x0 = x[0]
    x1 = x[1]
    x2 = x[2]
    x3 = x[3]
    x4 = x[4]
    u0 = u[0]
    u1 = u[1]
    xx0  = (1-(T*v/L))*x0 + (T*v/L)*x4 + 6*u0
    xx1  = (1-(T*v/L) - q)*x1 + (T*v/L)*x0
    xx2  = (1-(T*v/L))*x2 + (T*v/L)*x1 + 8*u1
    xx3  = (1-(T*v/L) - q)*x3 + (T*v/L)*x2
    xx4  = (1-(T*v/L))*x4 + (T*v/L)*x3  
    xp = [xx0, xx1, xx2, xx3, xx4]
    # zero-level saturation 
    for i in range(5):
        if xp[i] < 0:
            xp[i] = 0.
    return xp


load_nn       = False
save_nn       = True
save_symbolic = True


if __name__ == '__main__':
    ex_id = 3
    t0 = time.time()
    nn_synth = NNSynth(states, inputs, specs, noise, dynamics)    
    nn_synth.main(params, load_nn, save_nn, save_symbolic, ex_id)
    t1 = time.time()
    print('NNSynth total execution time [s]: %.2f\n' % (t1-t0))

     