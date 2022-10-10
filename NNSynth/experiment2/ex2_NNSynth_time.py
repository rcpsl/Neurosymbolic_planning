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
    'eta': [0.4, 0.4, 0.4, 0.4, 0.4],
    'lb':  [18.8, 18.8, 18.8, 18.8, 18.8],
    'ub':  [21.2, 21.2, 21.2, 21.2, 21.2]
}

# The number of added local actions is the product of elements in 'local_u_num'.
# Since control inputs to be applied are centers of grids, please inflate the upper/lower bounds by half grid size
# if want to achieve the bounds.
inputs = {
    'dim': 2,
    'eta': [0.05, 0.05],
    'lb':  [-0.025, -0.025],
    'ub':  [ 1.025,  1.025],
    'local_u_num': [7, 7]
}

# safe | reach | reachavoid
specs = {
	'type': 'safe',
    'time_steps': 8
}

# Parameters of Gaussian noise. 
noise = {
    'pdf_truncation': 'fixed_truncation',
    'std_deviation': [0.01, 0.01, 0.01, 0.01, 0.01],
    'cutting_radius': [0.7, 0.7, 0.7, 0.7, 0.7]
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
    eta = 0.30
    beta = 0.022
    gamma = 0.05
    a = 1.0 - 2.0*eta - beta
    T_h = 50.0
    T_e = -1.0
    x0 = x[0]
    x1 = x[1]
    x2 = x[2]
    x3 = x[3]
    x4 = x[4]
    u0 = u[0]
    u1 = u[1]
    xx0 = (a - gamma*u0)*x0 + eta*(x4 + x1) + gamma*T_h*u0 + beta*T_e
    xx1 = a*x1 + eta*(x0 + x2) + beta*T_e
    xx2 = (a - gamma*u1)*x2 + eta*(x1 + x3) + gamma*T_h*u1 + beta*T_e
    xx3 = a*x3 + eta*(x2 + x4) + beta*T_e
    xx4 = a*x4 + eta*(x3 + x0) + beta*T_e
    xx = [xx0, xx1, xx2, xx3, xx4]
    return xx


load_nn       = False
save_nn       = True
save_symbolic = True


if __name__ == '__main__':
    ex_id = 2
    t0 = time.time()
    nn_synth = NNSynth(states, inputs, specs, noise, dynamics)    
    nn_synth.main(params, load_nn, save_nn, save_symbolic, ex_id)
    t1 = time.time()
    print('NNSynth total execution time [s]: %.2f\n' % (t1-t0))

     