import numpy as np
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic

from ReducedGraph import *
from Environment import *

class GP():
    def __init__(self):
        # TODO: Separate GP for each dimension. 
        self.gpr = None

    def sample_data(self):
        """ Sample training data for GP """
        env = Environment(self.system_dict, self.has_theta, self.theta_partitions, self.disturb_bound)
        X_train, y_train = [], []
        for frm_node in self.node_dict.values():
            for _ in range(1):
                state     = [np.random.uniform(low, up) for low, up in zip(frm_node.q_low, frm_node.q_up)]
                action    = [np.random.uniform(-2, 2)]
                perfect_state = env.update_state_perfect(state, action)
                nominal_state = env.update_state_nominal(state, action)
                residual_x = perfect_state[0] - nominal_state[0]
                #residual_y = perfect_state[1] - nominal_state[1]
                X_train.append(state)
                y_train.append(residual_x)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        #X_train = np.array([[0.1, 0.2, 0.4], [1.1, 1.2, 1.4]])
        #y_train = np.array([9.1, 7.4])
        print('X_train.shape', X_train.shape)
        print('y_train.shape', y_train.shape)
        return X_train, y_train

    def gp_regression(self, X_train, y_train):
        k1 = 1.0 * RBF(length_scale=1.0) 
        k2 = WhiteKernel(noise_level=0.4)
        k3 = 0.66**2 * RationalQuadratic(length_scale=1.2, alpha=0.78)
        k4 = 0.18**2 * RBF(length_scale=0.134)
        kernel = k1 + k2
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=0, n_restarts_optimizer=20)
        gpr.fit(X_train, y_train)
        return gpr      

if __name__ == '__main__':

    root = './data/'

    load_reduced_graph = True
    if load_reduced_graph:
        fin_name = root + 'reduced_graph'
        with open(fin_name, 'rb') as fin:
            reduced_graph = pickle.load(fin)
        fin.close()   

    gp = GP()
    gp.__dict__.update(reduced_graph.__dict__)

    X_train, y_train = gp.sample_data()
    gpr = gp.gp_regression(X_train, y_train)

    #X_test = np.array([[0.5, 0.4, 0.6]])
    #mu, sigma = gpr.predict(X_test, return_std=True)
    #print('mu:', mu)
    #print('sigma:', sigma)

    save_gpr = True
    if save_gpr:        
        fout_name = root + 'gpr'
        with open(fout_name, 'wb') as fout:
            pickle.dump(gpr, fout)
        fout.close()   