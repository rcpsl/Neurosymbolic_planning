import numpy as np
import pickle

from LivenessGraph import *
from Environment import *

# TensorFlow 1.15.5
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam


class PPOAgent(object):
    def __init__(self):
        self.actor_layer_size = 6
        self.critic_num_layers = 2
        self.critic_layer_size = 10
        
        self.learning_rate = 1e-3
        self.gamma = 0.9
        self.loss_clipping = 0.2
        self.std_deviation = 1.0
        
        self.num_updates = 50 #150
        self.num_parallel_runners = 8 #16
        self.max_traj_length = 4
        self.fit_epochs = 10
        self.fit_batch_size  = 256
        
    def check_theta(self):     
        """ After normalization in PosteriorGraph, lower bound of theta should be within [0, 2*pi), while upper bound could be greater than 2*pi """
        for node in self.node_dict.values():
            if node.identity == 1:
                # Set theta range of the goal to [0, 4*pi), which is used when compute reward. 
                node.q_low[2] = 0
                node.q_up[2]  = 4*np.pi
            else:      
                assert 0 <= node.q_low[2] < 2*np.pi and node.q_low[2] < node.q_up[2] < node.q_low[2]+2*np.pi, 'Please change the [0, 4*pi] theta bound in ap2lr()' 

    def build_actor(self):
        """ Build NN for actor policy """
        state_input = Input(shape=(self.x_dim,), name='state_input')
        advantage = Input(shape=(1,), name='advantage')
        old_prediction = Input(shape=(self.u_dim,), name='old_prediction')
        x = Dense(self.actor_layer_size, activation='relu')(state_input)
        out_action = Dense(self.u_dim, name='out_action')(x)
        model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_action])
        model.compile(
            optimizer=Adam(lr=self.learning_rate),
            loss=[self.ppo_loss(advantage=advantage, old_prediction=old_prediction)]
            )
        #model.summary()
        return model

    def build_critic(self):
        """ Build NN for critic (value function) """
        state_input = Input(shape=(self.x_dim,), name='state_input')
        x = Dense(self.critic_layer_size, activation='relu')(state_input)
        for _ in range(self.critic_num_layers - 1):
            x = Dense(self.critic_layer_size, activation='relu')(x)
        out_value = Dense(1, name='out_value')(x)
        model = Model(inputs=[state_input], outputs=[out_value])
        model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
        #model.summary()
        return model

    def ppo_loss(self, advantage, old_prediction):
        """ PPO clipped loss function """
        # TODO: Add the entropy loss to avoid getting stuck at local minima.
        def loss(y_true, y_pred):
            var = K.square(self.std_deviation)
            denom = K.sqrt(2 * np.pi * var)
            prob_num = K.exp(-K.square(y_true - y_pred) / (2 * var))
            old_prob_num = K.exp(-K.square(y_true - old_prediction) / (2 * var))
            prob = prob_num/denom
            old_prob = old_prob_num/denom
            r = prob/(old_prob + 1e-10)
            clip_loss = -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1-self.loss_clipping, max_value=1+self.loss_clipping) * advantage))
            return clip_loss
        return loss 

    def train(self, nns_need_train, model_ppo_dir, save_models_ppo):
        """ Train NNs for all abstract states in specified_states """
        if self.has_theta:
            self.check_theta()
        print('\nstate dim: ', self.x_dim)
        print('action dim:', self.u_dim)     
        # Initialize environment build NNs.
        env = Environment(self.system_dict, self.has_theta, self.theta_partitions, self.disturb_bound)
        env.load_gp()
        actor  = self.build_actor()
        critic = self.build_critic()
        actor_init_weights  = actor.get_weights()
        critic_init_weights = critic.get_weights()
        
        for nn_idx in nns_need_train:
            frm_name, to_name, contr_partition = nn_idx
            frm_node = self.node_dict[frm_name]
            to_node  = self.node_dict[to_name]
            partition_low, partition_up = self.partitions_low[contr_partition], self.partitions_up[contr_partition]
            env.setup(frm_node, to_node, partition_low, partition_up)
            print('\nTransition: %d -> %d' % (frm_name, to_name), end = ', ')
            print('contr_partition =', contr_partition)
            #print('q_low:', frm_node.q_low)
            #print('q_up:', frm_node.q_up)
            assert frm_node.identity != -1, 'Obstacle should not appear in liveness graphs'
            if frm_node.identity == 1:
                print('\n========== WARNING: Should not try to train NN for the goal ==========\n')
                continue

            # Initialize all NNs with the same weights. 
            # TODO: Would it be better to not re-initialize weights, i.e., initial weights for the nest NN is the trained weights of the previous NN? 
            # Neighbor abstract states should have similar policy, which however may stuck at local optimal.
            actor.set_weights(actor_init_weights)
            critic.set_weights(critic_init_weights)
            actor = self.learn(actor, critic, env)
            if save_models_ppo:
                file_name = model_ppo_dir + 'frm' + str(frm_name) + '_to' + str(to_name) + '_contr' + str(contr_partition) + '.h5'
                actor.save_weights(file_name)

    def learn(self, actor, critic, env):
        """ PPO learning loop """
        for update in range(self.num_updates):
            verbose = False
            #if update % 20 == 0:
            #    print('\nupdate:', update)
            #    verbose = True
            states, actions, predicted_actions, rewards = self.run_agents(actor, env, verbose)
            old_predictions = predicted_actions
            predicted_values = critic.predict(states)            
            advantages = rewards - predicted_values
            #print('predicted_values.shape:', predicted_values.shape)
            actor.fit([states, advantages, old_predictions], [actions], epochs=self.fit_epochs, batch_size=self.fit_batch_size, verbose=verbose)
            critic.fit([states], [rewards], epochs=self.fit_epochs, batch_size=self.fit_batch_size, verbose=verbose)
        return actor

    def run_agents(self, actor, env, verbose):
        """ Make a batch of experiences """
        # A little different from gym environment, step() returns the next state and environment itself does not maintain state.
        batch_states, batch_actions, batch_predicted_actions, batch_rewards = [], [], [], []
        for runner in range(self.num_parallel_runners):
            state = env.reset()
            traj_states, traj_actions, traj_predicted_actions, traj_rewards = [], [], [], []
            for step in range(1000000):
                traj_states.append(state)
                action, predicted_action = self.get_action(actor, state)
                state, reward, done = env.step(state, action, verbose)
                traj_actions.append(action)
                traj_predicted_actions.append(predicted_action)
                traj_rewards.append(reward)
                
                if done or (step==self.max_traj_length-1):
                    traj_rewards = self.transform_reward(traj_rewards)
                    batch_states.extend(traj_states)
                    batch_actions.extend(traj_actions)
                    batch_predicted_actions.extend(traj_predicted_actions)
                    batch_rewards.extend(traj_rewards)
                    if verbose:
                        print('traj length:', len(traj_states))
                    break
        
        batch_states = np.array(batch_states)
        batch_actions = np.array(batch_actions)
        batch_predicted_actions = np.array(batch_predicted_actions)
        batch_rewards = np.array(batch_rewards).reshape(len(batch_rewards), 1)
        #print('batch_states.shape:', batch_states.shape)
        #print('batch_actions.shape:', batch_actions.shape)
        #print('batch_predicted_actions:', batch_predicted_actions.shape)
        #print('batch_rewards:', batch_rewards.shape)
        return batch_states, batch_actions, batch_predicted_actions, batch_rewards

    def get_action(self, actor, state):
        """ Sample action using Gaussian policy whose mean is given by the current trained NN """
        dummy_advantage, dummy_action = np.zeros((1, 1)), np.zeros((1, self.u_dim))        
        y = actor.predict([np.array(state).reshape(1, self.x_dim), dummy_advantage, dummy_action])
        predicted_action = y[0]
        action = predicted_action + np.random.normal(loc=0, scale=self.std_deviation, size=predicted_action.shape)
        return action, predicted_action

    def transform_reward(self, traj_rewards):
        """ Compute cost-to-go (Q-function) at each step (still call it reward), which is discounted sum of rewards in the current and future steps """
        # TODO: Generalized Advantage Estimation (GAE).
        for j in range(len(traj_rewards)-2, -1, -1):
            traj_rewards[j] += traj_rewards[j+1] * self.gamma
        return traj_rewards


