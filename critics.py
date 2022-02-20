import numpy as np
from abc import ABC, abstractmethod
from aux_ import *

class ValueFunctionCritic(ABC):
    def __init__(self, featuresGen, env, gamma, lmbda):
        self.featuresGen = featuresGen
        self.features_dim = featuresGen.get_dim()
        self.gamma = gamma
        self.lmbda = lmbda
        self.env = env
    
    @abstractmethod
    def value_function(self, states):
        pass
    
    @abstractmethod
    def learn(self, policy, **kwargs):
        pass
    
    def emp_fixed_point(self, data, j_hat):
        A = np.zeros((self.features_dim, self.features_dim))
        b = np.zeros(self.features_dim)

        for i in np.arange(len(data)):
            r = transform_rewards(data[i]['r'], self.lmbda, j_hat)
            s = data[i]['s']            
            next_s = data[i]['next_s']

            phi = self.featuresGen.features(s)
            next_phi = self.featuresGen.features(next_s)
            delta_phi = phi - self.gamma * next_phi
            A = A + np.matmul(phi.T, delta_phi)
            b = b + np.matmul(phi.T, r)
            
        sol = np.matmul(np.linalg.pinv(A), b)

        return sol
    
    def evaluate_b2e(self, test_trajectory, j_hat):
        r = transform_rewards(test_trajectory[0]['r'], self.lmbda, j_hat)
        s = test_trajectory[0]['s']            
        next_s = test_trajectory[0]['next_s']
        return np.mean((r + self.gamma * self.value_function(next_s) - self.value_function(s))**2)


class MiniBatch_TD_V(ValueFunctionCritic):
    def __init__(self, featuresGen, env, gamma, lmbda):
        ValueFunctionCritic.__init__(self, featuresGen, env, gamma, lmbda)
        self.omega = np.random.rand(self.features_dim)
    
    def value_function(self, states):
        feats_matrix = self.featuresGen.features(states)
        return feats_matrix.dot(self.omega)
        
    def learn(self, policy, **kwargs):
        n_trajc = kwargs.get("n_trajc", 25)
        steps_per_trajc = kwargs.get("steps_per_trajc", 50)
        critic_step_size = kwargs.get("critic_step_size", 0.0001)
        critic_decay = kwargs.get("critic_decay", 0)
        independent_sampling = kwargs.get("independent_sampling", True)
        
        print("\tCollecting samples for estimating the expected reward...")
        d_j = sample(self.env, policy, n_batches=n_trajc, 
                         n_steps_per_batch=steps_per_trajc, diff_trajectories=True)
        print("\t{} trajectories, {} steps per trajectory.".format(n_trajc, steps_per_trajc))
        
        j_hat = estimate_expected_reward(trajectories=d_j, gamma=self.gamma)
        print("\tEstimated expected reward = {}".format(j_hat))
        
        if (independent_sampling):
            n_updates = kwargs.get("critic_n_batches", 10)
            batch_size = kwargs.get("critic_batch_size", 20)
            print("\tCollecting samples for learning the transformed value function...")
            d_v = sample(self.env, policy, n_batches=n_updates, 
                         n_steps_per_batch=batch_size, diff_trajectories=False, discounted_sampling=True, gamma=self.gamma)
            print("\t{} batches, {} steps per batch.".format(n_updates, batch_size))
            t_c = n_updates
            m = batch_size

        else:
            print("\tReusing the samples for learning the transformed value function!")
            d_v = d_j
            t_c = n_trajc
            m = steps_per_trajc
              
        for i in np.arange(t_c):
            transformed_rewards = transform_rewards(d_v[i]['r'], self.lmbda, j_hat)
            v_s = self.value_function(d_v[i]['s'])
            v_s_prime = self.value_function(d_v[i]['next_s'])
            td_err_vec = transformed_rewards + self.gamma * v_s_prime - v_s
            feats_matrix = self.featuresGen.features(d_v[i]['s'])
            update = td_err_vec.dot(feats_matrix) / m
            self.omega = self.omega + (critic_step_size/(1+critic_decay*i)) * update
        logs = {}
        
        return j_hat, logs, d_j, d_v
        

class LSTD_V(ValueFunctionCritic):
    def __init__(self, featuresGen, env, gamma, lmbda):
        ValueFunctionCritic.__init__(self, featuresGen, env, gamma, lmbda)
        self.omega = np.random.rand(self.features_dim)
    
    def value_function(self, states):
        feats_matrix = self.featuresGen.features(states)
        return feats_matrix.dot(self.omega)
        
    def learn(self, policy, **kwargs):
        n_trajc = kwargs.get("n_trajc", 25)
        steps_per_trajc = kwargs.get("steps_per_trajc", 50)
        independent_sampling = kwargs.get("independent_sampling", True)
        
        print("\tCollecting samples for estimating the expected reward...")
        d_j = sample(self.env, policy, n_batches=n_trajc, 
                         n_steps_per_batch=steps_per_trajc, diff_trajectories=True)
        print("\t{} trajectories, {} steps per trajectory.".format(n_trajc, steps_per_trajc))
        
        j_hat = estimate_expected_reward(trajectories=d_j, gamma=self.gamma)
        print("\tEstimated expected reward = {}".format(j_hat))
        
        if (independent_sampling):
            n_updates = kwargs.get("critic_n_batches", 10)
            batch_size = kwargs.get("critic_batch_size", 20)
            print("\tCollecting samples for learning the transformed value function...")
            d_v = sample(self.env, policy, n_batches=n_updates, 
                         n_steps_per_batch=batch_size, diff_trajectories=True)
            print("\t{} batches, {} steps per batch.".format(n_updates, batch_size))
            t_c = n_updates
            m = batch_size

        else:
            print("\tReusing the samples for learning the transformed value function!")
            d_v = d_j
            t_c = n_trajc
            m = steps_per_trajc
            
        print("\tSampling an extra trajectoy for testing...")
        test_trajectory = sample(self.env, policy, n_batches=1, n_steps_per_batch=steps_per_trajc)
            
        print("\tCalculating the empirical fixed point...")
        self.omega = self.emp_fixed_point(d_v, j_hat)
        emp_fixed_point_b2e = self.evaluate_b2e(test_trajectory, j_hat)
        print("\tEmpirical fixed point B2E = {}".format(emp_fixed_point_b2e))
        
        logs = {}
        logs["fixed_point_b2e"] = emp_fixed_point_b2e
        logs["t_c"] = t_c
        logs["m"] = m
        
        return j_hat, logs, d_j, d_v