import numpy as np
from abc import ABC, abstractmethod
from scipy import stats

class Policy(ABC):
    def __init__(self, featuresGen):
        self.featuresGen = featuresGen
        self.features_dim = featuresGen.get_dim()
    
    @abstractmethod
    def pi(self, state, action):
        pass
        
    @abstractmethod
    def sample(self, state):
        pass
        
    @abstractmethod
    def score(self, state, action):
        pass
    
    @abstractmethod
    def get_gradient_shape(self):
        pass
    
    @abstractmethod
    def gradient_ascent_update(self, gradient, step_size):
        pass



class LinearGaussian1DPolicy(Policy):
    def __init__(self, featuresGen):
        Policy.__init__(self, featuresGen)
        self.theta = np.random.rand(2, self.features_dim)

    def spawn(self):
        cls = self.__class__
        return cls(self.featuresGen)    
        
    def get_mean_and_sd(self, state, abs_sd=True):
        [mean,sd] = self.theta.dot(self.featuresGen.features(state))
        if (abs_sd):
            sd = np.abs(sd)
        return mean, sd+1e-6 #sd=0 case
        
    def pi(self, state, action):
        mean, sd = self.get_mean_and_sd(state)
        return stats.norm.pdf(x=action, loc=mean, scale=sd)
    
    def sample(self, state):
        mean, sd = self.get_mean_and_sd(state)
        return np.array([stats.norm.rvs(loc=mean, scale=sd)])
    
    def score(self, state, action):
        mean, raw_sd = self.get_mean_and_sd(state, abs_sd=False)
        sd = np.abs(raw_sd)
        feats_vec = self.featuresGen.features(state)
        
        score_1 = ((action - mean)/sd**2) * feats_vec
        
        score_2 = (-1/sd + (action - mean)**2 / sd**3) * np.sign(raw_sd) * feats_vec
        
        return np.vstack((score_1, score_2))
    
    def get_gradient_shape(self):
        return self.theta.shape
    
    def gradient_ascent_update(self, gradient, step_size):
        self.theta = self.theta + step_size * gradient


class LinearSoftMaxPolicy(Policy):
    def __init__(self, a_n, featuresGen):
        Policy.__init__(self, featuresGen)
        self.a_n = a_n
        self.theta = np.random.rand(a_n, self.features_dim)

    def get_action_dist(self, state):
        dot_prods = np.dot(self.theta, self.featuresGen.features(state))
        exps = np.exp(dot_prods - np.max(dot_prods))
        return exps/np.sum(exps)
    
    def get_action_dist_dataset(self, states):
        extended_states = self.featuresGen.features(states)
        dot_prods_dataset = np.dot(extended_states, self.theta.transpose())
        row_maxs = np.amax(dot_prods_dataset, axis=1)
        dot_prods_dataset_nomralized = (dot_prods_dataset.transpose()-row_maxs).transpose()
        exps = np.exp(dot_prods_dataset_nomralized)
        return (exps.transpose() / exps.sum(axis=1)).transpose() 
    
    def pi(self, state, action):
        return self.get_action_dist(state)[action]
    
    def sample(self, state):
        return np.random.choice(self.a_n, p=self.get_action_dist(state))
    
    def score(self, state, action):
        dist = self.get_action_dist(state)
        features = np.zeros([self.a_n, self.features_dim])
        for i in np.arange(self.a_n):
            if action==i:
                factor = (1 - dist[i])
            else:
                factor = (0 - dist[i])
            features[i] = factor * self.featuresGen.features(state)
        return features
    
    def score_datasets(self, states, actions):
        dists = self.get_action_dist_dataset(states)
        factors_matrix = -1 * dists + np.eye(self.a_n)[actions]
        states = self.featuresGen.features(states)
        features_matrix = (factors_matrix[:,0] * states.transpose()).transpose()
        for i in np.arange(1,self.a_n):
            features_matrix = np.hstack((features_matrix, (factors_matrix[:,i] * states.transpose()).transpose()))
        return features_matrix

    def get_gradient_shape(self):
        return self.theta.shape
    
    def gradient_ascent_update(self, gradient, step_size):
        self.theta = self.theta + step_size * gradient