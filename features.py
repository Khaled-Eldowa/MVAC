import numpy as np
from abc import ABC, abstractmethod

#abstract class for feature generators
class FeaturesGen(ABC):
	def __init__(self, dim):
		self.dim = dim #number of dimensions
	def get_dim(self):
		return self.dim
	@abstractmethod
	def features(self, states):
		pass

class PolicyNetworkFeatures(FeaturesGen):
    def __init__(self, policy):
        FeaturesGen.__init__(self, policy.hid2)
        self.policy = policy

    def features(self, states):
        return self.policy.get_features(states)

class IdFeatures(FeaturesGen):
    def __init__(self, state_dim, subtract=0, normalize_by=1):
        FeaturesGen.__init__(self, state_dim)
        self.state_dim = state_dim
        self.subtract = subtract
        self.normalize_by = normalize_by

    def features(self, states):
        if(np.isscalar(states)):
            states = [states]
        states = np.array(states)
        states = states.reshape((-1,self.state_dim))
        states = states - self.subtract
        states = states / self.normalize_by
        return np.squeeze(states)

class PolyFeaturesGen(FeaturesGen):
	"""
	Polynomial Features (with one constant component)

	"""
	def __init__(self, degree, normalize_by=1):
		FeaturesGen.__init__(self, degree+1)
		self.degree = degree
		self.normalize_by = normalize_by

	def features(self, states):
	#input: one state or array of states
	#output: matrix of features, one row per input state
		if(np.isscalar(states)):
			states = [states]
		states = np.array(states)
		states = states.reshape((-1,))
		states = states / self.normalize_by
		features = np.zeros([len(states),self.degree+1])
		for i in np.arange(self.degree+1):
			features[:,i] = states**i
		return np.squeeze(features)


class RBF_1D_FeaturesGen(FeaturesGen):
	"""
	Radial Basis Functions Feature Generator
	It takes a 1D interval [low,high] and selects n_means means spread
	uniformly over the interval. The features of a state consist of
	one constant feature and, for each mean, a gaussian rbf function
	of the form e^(-((state-mean)/sd)**2) where sd is an adjustable
	parameter.

	"""
	def __init__(self, n_means, sd, low, high):
		FeaturesGen.__init__(self, n_means+1)
		self.n_means = n_means
		self.sd = sd
		self.low = low
		self.high = high
		self.means = np.linspace(low, high ,num=n_means)

	def features(self, states):
	#input: one state or array of states
	#output: matrix of features, one row per input state	
		if(np.isscalar(states)):
			states = [states]
		states = np.array(states)
		states = states.reshape((-1,))
		features = np.zeros([len(states),self.n_means+1])
		features[:,0] = np.ones(len(states))
		for i in np.arange(1, self.n_means+1):
			features[:,i] = np.exp(-1 * ((states-self.means[i-1]) / self.sd)**2)
		return np.squeeze(features)