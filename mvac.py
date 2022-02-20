import numpy as np
from abc import ABC, abstractmethod
from aux_ import *

class MVAC(ABC):
	def __init__(self, env, policy, gamma, lmbda, critic):
		self.env = env
		self.policy = policy        
		self.gamma = gamma        
		self.lmbda = lmbda
		self.critic = critic
		
	def learn(self, **kwargs):
		pass


class MV_Minibatch_Actor_rms(MVAC):
	def __init__(self, env, policy, gamma, lmbda, critic):
		MVAC.__init__(self, env, policy, gamma, lmbda, critic)
		
		
	def learn(self, **kwargs):
		print("----------------------------------MVAC: Minibatch Actor Version----------------------------------")
		actor_step_size = kwargs.get("actor_step_size", 0.0001)
		actor_decay = kwargs.get("actor_decay", 0)
		T = kwargs.get("actor_n_updates", 25)
		B = kwargs.get("actor_batch_size", 30)
		gradient_ms = np.zeros(self.policy.get_gradient_shape())
		
		logs = {}
		j_trace = []
		vol_trace = []
		mv_trace = []
		var_trace = []

		for t in np.arange(T):
			print("-----------Actor iteration : {}-----------".format(t+1))
			j_hat, critic_logs, d_j, _ = self.critic.learn(self.policy, **kwargs)
			vol = estimate_volatility(d_j, self.gamma, j_hat)
			mv = j_hat - self.lmbda * vol
			var = estimate_return_variance(d_j, self.gamma)
			print("Normalized J = {}".format(j_hat))
			print("Reward Volatility = {}".format(vol))
			j_trace.append(j_hat/(1-self.gamma))
			vol_trace.append(vol)
			mv_trace.append(mv)
			var_trace.append(var)
			
			d_a = sample(self.env, self.policy, 1, B, discounted_sampling=True, gamma=self.gamma)
			r = transform_rewards(d_a[0]['r'], self.lmbda, j_hat)
			s = d_a[0]['s']            
			a = d_a[0]['a']            
			next_s = d_a[0]['next_s']
			v_s = self.critic.value_function(s)
			v_s_prime = self.critic.value_function(next_s)
			td_err_vec = r + self.gamma * v_s_prime - v_s
			
			gradient = np.zeros(self.policy.get_gradient_shape())
			for i in np.arange(B):
				gradient = gradient + td_err_vec[i] * self.policy.score(s[i], a[i])
			gradient = gradient / B
			gradient_ms = 0.9 * gradient_ms + 0.1 * gradient**2

			self.policy.gradient_ascent_update(gradient=gradient/(np.sqrt(gradient_ms) + 1e-6), 
											   step_size=(actor_step_size/(1+actor_decay*t)))
			
		n_trajc = kwargs.get("n_trajc", 70)
		steps_per_trajc = kwargs.get("steps_per_trajc", 50)   
		d_j = sample(self.env, self.policy, n_batches=n_trajc, 
						 n_steps_per_batch=steps_per_trajc, diff_trajectories=True)
		final_j = estimate_expected_reward(trajectories=d_j, gamma=self.gamma)
		vol = estimate_volatility(d_j, self.gamma, final_j)
		mv = final_j - self.lmbda * vol
		var = estimate_return_variance(d_j, self.gamma)
		
		j_trace.append(final_j/(1-self.gamma))
		vol_trace.append(vol)
		mv_trace.append(mv)
		var_trace.append(var)

		logs["j_trace"] = np.array(j_trace)
		logs["vol_trace"] = np.array(vol_trace)
		logs["mv_trace"] = np.array(mv_trace)
		logs["var_trace"] = np.array(var_trace)
		return logs