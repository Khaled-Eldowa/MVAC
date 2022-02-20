import numpy as np
import gym
import envs
import features
import policies
import critics
import mvac
import os
import sys
import json

def main():
	kwargs = json.loads(sys.argv[1])

	np.random.seed(kwargs.get("seed"))
	gamma = kwargs.get("gamma")
	lmbda = kwargs.get("lmbda")
	env = gym.make(kwargs.get("env"))
	env.seed(kwargs.get("seed"))

	run_name = os.path.join(kwargs.get("dir"), str(kwargs.get("seed")))

	cr_feats = features.RBF_1D_FeaturesGen(n_means=kwargs.get("cr_n_means"), sd=kwargs.get("cr_sd"),
										   low=env.observation_space.low[0], high=env.observation_space.high[0])
	cr = critics.MiniBatch_TD_V(cr_feats, env, gamma=gamma, lmbda=lmbda)

	plcy_feats = features.RBF_1D_FeaturesGen(n_means=kwargs.get("plcy_n_means"), sd=kwargs.get("plcy_sd"), 
										 	 low=env.observation_space.low[0], high=env.observation_space.high[0])
	plcy = policies.LinearGaussian1DPolicy(plcy_feats)

	act = mvac.MV_Minibatch_Actor_rms(env, plcy, gamma, lmbda, cr)
	logs = act.learn(**kwargs)

	np.savez(run_name + ".npz", mv=logs["mv_trace"], v=logs["vol_trace"],
                         j=(1-gamma)*logs["j_trace"], var=logs["var_trace"])

if __name__ == "__main__":
	main()