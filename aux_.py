import numpy as np

def sample(env, policy, n_batches, n_steps_per_batch, diff_trajectories=True, 
           restart_when_done=True, discounted_sampling=False, gamma=None, start_state=None):
    data = []
    state = env.reset()
    
    if (start_state is not None):
        env = env.unwrapped
        env.state = np.array(start_state)
        state = np.array(start_state)
    
    for b in range(n_batches):
        s = []
        a = []
        r = []
        next_s = []
        
        for t in range(n_steps_per_batch):
            action = policy.sample(state)
            next_state, reward, done, info = env.step(action)
            
            s.append(state)
            a.append(action)
            r.append(reward)
            next_s.append(next_state)
            
            if (done and restart_when_done):
                state = env.reset()
                if (diff_trajectories):
                    break
                #print("Trajectory Reset!")
            elif (discounted_sampling and (np.random.rand() > gamma)):
                state = env.reset()
            else:
                state = next_state
                
        data.append({'s':np.array(s), 'a':np.array(a), 'r':np.array(r), 'next_s':np.array(next_s)})
        
        if (diff_trajectories):
            state = env.reset()
    
    return data

def transform_rewards(reward_vec, lmbda, j_hat):
    reward_vec = np.array(reward_vec)
    return reward_vec - lmbda*(reward_vec - j_hat)**2

def estimate_expected_reward(trajectories, gamma):
    returns = []
    for i in np.arange(len(trajectories)):
        reward_vec = trajectories[i]['r']
        reward_len = len(reward_vec)
        gamma_vec = np.logspace(0, reward_len-1, num=reward_len, base=gamma)
        returns.append(reward_vec.dot(gamma_vec))
    returns = np.array(returns)
    return (1-gamma) * np.mean(returns)

def estimate_return_variance(trajectories, gamma):
    returns = []
    for i in np.arange(len(trajectories)):
        reward_vec = trajectories[i]['r']
        reward_len = len(reward_vec)
        gamma_vec = np.logspace(0, reward_len-1, num=reward_len, base=gamma)
        returns.append(reward_vec.dot(gamma_vec))
    returns = np.array(returns)
    return np.var(returns)   

def estimate_volatility(trajectories, gamma, j_hat):
    returns = []
    for i in np.arange(len(trajectories)):
        reward_vec = (trajectories[i]['r'] - j_hat)**2
        reward_len = len(reward_vec)
        gamma_vec = np.logspace(0, reward_len-1, num=reward_len, base=gamma)
        returns.append(reward_vec.dot(gamma_vec))
    returns = np.array(returns)
    return (1-gamma) * np.mean(returns)    

def sample_and_estimate_expected_reward(env, policy, n_trajc, steps_per_trajc, gamma):
    d_j = sample(env, policy, n_batches=n_trajc, 
                         n_steps_per_batch=steps_per_trajc, diff_trajectories=True)
    return estimate_expected_reward(trajectories=d_j, gamma=gamma)