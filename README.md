# Mean-Volatility Actor-Critic

This repository contains the implementation of the MVAC algoorithm analyzed in the paper "Finite Sample Analysis of Mean-Volatility Actor-Critic for Risk-Averse Reinforcement Learning", along with instructions on how to run it on the PointReacher environment.

## Prerequisites

Create a virtual Conda environment:

```
conda create -n mvac_env python=3.7.11
```

Activate the environment:

```
conda activate mvac_env
```

Navigate to the main directory and run:

```
pip install -r requirements.txt
```

## Running an Experiment

The file kwargs.json contains the hyperparameters of the algorithm as described in the paper. The argument "runs" specifies the number of indepentent runs (executed in parallel) of the experiment starting from the seed specified by the argument "seed" and incremented by 1 for each run.
To run the experiment, execute the following command:
```
python mvac__exp.py kwargs.json
```

After executing the code, a folder is created inside the "exps" directory for the chosen value of lambda, inside of which a log file is stored for each run and named after its seed value.
The log for each run is a numpy .npz file containing the (estimated) expected return, reward-volatility, mean-volatility, and return variance for each policy encountered in that run. The "exps" folder currently includes the logs of the experiments used to create the plots in the paper.

## Visualizing the Results

A python notebook "plot_results.ipynb" is provided in the repository. It contains the code used to average the results of the independent runs for each value of lambda and plot the results.
