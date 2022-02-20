from datetime import datetime
from shutil import copyfile
import os
import sys
import json
import numpy as np
import subprocess
from concurrent.futures import ProcessPoolExecutor

def main():
	with open(sys.argv[1]) as json_kwargs:
		kwargs = json.load(json_kwargs)

	lmbda = kwargs.get("lmbda")
	env_name = kwargs.get("env")
	exp_dir = os.path.join(os.path.join(os.getcwd(), 'exps'), env_name + '__lambda_' + str(lmbda).replace('.','_'))
	if not os.path.exists(exp_dir):
		os.makedirs(exp_dir)
	now = datetime.now().strftime('%b%d_%H-%M-%S')
	copyfile(os.path.join(os.getcwd(), sys.argv[1]), os.path.join(exp_dir, str(now) + "_" + sys.argv[1]))
	kwargs["dir"] = exp_dir	
	runs = kwargs.pop('runs', None)
	
	with ProcessPoolExecutor() as executor:
		for i in np.arange(runs):
			executor.submit(subprocess.run, ["python", "mvac__run.py", json.dumps(kwargs)])
			kwargs["seed"] = kwargs["seed"] + 1


if __name__ == "__main__":
	main()