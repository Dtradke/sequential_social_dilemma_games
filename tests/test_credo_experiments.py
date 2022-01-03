import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import os.path
from math import sqrt
import natsort

import numpy as np
import pandas as pd
from scipy.stats import t
from ternary.helpers import simplex_iterator

from utility_funcs import get_all_subdirs

ray_results_path = os.path.expanduser("~/ray_results")
full_exp_folders = get_all_subdirs(ray_results_path)

scale = 5

self_weights = np.linspace(0,1,6)
team_weights = np.linspace(0,1,6)
sys_weights = np.linspace(0,1,6)


# exp_folders = []
# for e in full_exp_folders:
#     exp_folders.append(e.split("/")[-1])

for (i,j,k) in simplex_iterator(scale):
    if i == 5:
        ray_results_path2 = os.path.expanduser("~/ray_results6teams_6agents")
        path = "/h/dtradke/ray_results/cleanup_baseline_PPO_6teams_6agents_custom_metrics_rgb"
    elif j == 5:
        ray_results_path2 = os.path.expanduser("~/ray_results3teams_6agents")
        path = "/h/dtradke/ray_results/cleanup_baseline_PPO_3teams_6agents_custom_metrics_rgb"
    elif k == 5:
        ray_results_path2 = os.path.expanduser("~/ray_results1teams_6agents")
        path = "/h/dtradke/ray_results/cleanup_baseline_PPO_1teams_6agents_custom_metrics_rgb"
    else:
        ray_results_path2 = os.path.expanduser("~/ray_results3teams_6agents")
        path = "/h/dtradke/ray_results3teams_6agents/cleanup_baseline_PPO_3teams_6agents_" + str(i/5) + "-" + str(j/5) + "-" + str(k/5) + "_rgb"

    full_exp_folders2 = get_all_subdirs(ray_results_path2)
    full_exp_folders = full_exp_folders + full_exp_folders2
    
    exp_count = 0
    if path in full_exp_folders:
        exps = get_all_subdirs(path)
        for exp in exps:
            if "BaselinePPOTrainer" in exp.split("/")[-1]:
                exp_count+=1

    print(path.split("/")[-1], ": ", exp_count)
