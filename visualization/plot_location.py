import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from math import sqrt
import natsort

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import ternary
from ternary.helpers import simplex_iterator
import numpy as np
import pandas as pd
import scipy
import copy
import sys



ray_results_path = os.path.expanduser("~/ray_results")
plot_path = os.path.expanduser("~/ray_results_plot")

CLEANUP_MAP = np.zeros((25,18))

np.set_printoptions(threshold=sys.maxsize)


def getExperimentParameters(path):
    ''' returns the number of teams and agents in the 'path' experiment '''
    path_arr = path.split("_")

    for word in path_arr:
        if word[1:] == "teams":
            nteams = int(word[0])
        elif word[1:] == "agents":
            nagents = int(word[0])
    return nteams, nagents

def getAgentData(exp_dir, agent):
    ''' loads agent location dataframe '''

    files = os.listdir(exp_dir+"/agent_values/agent-"+str(agent)+"/")

    if 'episode_list.npy' in files:
        files = files.remove("episode_list.npy")

    files = natsort.natsorted(files)

    file_to_load = str(files[-1])
    df = pd.read_csv(exp_dir+"/agent_values/agent-"+str(agent)+"/"+file_to_load)
    return df.loc[:, ~df.columns.str.contains('^Unnamed')], file_to_load[:-4]

def populateValueMap(df, metric):
    ''' populates the map with values according to the defined metric from an experiment '''

    np_env = copy.deepcopy(CLEANUP_MAP)

    env = []
    for i in range(CLEANUP_MAP.shape[0]):
        row = []
        for j in range(CLEANUP_MAP.shape[1]):
            row.append([])
        env.append(row)

    if metric == 'location':
        for index, row in df.iterrows():
            env[int(row['x'])][int(row['y'])].append(1)

        for i in range(CLEANUP_MAP.shape[0]):
            for j in range(CLEANUP_MAP.shape[1]):
                if len(env[i][j]) > 0:
                    np_env[i,j] = np.sum(np.array(env[i][j]))
    else:
        for index, row in df.iterrows():
            env[int(row['x'])][int(row['y'])].append(row[metric])

        for i in range(CLEANUP_MAP.shape[0]):
            for j in range(CLEANUP_MAP.shape[1]):
                if len(env[i][j]) > 0:
                    np_env[i,j] = np.mean(np.array(env[i][j]))
    return np_env

def currentDirectory(exp_dir, trial_num):
    nteams, nagents = getExperimentParameters(exp_dir)
    fname = "../results/cleanup/baseline_PPO_"+str(nteams)+"teams_"+str(nagents)+"agents_rgb/tmp/trial-"+str(trial_num)+"/agent_values/"
    return fname

def saveFinalMap(final_map, exp_dir, agent, trial_num, metric, episode_num):
    ''' saves map of the defined metric as numpy array to be plotted later '''

    fname = currentDirectory(exp_dir, trial_num)
    fname = fname+str(episode_num)+"/"

    if not os.path.exists(fname):
        os.makedirs(fname)
    np.save(fname+"agent-"+str(agent)+metric+".npy", final_map)
    print("saved: ", fname+"agent-"+str(agent)+metric+".npy")

def loadNpyEnvs(exp_dir, trial_num, agent, metric):
    fname = currentDirectory(exp_dir, trial_num)

    episodes = os.listdir(fname)
    files = natsort.natsorted(files)
    # epi_to_load = str(files[-1])

    envs = []
    for f in files:
        envs.append(np.load(fname+'/'+f+'/agent-'+agent+metric+".npy"))



def inspectLocationData(category_folders):
    ''' main function to loop through metrics/agents and plot their maps '''

    metrics = ['vf_preds', 'rewards', 'value_targets', 'location']

    for cat_count, category_folder in enumerate(category_folders):
        cat_folders = get_all_subdirs(category_folder)
        experiment_folders = natsort.natsorted(cat_folders)

        for trial_num, exp_dir in enumerate(experiment_folders):
            num_agents = len(os.listdir(exp_dir+"/agent_values/"))
            for agent in range(num_agents):

                df, episode_num = getAgentData(exp_dir, agent)
                for metric in metrics:
                    final_map = populateValueMap(df, metric)
                    saveFinalMap(final_map, exp_dir, agent, trial_num, metric, episode_num)


def plotLocations(envs):

    for env in envs:
        env = np.flipud(env)
        c = plt.imshow(env, cmap='Reds')
        plt.colorbar(c)
        plt.show()
        plt.close()
    exit()


def plotLocationData(category_folders):
    # metrics = ['vf_preds', 'rewards', 'value_targets']
    metrics = ['location']

    for cat_count, category_folder in enumerate(category_folders):
        nteams, nagents = getExperimentParameters(category_folder)
        fname = "../results/cleanup/baseline_PPO_"+str(nteams)+"teams_"+str(nagents)+"agents_rgb/tmp/"
        num_trials = len(os.listdir(fname))
        for trial_num in range(num_trials):
            episodes = os.listdir(fname+"trial-"+str(trial_num)+"/agent_values/")
            for metric in metrics:
                envs = []
                for epi in episodes:
                    for agent in range(nagents):
                        envs.append(np.load(fname+"trial-"+str(trial_num)+"/agent_values/"+str(epi)+"/agent-"+str(agent)+metric+'.npy'))
                        
                plotLocations(envs)
                    




if __name__ == "__main__":

    nteam_arr = [1]
    nagent_arr = [2]

    category_folders = []
    for nteam in nteam_arr:
        for nagent in nagent_arr:
            # category_folders.append("/scratch/ssd004/scratch/dtradke/ray_results/cleanup_baseline_PPO_"+str(nteam)+"teams_"+str(nagent)+"agents_custom_metrics_rgb")
            category_folders.append("../../ray_results"+str(nteam)+"teams_"+str(nagent)+"agents_copy/cleanup_baseline_PPO_"+str(nteam)+"teams_"+str(nagent)+"agents_custom_metrics_rgb")

    try:
        from utility_funcs import get_all_subdirs
        inspectLocationData(category_folders)
    except:
        plotLocationData(category_folders)
