import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from math import sqrt
import natsort

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import ternary
from ternary.helpers import simplex_iterator
import numpy as np
import pandas as pd
import scipy
import copy
import sys
import imageio



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

def getAgentData(exp_dir, agent, file_to_load):
    ''' loads agent location dataframe '''

    files = os.listdir(exp_dir+"/agent_values/agent-"+str(agent)+"/")
    files = natsort.natsorted(files)

    if 'episode_list.npy' in files:
        files = files[:-1]

    file_to_load = str(files[file_to_load])
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
    fname = "../results/cleanup/baseline_PPO_"+str(nteams)+"teams_"+str(nagents)+"agents_rgb/maps/trial-"+str(trial_num)+"/agent_values/"
    return fname

def saveFinalMap(final_map, exp_dir, agent, trial_num, metric, episode_num):
    ''' saves map of the defined metric as numpy array to be plotted later '''

    fname = currentDirectory(exp_dir, trial_num)
    fname = fname+str(episode_num)+"/npy_arrays/"

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
        envs.append(np.load(fname+'/'+f+'/npy_arrays/agent-'+agent+metric+".npy"))



def inspectLocationData(category_folders):
    ''' main function to loop through metrics/agents and plot their maps '''

    metrics = ['vf_preds', 'rewards', 'value_targets', 'location']

    files_to_load = [0, -1]

    for cat_count, category_folder in enumerate(category_folders):
        cat_folders = get_all_subdirs(category_folder)
        experiment_folders = natsort.natsorted(cat_folders)

        for trial_num, exp_dir in enumerate(experiment_folders):
            num_agents = len(os.listdir(exp_dir+"/agent_values/"))
            for agent in range(num_agents):
                for file_to_load in files_to_load:
                    df, episode_num = getAgentData(exp_dir, agent, file_to_load)
                    for metric in metrics:
                        final_map = populateValueMap(df, metric)
                        saveFinalMap(final_map, exp_dir, agent, trial_num, metric, episode_num)


def plotLocations(envs, nagents, metric, episodes, base_fname):

    if metric == 'rewards':
        all_vmin = -1
        all_vmax = 1
        color_map = 'seismic'
    else:
        all_vmin = -1*np.amax(np.absolute(np.array(envs))) # np.amin(np.array(envs))
        all_vmax = np.amax(np.absolute(np.array(envs))) # np.amax(np.array(envs))
        color_map = 'seismic' # 'Reds'
        


    # im = imageio.imread('../results/cleanup/cleanup_basemap.png')
    im = plt.imread('../results/cleanup/cleanup_basemap.png')

    # plt.figure()
    # # plt.subplot(1,2,1)
    # plt.show()
    # # print(im.shape)
    # exit()

    # stacked_envs = []
    # to_stack = []
    # epi_idx = -1
    # for idx, env in enumerate(envs):
    #     agent_num = idx%nagents
    #     if agent_num == 0:

    #         if len(to_stack) > 0:
    #             to_stack = np.array(to_stack)
    #             stacked_envs.append(np.mean(to_stack, axis=0))
    #             to_stack = []
    #     to_stack.append(env)
    
    # print(len(stacked_envs))
    # exit()

    
    # for idx, env in enumerate(envs):
    # for idx, env in enumerate(stacked_envs):
    epi_idx = 0
    for i in range(0, len(envs), nagents):

        to_stack = np.array(envs[i:i+nagents])
        env = np.mean(to_stack, axis=0)
        env = np.flipud(env)

        plt.figure()
        ax = plt.gca()
        # ax.imshow(im, 'gray', interpolation='none')

        ax.imshow(im, 'gray', interpolation='nearest', extent=[-0.5,17.5,-0.5,24.5])
        # plt.show()
        # exit()
        c = ax.imshow(env, alpha=0.5, vmin=all_vmin, vmax=all_vmax, cmap=color_map)
        # plt.colorbar(c, orientation='horizontal')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.25)
        clb = plt.colorbar(c, cax=cax, orientation="horizontal")

        tick_locator = ticker.MaxNLocator(nbins=4)
        clb.locator = tick_locator
        clb.update_ticks()

        clb.ax.tick_params(labelsize=12) 
        clb.ax.set_xlabel(metric,fontsize=20)
        

        # agent_num = idx%nagents
        # if agent_num == 0: 
        #     epi_idx += 1

        # ax.set_title("Agent: "+str(agent_num)+'\n'+"Epi: "+str(int(episodes[epi_idx])), fontsize=22)
        ax.set_title("Episode: "+str(int(episodes[epi_idx])), fontsize=22)
        ax.text(-5, 14, "River", fontsize=20, rotation=90)
        ax.text(18, 15, "Orchard", fontsize=20, rotation=270)
        plt.show()
        plt.close()

        epi_idx+=1

        # fname = base_fname +episodes[epi_idx]+"/figs/agent-"+str(agent_num)+metric+'.png'
        # plt.savefig(fname+'.png',bbox_inches='tight', dpi=300)
    exit()


def plotLocationData(category_folders):
    # metrics = ['vf_preds', 'rewards', 'value_targets']
    metrics = ['value_targets']

    for cat_count, category_folder in enumerate(category_folders):
        nteams, nagents = getExperimentParameters(category_folder)
        fname = "../results/cleanup/baseline_PPO_"+str(nteams)+"teams_"+str(nagents)+"agents_rgb/maps/"
        num_trials = len(os.listdir(fname))
        for trial_num in range(num_trials):
            episodes = natsort.natsorted(os.listdir(fname+"trial-"+str(trial_num)+"/agent_values/"))
            for metric in metrics:
                base_fname = fname+"trial-"+str(trial_num)+"/agent_values/"
                envs = []
                for epi in episodes:
                    for agent in range(nagents):
                        envs.append(np.load(base_fname+str(epi)+"/npy_arrays/agent-"+str(agent)+metric+'.npy'))
                        
                plotLocations(envs, nagents, metric, episodes, base_fname)
                    




if __name__ == "__main__":

    nteam_arr = [1]
    nagent_arr = [2]

    category_folders = []
    for nteam in nteam_arr:
        for nagent in nagent_arr:
            # category_folders.append("/scratch/ssd004/scratch/dtradke/ray_results/cleanup_baseline_PPO_"+str(nteam)+"teams_"+str(nagent)+"agents_custom_metrics_rgb")
            category_folders.append("../../ray_results"+str(nteam)+"teams_"+str(nagent)+"agents/cleanup_baseline_PPO_"+str(nteam)+"teams_"+str(nagent)+"agents_custom_metrics_rgb")

    try:
        from utility_funcs import get_all_subdirs
        inspectLocationData(category_folders)
        exit()
    except:
        plotLocationData(category_folders)
