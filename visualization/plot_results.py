import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import os.path
from math import sqrt
import natsort

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.ticker import MaxNLocator
import ternary
from ternary.helpers import simplex_iterator
import numpy as np
import pandas as pd
import scipy
from scipy.stats import t
import datashader as ds
import datashader.transfer_functions as tf
import PIL
from PIL import Image

from utility_funcs import get_all_subdirs

# ray_results_path = os.path.expanduser("~/ray_results/trial_2_cleanup")
ray_results_path = os.path.expanduser("~/ray_results")
plot_path = os.path.expanduser("~/ray_results_plot")

files = []

class PlotGraphics(object):
    def __init__(self, column_name, legend_name, color):
        self.column_name = column_name
        self.legend_name = legend_name
        self.color = color


class PlotData(object):
    def __init__(self, x_data, y_data, column_name, legend_name, color):
        self.x_data = x_data
        self.y_data = y_data
        self.plot_graphics = PlotGraphics(column_name, legend_name, color)


def plot_and_save(fn, path, file_name_addition, nteams, nagents, rogue, compare=False, trial=None, credo=False, reward=None):
    global files
    # Clear plot to prevent slowdown when drawing multiple figures
    plt.style.use('seaborn-whitegrid')
    # figure(figsize=(12, 4), dpi=80)
    if "cleaning" in file_name_addition or "apples" in file_name_addition or "loss" in file_name_addition:
        figure(figsize=(4, 2.5), dpi=300)
    elif "diffsizeteams" in file_name_addition:
        figure(figsize=(2.5, 4), dpi=300)
    elif compare:
        figure(figsize=(10, 4), dpi=300)
    else:
        figure(figsize=(4, 2), dpi=300)
    plt.clf()
    fn()
    # Sort legend by label name
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0].lower()))

    if "ginis" in file_name_addition or ("cleaning" in file_name_addition and not credo):
        # plt.legend(handles, labels, fontsize=15, loc='lower left', frameon=True)
        print("no legend")
    elif credo and not compare:
        plt.legend(handles, labels, fontsize=13,ncol=1, loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    else:
        if "apples" in file_name_addition and not compare:
            plt.legend(handles, labels, fontsize=13, loc='lower left', frameon=True)
            plt.title(str(nteams)+"/"+str(int(nagents/nteams)), fontsize=22)
        elif "apples" in file_name_addition and credo:
            credo_str = file_name_addition.split("_")[2].split("-")
            plt.title(r'$\mathbf{cr}_{i} = \langle$ '+str(credo_str[0])+r', '+str(credo_str[1])+r', '+str(credo_str[2])+r'$\rangle$'+'\n'+r'$\overline{R^{\mathbf{cr}}}$'+' = '+str(round(reward,1)), fontsize=22)
        elif ("cleaning" in file_name_addition and credo) or ("Met:" in file_name_addition and credo):
            plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, -0.35), frameon=True)
        elif "diffsizeteams" in file_name_addition:
            plt.legend(handles, labels, fontsize=13, loc='lower left', frameon=True)

    # Strip path of all but last folder
    path_split = os.path.dirname(path).split("/")
    pngpath = plot_path + "/png/" + path_split[-2] + "/"
    if compare:
        if credo:
            pngpath2 = '../results/'+file_name_addition.split("_")[0]+'/team_rewards_rgb/'+file_name_addition.split("_")[2]+"/"
        else:
            pngpath2 = '../results/'+file_name_addition.split("_")[0]+'/team_rewards_rgb/'
    else:
        pngpath2 = '../results/'+file_name_addition.split("_")[0]+'/baseline_PPO_'+str(nteams)+'teams_'+str(nagents)+'agents_rgb/'

    if rogue["flag"]:
        pngpath = pngpath + "rogue/"+str(rogue["nrogue"])+"rogue/deg"+rogue["deg_rogue"]+"/"
        pngpath2 = pngpath2 + "rogue/"+str(rogue["nrogue"])+"rogue/deg"+rogue["deg_rogue"]+"/"

    if trial is not None:
        pngpath = pngpath + "trial-"+str(trial)+"/"
        pngpath2 = pngpath2 + "trial-"+str(trial)+"/"

    pngfile = pngpath + file_name_addition + ".png"
    pngfile2 = pngpath2 + file_name_addition + ".png"

    if not os.path.exists(pngpath2):
        os.makedirs(pngpath2)


    # plt.savefig(pngfile,bbox_inches='tight', dpi=300)
    plt.savefig(pngfile2,bbox_inches='tight', dpi=300)
    # plt.savefig(svgfile,bbox_inches='tight', dpi=100)
    print("SAVED: ", pngfile2)
    if pngfile2 not in files:
        files.append(pngfile2)
    plt.close()
    # exit()


def plot_multiple_category_result(plotdata_list, gini=False):
    for plotdata in plotdata_list:
        plot_single_category_result(
            plotdata.x_data,
            plotdata.y_data,
            plotdata.plot_graphics.color,
            plotdata.plot_graphics.legend_name,
            plotdata.plot_graphics.column_name,
            with_individual_experiments=False,
            with_label=True,
            gini=gini
        )


def plot_single_category_result(
    x_lists,
    y_lists,
    color,
    legend_name,
    y_label_name,
    with_individual_experiments=True,
    with_label=False,
    gini=False,
):
    most_timesteps = np.max(list(map(len, x_lists)))
    x_min = np.nanmin(list(map(np.nanmin, x_lists)))
    x_max = np.nanmax(list(map(np.nanmax, x_lists)))
    y_max = np.nanmax(list(map(np.nanmax, y_lists)))
    interpolated_time = np.linspace(x_min, x_max, most_timesteps)
    interpolated_scores = []
    individual_experiment_label_added = False

    for x, y in zip(x_lists, y_lists):
        interpolated_score = np.interp(interpolated_time, x, y, left=np.nan, right=np.nan)
        interpolated_scores.append(interpolated_score)
        
        light_color = change_color_luminosity(color, 0.5)
        if with_label:
            label_name = legend_name
        elif not individual_experiment_label_added:
            label_name = legend_name + ": Individual experiment"
            individual_experiment_label_added = True
        else:
            label_name = None
        if with_individual_experiments:
            plt.plot(
                interpolated_time, interpolated_score, color=light_color, label=label_name, alpha=0.7
            )

    # Plot the mean and confidence intervals
    # Calculate t-value for p<0.05 CI
    interpolated_scores = np.array(interpolated_scores)
    num_experiments = interpolated_scores.shape[0]
    significance_level = 0.05
    t_value = t.ppf(1 - significance_level / 2, num_experiments - 1)
    sqrt_n = sqrt(num_experiments)
    means = []
    confidence_limits = []

    for std_dev_index in range(interpolated_scores.shape[-1]):
        std_dev = np.std(interpolated_scores[:, std_dev_index], ddof=1)
        mean_confidence_limit = std_dev * t_value / sqrt_n
        confidence_limits.append(mean_confidence_limit)
        mean = np.mean(interpolated_scores[:, std_dev_index])
        means.append(mean)

    lower_confidence_bound = means - np.array(confidence_limits)
    upper_confidence_bound = means + np.array(confidence_limits)

    # lines = ["--","-.",":"]
    # if "1.0" in legend_name.split("-"):
    #     line_sty = "-"
    # else:
    #     line_sty = lines[np.random.randint(0,len(lines))]
    line_sty = "-"

    plt.plot(interpolated_time, means, color=color, linestyle=line_sty, label=legend_name)
    fill_color = change_color_luminosity(color, 0.2)
    plt.fill_between(
        interpolated_time,
        lower_confidence_bound,
        upper_confidence_bound,
        color=fill_color,
        alpha=0.5,
    )

    # plt.xlabel("Environment steps (1e8)", fontsize=24)
    plt.xlabel("Timesteps (1e8)", fontsize=24)
    plt.ylabel(y_label_name, fontsize=24)

    if "loss" not in y_label_name:
        old_bot, old_top = plt.ylim()
        bottom = old_bot
        y_max = max(y_max, old_top)
        if "Equality" in y_label_name:
            bottom=0.0
            y_max=1.05
        elif "Cleaning" in y_label_name:
            y_max=600
            bottom = -1
        elif "Apples" in y_label_name:
            y_max=800
            bottom = -1
        elif "Mean Pop. Reward" == y_label_name or "Reward" in y_label_name:
            y_max = 400 # 350
            bottom = -1
        elif 'entropy' in y_label_name:
            bottom = -0.1
            y_max = 2

        # print(y_max)
        
        # y_maxs = [np.median(np.sort(np.array(yl)[-int(len(yl)*0.75):])[-100:]) for yl in y_lists]
        # y_max = np.amax(y_maxs)
    else:
        if "vf_loss" in y_label_name:
            bottom=0.0
            y_max=100
        elif "policy_loss" in y_label_name:
            bottom=-0.03
            y_max=0.01
        elif "total_loss" in y_label_name:
            bottom=-0.03
            y_max=0.01
        
    plt.ylim(bottom=bottom, top=y_max)

    plt.xlim(left=0, right=1.6)
    plt.ticklabel_format(axis='y',useOffset=False)
    # plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))
    # plt.xticks([0.2,0.4,0.6,0.8,1.0, 1.2, 1.4, 1.6], ["0.2", "0.4", "0.6", "0.8", "1.0", "1.2", "1.4", "1.6"], fontsize=18)
    plt.xticks([0.0, 0.4,0.8, 1.2, 1.6], ["0", "0.4", "0.8", "1.2", "1.6"], fontsize=18)
    plt.yticks(fontsize=18)


def plot_multiple_category_result_fill(plotdata_list, column_name, gini=False):
    # print(len(plotdata_list))
    # print(len(plotdata_list[0][0]))
    # exit()
    # for plotdata in plotdata_list:
    plot_single_category_result_fill(
            # plotdata.x_data,
            plotdata_list,
            column_name,
            with_individual_experiments=False,
            with_label=True,
            gini=gini
        )

def plot_single_category_result_fill(
    xy_lists,
    y_label_name,
    with_individual_experiments=True,
    with_label=False,
    gini=False,
):
    x_lists, y_lists = [], []
    for val in xy_lists:
        x_lists.append(val[0][0])
        y_lists.append(val[1][0])

    most_timesteps = np.max(list(map(len, x_lists)))
    x_min = np.nanmin(list(map(np.nanmin, x_lists)))
    x_max = np.nanmax(list(map(np.nanmax, x_lists)))
    y_max = np.nanmax(list(map(np.nanmax, y_lists)))
    interpolated_time = np.linspace(x_min, x_max, most_timesteps)
    interpolated_scores, interpolated_times = [], []
    individual_experiment_label_added = False
    for x, y in zip(x_lists, y_lists):
        interpolated_score = np.interp(interpolated_time, x, y, left=np.nan, right=np.nan)
        interpolated_scores.append(interpolated_score)
        interpolated_times.append(interpolated_time)
        
        # light_color = change_color_luminosity(color, 0.5)
        # if with_label:
        #     label_name = legend_name
        # elif not individual_experiment_label_added:
        #     label_name = legend_name + ": Individual experiment"
        #     individual_experiment_label_added = True
        # else:
        #     label_name = None
        # if with_individual_experiments:
        #     plt.plot(
        #         interpolated_time, interpolated_score, color=light_color, label=label_name, alpha=0.7
        #     )

    # Plot the mean and confidence intervals
    # Calculate t-value for p<0.05 CI
    interpolated_scores = np.array(interpolated_scores)
    num_experiments = interpolated_scores.shape[0]
    significance_level = 0.05
    t_value = t.ppf(1 - significance_level / 2, num_experiments - 1)
    sqrt_n = sqrt(num_experiments)
    means = []
    confidence_limits = []

    for std_dev_index in range(interpolated_scores.shape[-1]):
        std_dev = np.std(interpolated_scores[:, std_dev_index], ddof=1)
        mean_confidence_limit = std_dev * t_value / sqrt_n
        confidence_limits.append(mean_confidence_limit)
        mean = np.mean(interpolated_scores[:, std_dev_index])
        means.append(mean)

    bottom = 0 #if "reward" in y_label_name.lower() else None
    old_bot, old_top = plt.ylim()
    y_max = max(y_max, old_top)
    if "Equality" in y_label_name:
        bottom=0.0
        y_max=1.05
    elif "Cleaning" in y_label_name:
        y_max=600
    elif "Apples" in y_label_name:
        y_max=800
    elif "Mean Pop. Reward" == y_label_name:
        y_max = 400 # 350


    interpolated_times = np.array(interpolated_times)

    interpolated_times = np.c_[interpolated_times, np.zeros(interpolated_times.shape[0])*np.nan]
    interpolated_scores = np.c_[interpolated_scores, np.zeros(interpolated_scores.shape[0])*np.nan]

    # print(interpolated_scores.size)
    # print(np.nanmax(interpolated_scores))
    # exit()

    d = {'x': np.array(interpolated_times).flatten(),
            'y': interpolated_scores.flatten()}
    df = pd.DataFrame(data=d, index=np.tile(np.arange(interpolated_scores.shape[1]), interpolated_scores.shape[0])) #np.tile(df.index.values, interpolated_scores.shape[0])

    x_range = (df["x"].min(), df["x"].max())
    y_range = (df["y"].min(), df["y"].max())
    w = most_timesteps
    h = int(round(np.nanmax(interpolated_scores),0))
    # dpi=150
    cvs = ds.Canvas(x_range=x_range, y_range=y_range, plot_height=h, plot_width=w)


    aggs = cvs.line(df, 'x', 'y', ds.count())

    heatmap_img = tf.Image(tf.shade(aggs, cmap=plt.cm.Spectral_r))
    # plt.imshow(heatmap_img.to_pil())

    # plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    # ax1 = f1.add_subplot(111)
    heatmap_img = heatmap_img.to_pil()
    plt.imshow(heatmap_img.transpose(PIL.Image.FLIP_TOP_BOTTOM))
    # ax1.grid(False)
    # plt.savefig("../results/credo/heatmap.png", bbox_inches="tight", dpi=dpi)
    # print("saved")
    # exit()


    # # lower_confidence_bound = means - np.array(confidence_limits)
    # # upper_confidence_bound = means + np.array(confidence_limits)

    # # lines = ["--","-.",":"]
    # # if "1.0" in legend_name.split("-"):
    # #     line_sty = "-"
    # # else:
    # #     line_sty = lines[np.random.randint(0,len(lines))]
    # line_sty = "-"

    # plt.plot(interpolated_time, means, color='b', linestyle="", label=legend_name)
    # fill_color = change_color_luminosity('b', 0.2)
    # # plt.fill_between(
    # #     interpolated_time,
    # #     lower_confidence_bound,
    # #     upper_confidence_bound,
    # #     color=fill_color,
    # #     alpha=0.5,
    # # )
    # plt.fill_between(
    #     interpolated_time,
    #     means,
    #     np.zeros(interpolated_time.shape),
    #     color=fill_color,
    #     alpha=0.5,
    # )

    # plt.xlabel("Environment steps (1e8)", fontsize=24)
    plt.xlabel("Timesteps (1e8)", fontsize=24)
    plt.ylabel(y_label_name, fontsize=24)
    

    plt.ylim(bottom=bottom, top=y_max)
    # plt.xlim(left=0, right=1.6)
    plt.ticklabel_format(axis='y',useOffset=False)
    # plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))
    # plt.xticks([0.2,0.4,0.6,0.8,1.0, 1.2, 1.4, 1.6], ["0.2", "0.4", "0.6", "0.8", "1.0", "1.2", "1.4", "1.6"], fontsize=18)
    # plt.xticks([0.0, 0.4,0.8, 1.2, 1.6], ["0", "0.4", "0.8", "1.2", "1.6"], fontsize=18)
    plt.xticks(np.linspace(0,w,5), ["0", "0.4", "0.8", "1.2", "1.6"], fontsize=18)
    plt.yticks(fontsize=18)

def extract_stats(dfs, requested_keys):

    column_names = [df.columns.values.tolist() for df in dfs]
    column_names = [item for sublist in column_names for item in sublist]
    available_keys = [name.split("/")[-1] for name in column_names]
    unique_keys = set()
    [unique_keys.add(key) for key in available_keys]
    available_keys = list(unique_keys)
    keys = [key for key in requested_keys if key in available_keys]

    all_df_lists = {}
    for key in keys:
        all_df_lists[key] = []


    # Per file, extract the mean trajectory for each key.
    # The mean is taken over all distinct agents, per metric,
    # to create a mean value per metric.
    for df in dfs:
        df_list = {}
        for key in keys:
            key_column_names = [name for name in column_names if key == name.split("/")[-1]]
            key_columns = df[key_column_names]
            mean_trajectory = list(key_columns.mean(axis=1))
            df_list[key] = mean_trajectory

        for key, value in df_list.items():
            all_df_lists[key].append(value)
    
    return all_df_lists


# Plot the results for a given generated progress.csv file, found in your ray_results folder.
def plot_csvs_results(paths, nteams, nagents, rogue):
    path = paths[0]
    env, model_name = get_env_and_model_name_from_path(path)

    dfs = []
    for path in paths:
        df = pd.read_csv(path, sep=",")
        # Set NaN values to 0, common at start of training due to ray behavior
        df = df.fillna(0)
        dfs.append(df)

    plots = []

    # Convert environment steps to 1e8 representation
    timesteps_totals = [df.timesteps_total for df in dfs]
    timesteps_totals = [
        [timestep / 1e8 for timestep in timesteps_total] for timesteps_total in timesteps_totals
    ]

    reward_color = get_color_from_model_name(model_name)
    reward_means = [df.episode_reward_mean for df in dfs]
    # plots.append(
    #     PlotData(
    #         timesteps_totals, reward_means, "Reward", "Mean collective episode reward", reward_color
    #     )
    # )

    episode_len_means = [df.episode_len_mean for df in dfs]
    # plots.append(
    #     PlotData(
    #         timesteps_totals, episode_len_means, "episode_length", "Mean episode length", "pink",
    #     )
    # )

    metric_details = [
        PlotGraphics("cur_lr", "Learning rate", "purple"),
        PlotGraphics("policy_entropy", "Policy Entropy", "b"),
        PlotGraphics("policy_loss", "Policy loss", "r"),
        PlotGraphics("vf_loss", "Value function loss", "orange"),
        PlotGraphics("total_a3c_loss", "Total A3C loss", "blue"),
        PlotGraphics("total_loss", "Total loss", "blue"),
        PlotGraphics("moa_loss", "MOA loss", "black"),
        PlotGraphics("scm_loss", "SCM loss", "black"),
        PlotGraphics("social_influence_reward", "MOA reward", "black"),
        PlotGraphics("social_curiosity_reward", "Curiosity reward", "black"),
        PlotGraphics("cur_influence_reward_weight", "Influence reward weight", "orange"),
        PlotGraphics("cur_curiosity_reward_weight", "Curiosity reward weight", "orange"),
        PlotGraphics("extrinsic_reward", "Extrinsic reward", "g"),
        PlotGraphics("total_successes_mean", "Total successes", "black"),
        # Switch environment metrics
        PlotGraphics("switches_on_at_termination_mean", "Switches on at termination", "black"),
        PlotGraphics("total_pulled_on_mean", "Total switched on", "black"),
        PlotGraphics("total_pulled_off_mean", "Total switched off", "black"),
        PlotGraphics("timestep_first_switch_pull_mean", "Time at first switch pull", "black"),
        PlotGraphics("timestep_last_switch_pull_mean", "Time at last switch pull", "black"),
    ]


    extracted_data = extract_stats(dfs, [detail.column_name for detail in metric_details])

    for metric in metric_details:
        if metric.column_name in extracted_data:
            plots.append(
                PlotData(
                    timesteps_totals,
                    extracted_data[metric.column_name],
                    metric.column_name,
                    metric.legend_name,
                    metric.color,
                )
            )

    for plot in plots:

        def plot_fn():
            plot_single_category_result(
                plot.x_data,
                plot.y_data,
                plot.plot_graphics.color,
                plot.plot_graphics.legend_name,
                plot.plot_graphics.column_name,
            )

        try:
            plot_and_save(
                plot_fn, path, env + "_" + plot.plot_graphics.column_name + "_" + str(nteams) + "teams_" + str(nagents) + "agents", nteams, nagents, rogue
            )
        except ZeroDivisionError:
            pass

# Plot the results for a given generated progress.csv file, found in your ray_results folder.
def collect_losses(paths, nteams, nagents, rogue):
    path = paths[0]
    env, model_name = get_env_and_model_name_from_path(path)

    dfs = []
    for path in paths:
        df = pd.read_csv(path, sep=",")
        # Set NaN values to 0, common at start of training due to ray behavior
        df = df.fillna(0)
        dfs.append(df)

    plots = []

    # Convert environment steps to 1e8 representation
    timesteps_totals = [df.timesteps_total for df in dfs]
    timesteps_totals = [
        [timestep / 1e8 for timestep in timesteps_total] for timesteps_total in timesteps_totals
    ]

    reward_color = get_color_from_model_name(model_name)
    reward_means = [df.episode_reward_mean for df in dfs]
    episode_len_means = [df.episode_len_mean for df in dfs]

    metric_details = [
        PlotGraphics("cur_lr", "Learning rate", "purple"),
        PlotGraphics("policy_entropy", "Policy Entropy", "b"),
        PlotGraphics("policy_loss", "Policy loss", "r"),
        PlotGraphics("vf_loss", "Value function loss", "orange"),
        PlotGraphics("total_a3c_loss", "Total A3C loss", "blue"),
        PlotGraphics("total_loss", "Total loss", "blue"),
        PlotGraphics("moa_loss", "MOA loss", "black"),
        PlotGraphics("scm_loss", "SCM loss", "black"),
        PlotGraphics("social_influence_reward", "MOA reward", "black"),
        PlotGraphics("social_curiosity_reward", "Curiosity reward", "black"),
        PlotGraphics("cur_influence_reward_weight", "Influence reward weight", "orange"),
        PlotGraphics("cur_curiosity_reward_weight", "Curiosity reward weight", "orange"),
        PlotGraphics("extrinsic_reward", "Extrinsic reward", "g"),
        PlotGraphics("total_successes_mean", "Total successes", "black"),
        # Switch environment metrics
        PlotGraphics("switches_on_at_termination_mean", "Switches on at termination", "black"),
        PlotGraphics("total_pulled_on_mean", "Total switched on", "black"),
        PlotGraphics("total_pulled_off_mean", "Total switched off", "black"),
        PlotGraphics("timestep_first_switch_pull_mean", "Time at first switch pull", "black"),
        PlotGraphics("timestep_last_switch_pull_mean", "Time at last switch pull", "black"),
    ]


    extracted_data = extract_stats(dfs, [detail.column_name for detail in metric_details])

    loss_dict = {}
    for key in extracted_data.keys():
        if key not in loss_dict.keys():
            loss_dict[key] = []
        for data_arr in extracted_data[key]:
            loss_dict[key].append(np.mean(np.array(data_arr)))

    for key in loss_dict.keys():
        loss_dict[key] = np.mean(np.array(loss_dict[key]))
    return loss_dict

def getMetricNames(path):
    cols = []
    df = pd.read_csv(path, sep=",", error_bad_lines=False)
    df = df.fillna(0)
    for col_name in df.columns:
        if "info/learner/agent-" in col_name and col_name.split("/")[-1] not in cols:
            cols.append(col_name.split("/")[-1])
    return cols

def getPathForColumnNames(category_folder):
    col_names = None
    path = category_folder.split("/")[-2]
    nteams, nagents, rogue = getExperimentParameters(path)
    csv_path = category_folder + "/progress.csv"
    if os.path.getsize(csv_path) > 0:
        col_names = getMetricNames(csv_path)
    assert col_names is not None, "Size of the progress.csv file was 0"
    return col_names

def plotTrialLearningMetrics(learning_metric, trial, category_folder, reward, credo):
    scale=5
    
    d = dict()
    env_rewards = {}
    rogue = {"flag": False}
    # for category_folder in category_folders: #get_all_subdirs(ray_results_path):
    #     if category_folder.split("/")[-1] in scenarios:
    path = category_folder.split("/")[-2]

    nteams, nagents, rogue = getExperimentParameters(path)
    graph_label = str(nteams)+"/"+str(int(nagents/nteams))
    collective = False

    graph_label = getCredoVals(path, nteams)
    i = int(float(graph_label.split("-")[0])*scale)
    j = int(float(graph_label.split("-")[1])*scale)
    k = int(float(graph_label.split("-")[2])*scale)

    csvs = []
    # experiment_folders = get_all_subdirs(category_folder)
    # # grabs one trial
    # experiment_folders = natsort.natsorted(experiment_folders)
    experiment_folders = [category_folder] #[experiment_folders[trial]]
    
    for experiment_folder in experiment_folders:
        csv_path = experiment_folder + "/progress.csv"
        if os.path.getsize(csv_path) > 0:
            csvs.append(csv_path)


    for agent_num in range(nagents):
        agent_loss, env = get_learning_metric(csvs, nagents, nteams, agent_num, learning_metric, credo=credo) #(csvs, nagents, nteams, agent_num, moi)
        # agent_metric, env = get_custom_metrics(csvs, nagents, nteams, agent_num, metric_type, credo=True) #(csvs, nagents, nteams, agent_num, moi)
        if env not in env_rewards:
            env_rewards[env] = []
        env_rewards[env].append(agent_loss)

    for env, agent_loss in env_rewards.items():
        def plot_fn():
            plot_multiple_category_result(agent_loss)

        # Add filler to path which will be removed
        collective_env_path = "collective_results/filler/"
        # plot_and_save(plot_fn, collective_env_path, env+"_gini_specialization_"+str(nteams)+"teams_"+str(nagents)+"agents", nteams, nagents, rogue)
        if credo:
            plot_and_save(plot_fn, collective_env_path, "credo_" + env+ "_"+graph_label+"_"+learning_metric+"_"+str(nagents)+"agents", nteams, nagents, rogue, compare=True, trial=trial, credo=credo, reward=reward)
        else:
            plot_and_save(plot_fn, collective_env_path, env+"_"+str(nteams)+"teams_"+str(nagents)+"agents_"+learning_metric, nteams, nagents, rogue, trial=trial)




def get_color_from_num_teams(nteams):
    name_to_color = {
        1: "blue",
        2: "red",
        3: "green",
        6: "orange",
        7: "cyan"
    }
    # name_lower = model_name.lower()
    team_label_str = nteams
    if team_label_str in name_to_color.keys():
        return name_to_color[team_label_str]
    else:
        default_color = "darkgreen"
        print(
            "Warning: model name "
            + model_name
            + " has no default plotting color. Defaulting to "
            + default_color
        )
        return default_color

def get_color_from_agent_num(agent_num, nteams):
    name_to_color = {
        "agent-0": {1:"navy",       2:"navy",           3:"navy",               6:"blue"},
        "agent-1": {1:"blue",       2:"dodgerblue",     3:"deepskyblue",        6:"red"},
        "agent-2": {1:"dodgerblue", 2:"deepskyblue",    3:"darkred",            6:"orange"},
        "agent-3": {1:"deepskyblue", 2:"darkred",       3:"lightcoral",             6:"green"},
        "agent-4": {1:"teal",       2:"red",            3:"green",              6:"magenta"},
        "agent-5": {1:"cyan",       2:"lightcoral",     3:"lime",  6:"cyan"},
        "agent-6": {1:"aquamarine", 2:"", 3:"", 6:"grey"},
    }
    # name_lower = model_name.lower()
    agent_label_str = "agent-"+str(agent_num)
    if agent_label_str in name_to_color.keys():
        return name_to_color[agent_label_str][nteams]
    else:
        default_color = "darkgreen"
        print(
            "Warning: model name "
            + model_name
            + " has no default plotting color. Defaulting to "
            + default_color
        )
        return default_color

def get_color_from_agent_num_single_team(agent_num, nteams):
    name_to_color = {
        "agent-0": {1:"navy",       2:"navy",           3:"navy",               6:"blue"},
        "agent-1": {1:"navy",       2:"dodgerblue",     3:"deepskyblue",        6:"red"},
        "agent-2": {1:"red", 2:"deepskyblue",    3:"darkred",            6:"orange"},
        "agent-3": {1:"magenta", 2:"darkred",       3:"lightcoral",             6:"green"},
        "agent-4": {1:"teal",       2:"red",            3:"green",              6:"magenta"},
        "agent-5": {1:"purple",       2:"lightcoral",     3:"lime",  6:"cyan"},
        "agent-6": {1:"aquamarine", 2:"", 3:"", 6:"grey"},
    }
    # name_lower = model_name.lower()
    agent_label_str = "agent-"+str(agent_num)
    if agent_label_str in name_to_color.keys():
        return name_to_color[agent_label_str][nteams]
    else:
        default_color = "darkgreen"
        print(
            "Warning: model name "
            + model_name
            + " has no default plotting color. Defaulting to "
            + default_color
        )
        return default_color

def get_color_from_team_label(team_label):
    name_to_color = {
        "team-0": "blue",
        "team-1": "red",
        "team-2": "green",
        "team-3": "orange",
        "team-4": "magenta",
        "team-5": "cyan",
        "team-6": "grey",
    }
    # name_lower = model_name.lower()
    team_label_str = "team-"+str(team_label)
    if team_label_str in name_to_color.keys():
        return name_to_color[team_label_str]
    else:
        default_color = "darkgreen"
        print(
            "Warning: model name "
            + model_name
            + " has no default plotting color. Defaulting to "
            + default_color
        )
        return default_color

def get_color_from_model_name(model_name):
    name_to_color = {
        "baseline": "blue",
        "moa": "red",
        "scm": "orange",
        "scm no influence reward": "green",
    }
    name_lower = model_name.lower()
    if name_lower in name_to_color.keys():
        return name_to_color[name_lower]
    else:
        default_color = "darkgreen"
        print(
            "Warning: model name "
            + model_name
            + " has no default plotting color. Defaulting to "
            + default_color
        )
        return default_color


def get_env_and_model_name_from_path(path):
    category_path = path.split("/")[-3]
    if "baseline" in category_path:
        model_name = "baseline"
    elif "moa" in category_path:
        model_name = "MOA"
    elif "scm" in category_path:
        if "no_influence" in category_path:
            model_name = "SCM no influence reward"
        else:
            model_name = "SCM"
    else:
        raise NotImplementedError
    env = category_path.split("_")[0]
    return env, model_name

def get_team_dict(nagents, nteams):
    teams = {}
    for idx in range(nteams):
        teams[idx] = []

    agents_per_team = nagents // nteams
    agents_on_team = 0
    team_num_count = 0
    for i in range(nagents):
        if agents_on_team < agents_per_team:
            agents_on_team += 1
        else:
            agents_on_team = 1
            team_num_count += 1
        teams[team_num_count].append(i)
    return teams


def get_team_rewards(paths, nagents, nteams, team_count):
    team_dict = get_team_dict(nagents, nteams)

    dfs = []
    for path in paths:
        # try:
        df = pd.read_csv(path, sep=",", error_bad_lines=False)
        # except:
            # print(path)
            # exit()
        # Set NaN values to 0, common at start of training due to ray behavior
        df = df.fillna(0)
        dfs.append(df)

    env, model_name = get_env_and_model_name_from_path(paths[0])
    color = get_color_from_team_label(team_count)

    # Convert environment steps to 1e8 representation
    timesteps_totals = [df.timesteps_total for df in dfs]
    timesteps_totals = [
        [timestep / 1e8 for timestep in timesteps_total] for timesteps_total in timesteps_totals
    ]

    most_timesteps = np.max(list(map(len, timesteps_totals)))
    x_min = np.nanmin(list(map(np.nanmin, timesteps_totals)))
    x_max = np.nanmax(list(map(np.nanmax, timesteps_totals)))
    # Cut off plotting at 5e8 steps
    x_max = min(x_max, 5.0)
    interp_x = np.linspace(x_min, x_max, most_timesteps)
    interpolated = []
    # print(list(dfs[0].columns))
    # exit()
    reward_arr = []
    for df in dfs:
        for agent_num in team_dict[team_count]:
            reward_arr.append(list(df["policy_reward_mean/agent-"+str(agent_num)]))
    

    # changed for gini
    # lengths = [np.size(n) for n in np.array(timesteps_totals)]
    # timesteps_totals = [timesteps_totals[np.argmax([np.size(n) for n in np.array(timesteps_totals)])]]

    # for x, y, in zip(timesteps_totals, [df.episode_reward_mean for df in dfs]):
    for x, y, in zip(timesteps_totals, reward_arr):                     # NOTE: this used to be [reward_arr] but was getting "object too deep for desired array"
        interp_y = np.interp(interp_x, x, y, left=np.nan, right=np.nan)
        interpolated.append(interp_y)

    reward_plotdata = PlotData(
        [interp_x] * 5, interpolated, "Team Reward", "team-"+str(team_count), color,
    )
    return reward_plotdata, env, interpolated, interp_x



def get_team_rewards_gini(paths, nagents, nteams, team_count):
    team_dict = get_team_dict(nagents, nteams)

    dfs = []
    for path in paths:
        # try:
        df = pd.read_csv(path, sep=",", error_bad_lines=False)
        # except:
            # print(path)
            # exit()
        # Set NaN values to 0, common at start of training due to ray behavior
        df = df.fillna(0)
        dfs.append(df)

    env, model_name = get_env_and_model_name_from_path(paths[0])
    color = get_color_from_team_label(team_count)

    # Convert environment steps to 1e8 representation
    timesteps_totals = [df.timesteps_total for df in dfs]
    timesteps_totals = [
        [timestep / 1e8 for timestep in timesteps_total] for timesteps_total in timesteps_totals
    ]


    most_timesteps = np.max(list(map(len, timesteps_totals)))
    x_min = np.nanmin(list(map(np.nanmin, timesteps_totals)))
    x_max = np.nanmax(list(map(np.nanmax, timesteps_totals)))
    # Cut off plotting at 5e8 steps
    x_max = min(x_max, 5.0)
    interp_x = np.linspace(x_min, x_max, most_timesteps)
    interpolated = []
    # print(list(dfs[0].columns))
    # exit()
    df_rewards = []
    for df in dfs:
        reward_arr = []
        for agent_num in team_dict[team_count]:
            reward_arr.append(df["policy_reward_mean/agent-"+str(agent_num)])
        reward_arr = np.mean(np.array(reward_arr), axis=0)
        df_rewards.append(reward_arr)

    # print(reward_arr)
    # exit()
    # reward_arr = np.mean(np.array(reward_arr), axis=0)


    # for x, y, in zip(timesteps_totals, [df.episode_reward_mean for df in dfs]):
    for x, y, in zip(timesteps_totals, df_rewards):
        interp_y = np.interp(interp_x, x, y, left=np.nan, right=np.nan)
        interpolated.append(interp_y)

    reward_plotdata = PlotData(
        [interp_x] * 5, interpolated, "Team Reward", "team-"+str(team_count), color,
    )
    return reward_plotdata, env, interpolated, interp_x, timesteps_totals

def get_metrics_arr_agg(paths, agent_num, moi, nteams):
    metrics_arr_agg = []
    for metric_type in moi:

        if metric_type == "cleaning":
            metric_column_name = "custom_metrics/cleaning_beam_agent-"
        elif metric_type == "apples":
            metric_column_name = "custom_metrics/apples_agent-"

        dfs = []
        for path in paths:
            # try:
            df = pd.read_csv(path, sep=",", error_bad_lines=False)
            # except:
                # print(path)
                # exit()
            # Set NaN values to 0, common at start of training due to ray behavior
            df = df.fillna(0)
            dfs.append(df)

        # for col in list(dfs[0].columns):
        #     print(col)
        # exit()

        env, model_name = get_env_and_model_name_from_path(paths[0])
        color = get_color_from_agent_num(agent_num, nteams)

        # Convert environment steps to 1e8 representation
        timesteps_totals = [df.timesteps_total for df in dfs]
        timesteps_totals = [
            [timestep / 1e8 for timestep in timesteps_total] for timesteps_total in timesteps_totals
        ]

        most_timesteps = np.max(list(map(len, timesteps_totals)))
        x_min = np.nanmin(list(map(np.nanmin, timesteps_totals)))
        x_max = np.nanmax(list(map(np.nanmax, timesteps_totals)))
        # Cut off plotting at 5e8 steps
        x_max = min(x_max, 5.0)
        interp_x = np.linspace(x_min, x_max, most_timesteps)
        
        # print(list(dfs[0].columns))
        # exit()
        metrics_arr = []
        for df in dfs:
            metrics_arr.append(df[metric_column_name+str(agent_num)+"_mean"])
        metrics_arr_agg.append(np.mean(np.array(metrics_arr), axis=0))
    return np.array(metrics_arr_agg), timesteps_totals, interp_x, color, env

def get_custom_metrics_gini(paths, nagents, nteams, agent_num, moi):
    team_dict = get_team_dict(nagents, nteams)

    metrics_arr_agg, timesteps_totals, interp_x, color, env = get_metrics_arr_agg(paths, agent_num, moi, nteams)

    specialization = []
    for idx in range(metrics_arr_agg.shape[1]):
        specialization.append(gini(metrics_arr_agg[:,idx]))

    specialization = np.array(specialization)

    interpolated = []
    # for x, y, in zip(timesteps_totals, [df.episode_reward_mean for df in dfs]):
    for x, y, in zip(timesteps_totals, [specialization]):
        interp_y = np.interp(interp_x, x, y, left=np.nan, right=np.nan)
        interpolated.append(interp_y)

    for team_num in team_dict.keys():
        if agent_num in team_dict[team_num]:
            t_num = team_num

    agent_plot_data = PlotData(
        [interp_x] * 5, interpolated, "Specialization", r'a-'+str(agent_num) + r'/$T_{'+str(t_num) + r'}$', color,
    )
    return agent_plot_data, env

def get_team_custom_metrics_gini(paths, nagents, nteams, team_num, moi):
    team_dict = get_team_dict(nagents, nteams)

    split_metrics = {key: [] for key in moi}

    for agent_num in team_dict[team_num]:
        metrics_arr_agg, timesteps_totals, interp_x, _, env = get_metrics_arr_agg(paths, agent_num, moi, nteams)
        for idx, metric in enumerate(moi):
            split_metrics[metric].append(metrics_arr_agg[idx])

    metric_gini = {key: [] for key in moi}
    for key, metric_arr in split_metrics.items():
        np_metric_arr = np.array(metric_arr)
        for idx in range(np_metric_arr.shape[1]):
            metric_gini[key].append(gini(np_metric_arr[:,idx]))

    _,_,rewards,_ = get_team_rewards(paths, nagents, nteams, team_num)

    color = get_color_from_team_label(team_num)

    goodness = []
    for idx in range(len(list(metric_gini.values())[0])):
        ginis = []
        for key in moi:
            ginis.append(metric_gini[key][idx])

        goodness.append(rewards[0][idx] * np.mean(np.array([ginis])))
    goodness = np.array(goodness)

    interpolated = []
    # for x, y, in zip(timesteps_totals, [df.episode_reward_mean for df in dfs]):
    for x, y, in zip(timesteps_totals, [goodness]):
        interp_y = np.interp(interp_x, x, y, left=np.nan, right=np.nan)
        interpolated.append(interp_y)

    team_plot_data = PlotData(
        [interp_x] * 5, interpolated, "Team Performance", "Team-"+str(team_num), color,
    )
    return team_plot_data, env


def get_custom_metrics(paths, nagents, nteams, agent_num, metric_type, credo=False):
    if credo: nteam_to_pass = 3
    else: nteam_to_pass = nteams
    team_dict = get_team_dict(nagents, nteam_to_pass)

    if metric_type == "cleaning":
        metric_column_name = "custom_metrics/cleaning_beam_agent-"
        y_axis_label = "Cleaning"
    elif metric_type == "fire":
        metric_column_name = "custom_metrics/fire_beam_agent-"
        y_axis_label = "Punish"
    elif metric_type == "apples":
        metric_column_name = "custom_metrics/apples_agent-"
        y_axis_label = "Apples"

    dfs = []
    for path in paths:
        # try:
        df = pd.read_csv(path, sep=",", error_bad_lines=False)
        # except:
            # print(path)
            # exit()
        # Set NaN values to 0, common at start of training due to ray behavior
        df = df.fillna(0)
        dfs.append(df)

    # for col in list(dfs[0].columns):
    #     print(col)
    # exit()

    env, model_name = get_env_and_model_name_from_path(paths[0])

    # hack for credo to color agents as 3 teams
    if credo: nteam_to_pass = 3
    else: nteam_to_pass = nteams
    color = get_color_from_agent_num(agent_num, nteam_to_pass)

    # Convert environment steps to 1e8 representation
    timesteps_totals = [df.timesteps_total for df in dfs]
    timesteps_totals = [
        [timestep / 1e8 for timestep in timesteps_total] for timesteps_total in timesteps_totals
    ]

    most_timesteps = np.max(list(map(len, timesteps_totals)))
    x_min = np.nanmin(list(map(np.nanmin, timesteps_totals)))
    x_max = np.nanmax(list(map(np.nanmax, timesteps_totals)))
    # Cut off plotting at 5e8 steps
    x_max = min(x_max, 5.0)
    interp_x = np.linspace(x_min, x_max, most_timesteps)
    interpolated = []
    # print(list(dfs[0].columns))
    # exit()
    reward_arr = []
    for df in dfs:
        reward_arr.append(list(df[metric_column_name+str(agent_num)+"_mean"]))

    # reward_arr = np.mean(np.array(reward_arr), axis=0)[:-2]
    # timesteps_totals = [timesteps_totals[0][:-2]]

    # for x, y, in zip(timesteps_totals, [df.episode_reward_mean for df in dfs]):
    for x, y, in zip(timesteps_totals, reward_arr):
        # print(len(x))
        # print(y)
        # print(interp_x.shape)
        # exit()
        interp_y = np.interp(interp_x, x, y, left=np.nan, right=np.nan)
        interpolated.append(interp_y)

    for team_num in team_dict.keys():
        if agent_num in team_dict[team_num]:
            t_num = team_num

    agent_plot_data = PlotData(
        [interp_x] * 5, interpolated, y_axis_label, 'a-' + str(agent_num) + r'/$T_{' + str(t_num) + r'}$', color,
    )
    return agent_plot_data, env




def get_learning_metric(paths, nagents, nteams, agent_num, metric_type, credo=False):

    if credo: nteam_to_pass = 3
    else: nteam_to_pass = nteams
    
    team_dict = get_team_dict(nagents, nteam_to_pass)

    dfs = []
    for path in paths:
        df = pd.read_csv(path, sep=",", error_bad_lines=False)
        df = df.fillna(0)
        dfs.append(df)

    env, model_name = get_env_and_model_name_from_path(paths[0])

    # # hack for credo to color agents as 3 teams
    # if credo: nteam_to_pass = 3
    # else: nteam_to_pass = nteams
    color = get_color_from_agent_num(agent_num, nteam_to_pass)

    # Convert environment steps to 1e8 representation
    timesteps_totals = [df.timesteps_total for df in dfs]
    timesteps_totals = [
        [timestep / 1e8 for timestep in timesteps_total] for timesteps_total in timesteps_totals
    ]

    most_timesteps = np.max(list(map(len, timesteps_totals)))
    x_min = np.nanmin(list(map(np.nanmin, timesteps_totals)))
    x_max = np.nanmax(list(map(np.nanmax, timesteps_totals)))
    # Cut off plotting at 5e8 steps
    x_max = min(x_max, 5.0)
    interp_x = np.linspace(x_min, x_max, most_timesteps)
    interpolated = []
    loss_arr = []
    for df in dfs:
        loss_arr.append(list(df["info/learner/agent-"+str(agent_num)+"/"+metric_type]))

    # for x, y, in zip(timesteps_totals, [df.episode_reward_mean for df in dfs]):
    for x, y, in zip(timesteps_totals, loss_arr):
        interp_y = np.interp(interp_x, x, y, left=np.nan, right=np.nan)
        interpolated.append(interp_y)

    for team_num in team_dict.keys():
        if agent_num in team_dict[team_num]:
            t_num = team_num

    agent_plot_data = PlotData(
        [interp_x] * 5, interpolated, "Met: "+metric_type, 'a-' + str(agent_num) + r'/$T_{' + str(t_num) + r'}$', color,
    )
    return agent_plot_data, env




def get_custom_metrics2(paths, nagents, nteams, agent_num, metric_type):
    team_dict = get_team_dict(nagents, nteams)

    if metric_type == "cleaning":
        metric_column_name = "custom_metrics/cleaning_beam_agent-"
        y_axis_label = "Cleaning"
    elif metric_type == "fire":
        metric_column_name = "custom_metrics/fire_beam_agent-"
        y_axis_label = "Punish"
    elif metric_type == "apples":
        metric_column_name = "custom_metrics/apples_agent-"
        y_axis_label = "Apples"

    dfs = []
    for path in paths:
        # try:
        df = pd.read_csv(path, sep=",", error_bad_lines=False)
        # except:
            # print(path)
            # exit()
        # Set NaN values to 0, common at start of training due to ray behavior
        df = df.fillna(0)
        dfs.append(df)

    # for col in list(dfs[0].columns):
    #     print(col)
    # exit()

    env, model_name = get_env_and_model_name_from_path(paths[0])
    color = get_color_from_agent_num(agent_num, nteams)

    # Convert environment steps to 1e8 representation
    timesteps_totals = [df.timesteps_total for df in dfs]
    timesteps_totals = [
        [timestep / 1e8 for timestep in timesteps_total] for timesteps_total in timesteps_totals
    ]

    most_timesteps = np.max(list(map(len, timesteps_totals)))
    x_min = np.nanmin(list(map(np.nanmin, timesteps_totals)))
    x_max = np.nanmax(list(map(np.nanmax, timesteps_totals)))
    # Cut off plotting at 5e8 steps
    x_max = min(x_max, 5.0)
    interp_x = np.linspace(x_min, x_max, most_timesteps)
    interpolated = []
    # print(list(dfs[0].columns))
    # exit()
    reward_arr = []
    for df in dfs:
        reward_arr.append(list(df[metric_column_name+str(agent_num)+"_mean"]))

    # reward_arr = np.mean(np.array(reward_arr), axis=0)[:-2]
    # timesteps_totals = [timesteps_totals[0][:-2]]

    # for x, y, in zip(timesteps_totals, [df.episode_reward_mean for df in dfs]):
    for x, y, in zip(timesteps_totals, reward_arr):
        # print(len(x))
        # print(y)
        # print(interp_x.shape)
        # exit()
        interp_y = np.interp(interp_x, x, y, left=np.nan, right=np.nan)
        interpolated.append(interp_y)

    for team_num in team_dict.keys():
        if agent_num in team_dict[team_num]:
            t_num = team_num

    agent_plot_data = PlotData(
        [interp_x] * 5, interpolated, y_axis_label, 'a-' + str(agent_num) + r'/$T_{' + str(t_num) + r'}$', color,
    )
    return [interp_x] * 5, interpolated, y_axis_label, env



def get_experiment_gini_label(paths, nteams, nagents, label, collective):
    dfs = []
    for path in paths:
        df = pd.read_csv(path, sep=",", error_bad_lines=False)
        # Set NaN values to 0, common at start of training due to ray behavior
        df = df.fillna(0)
        dfs.append(df)

    env, model_name = get_env_and_model_name_from_path(paths[0])
    # color = get_color_from_model_name(model_name)
    if collective: nteams+=1
    color = get_color_from_num_teams(nteams)

    # Convert environment steps to 1e8 representation
    timesteps_totals = [df.timesteps_total for df in dfs]
    timesteps_totals = [
        [timestep / 1e8 for timestep in timesteps_total] for timesteps_total in timesteps_totals
    ]

    most_timesteps = np.max(list(map(len, timesteps_totals)))
    x_min = np.nanmin(list(map(np.nanmin, timesteps_totals)))
    x_max = np.nanmax(list(map(np.nanmax, timesteps_totals)))
    # Cut off plotting at 5e8 steps
    x_max = min(x_max, 5.0)
    interp_x = np.linspace(x_min, x_max, most_timesteps)
    interpolated = []

    df_ginis = []
    for df in dfs:
        reward_arr = []
        for agent_num in range(nagents):
            reward_arr.append(df["policy_reward_mean/agent-"+str(agent_num)])

        gini_arr = get_gini_from_reward_arr(reward_arr)
        df_ginis.append(gini_arr)


    # for x, y, in zip(timesteps_totals, [df.episode_reward_mean for df in dfs]):
    for x, y, in zip(timesteps_totals, df_ginis): #[reward_arr]
        interp_y = np.interp(interp_x, x, y, left=np.nan, right=np.nan)
        interpolated.append(interp_y)

    gini_plotdata = PlotData(
        [interp_x] * 5, interpolated, "Equality", label, color,
    )
    return gini_plotdata, env

def get_gini_from_reward_arr(reward_arr):
    ginis = []
    reward_arr = np.array(reward_arr)
    for idx in range(reward_arr.shape[1]):
        ginis.append(1 - gini(reward_arr[:,idx]))
    # gini_arr.append(np.array(ginis))
    return np.array(ginis)


def get_experiment_rewards_label(paths, nteams, nagents, label, collective, credo, diff_sizes=False):
    dfs = []
    for path in paths:
        df = pd.read_csv(path, sep=",", error_bad_lines=False)
        # Set NaN values to 0, common at start of training due to ray behavior
        df = df.fillna(0)
        dfs.append(df)

    env, model_name = get_env_and_model_name_from_path(paths[0])
    # color = get_color_from_model_name(model_name)
    if collective: nteams+=1

    if diff_sizes:
        # color = get_color_from_agent_num(nagents, nteams)
        color = get_color_from_agent_num_single_team(nagents, nteams)
    elif not credo:
        color = get_color_from_num_teams(nteams)
    else:
        r = float(label.split("-")[0]) #np.random.random()
        b = float(label.split("-")[1])
        g = float(label.split("-")[2])
        color = (r, g, b)

    # Convert environment steps to 1e8 representation
    timesteps_totals = [df.timesteps_total for df in dfs]
    timesteps_totals = [
        [timestep / 1e8 for timestep in timesteps_total] for timesteps_total in timesteps_totals
    ]

    most_timesteps = np.max(list(map(len, timesteps_totals)))
    x_min = np.nanmin(list(map(np.nanmin, timesteps_totals)))
    x_max = np.nanmax(list(map(np.nanmax, timesteps_totals)))
    # Cut off plotting at 5e8 steps
    x_max = min(x_max, 5.0)
    interp_x = np.linspace(x_min, x_max, most_timesteps)
    interpolated = []
    # print(list(dfs[0].columns))
    # exit()

    # print(x_min)

    # print(dfs[0].episode_reward_mean)
    # for x, y, in zip(timesteps_totals, [df.episode_reward_mean for df in dfs]):
    #     print(y)
    #     exit()

    # a = [df.episode_reward_mean for df in dfs]
    # print(len(a))
    # for b in a:
    #     print(b)

    # print()

    df_rewards = []
    for df in dfs:
        reward_arr = []
        for agent_num in range(nagents):
            reward_arr.append(df["policy_reward_mean/agent-"+str(agent_num)])
        reward_arr = np.mean(np.array(reward_arr), axis=0)
        # if nteams == 2: reward_arr = reward_arr[:-2]
        df_rewards.append(reward_arr)
    # if collective:
    #     reward_arr = reward_arr / 6

    # print(len(df_rewards))
    # for b in df_rewards:
    #     print(b)
    # exit()

    # if nteams == 2:
    #     df_rewards = [df_rewards[0][:-2]]
    #     timesteps_totals = [timesteps_totals[0][:-2]]


    # for x, y, in zip(timesteps_totals, [df.episode_reward_mean for df in dfs]):
    for x, y, in zip(timesteps_totals, df_rewards): #[reward_arr]
        interp_y = np.interp(interp_x, x, y, left=np.nan, right=np.nan)
        interpolated.append(interp_y)

    reward_plotdata = PlotData(
        [interp_x] * 5, interpolated, "Mean Pop. Reward", label, color,
    )
    return reward_plotdata, env


def get_credo_experiment_custom_metrics(paths, metric_type):
    if metric_type == "cleaning":
        metric_column_name = "custom_metrics/cleaning_beam_agent-"
        other_metric_column_name = "custom_metrics/apples_agent-"
    elif metric_type == "apples":
        metric_column_name = "custom_metrics/apples_agent-"
        other_metric_column_name = "custom_metrics/cleaning_beam_agent-"

    dfs = []
    agent_metrics_full = []
    for path in paths:
        df = pd.read_csv(path, sep=",", error_bad_lines=False)
        # Set NaN values to 0, common at start of training due to ray behavior
        df = df.fillna(0)

        ''' calculates totals '''
        # agent_total_metrics = []
        # for agent_num in range(6):
        #     agent_record = np.array(df[metric_column_name+str(agent_num)+"_mean"])
        #     agent_total_metrics.append(agent_record[-int(agent_record.size*0.25):])
        # agent_metrics_full.append(np.mean(np.sum(np.array(agent_total_metrics), axis=0)))

        ''' calculates # of agents '''
        agent_num = []
        count = 0
        for agent_num in range(6):
            agent_record = np.array(df[metric_column_name+str(agent_num)+"_mean"])
            if np.mean(np.array(agent_record[-int(agent_record.size*0.25):])) > 333:
                count+=1
        agent_metrics_full.append(count)

    return np.mean(np.array(agent_metrics_full))


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h


def get_credo_experiment_rewards(paths, nagents):
    dfs = []
    mean_rewards = []
    for path in paths:
        df = pd.read_csv(path, sep=",", error_bad_lines=False)
        # Set NaN values to 0, common at start of training due to ray behavior
        df = df.fillna(0)
        # dfs.append(df)
        mean_rewards.append(np.mean(np.array([df.episode_reward_mean[-int(df.episode_reward_mean.size*0.25):]])) / nagents )
    return np.mean(np.array(mean_rewards)), mean_confidence_interval(np.array(mean_rewards))

def get_credo_experiment_gini(paths, nagents):
    dfs = []
    mean_ginis = []
    for path in paths:
        df = pd.read_csv(path, sep=",", error_bad_lines=False)
        # Set NaN values to 0, common at start of training due to ray behavior
        df = df.fillna(0)
        # dfs.append(df)
        # print(df.episode_reward_mean)
        reward_arr = []
        for agent_num in range(6):
            agent_rewards = df["policy_reward_mean/agent-"+str(agent_num)]
            reward_arr.append(agent_rewards[-int(agent_rewards.size*0.25):])

        mean_ginis.append(1 - gini(np.array(reward_arr)))
    return np.mean(np.array(mean_ginis))

def get_credo_experiment_matrix(paths, nagents, metric):
    dfs = []
    matrix = np.zeros((len(paths),6))
    for path_count, path in enumerate(paths):
        df = pd.read_csv(path, sep=",", error_bad_lines=False)
        # Set NaN values to 0, common at start of training due to ray behavior
        df = df.fillna(0)
        # dfs.append(df)
        for agent_num in range(6):
            if 'custom_metrics' in metric:
                agent_metric = df[metric+str(agent_num)+"_mean"]
            else:
                agent_metric = df[metric+str(agent_num)]
            
            matrix[path_count,agent_num] = np.mean(np.array(agent_metric[-int(agent_metric.size*0.25):]))
    return matrix





def get_experiment_rewards(paths):
    dfs = []
    for path in paths:
        df = pd.read_csv(path, sep=",", error_bad_lines=False)
        # Set NaN values to 0, common at start of training due to ray behavior
        df = df.fillna(0)
        dfs.append(df)

    env, model_name = get_env_and_model_name_from_path(paths[0])
    color = get_color_from_model_name(model_name)

    # Convert environment steps to 1e8 representation
    timesteps_totals = [df.timesteps_total for df in dfs]
    timesteps_totals = [
        [timestep / 1e8 for timestep in timesteps_total] for timesteps_total in timesteps_totals
    ]

    most_timesteps = np.max(list(map(len, timesteps_totals)))
    x_min = np.nanmin(list(map(np.nanmin, timesteps_totals)))
    x_max = np.nanmax(list(map(np.nanmax, timesteps_totals)))
    # Cut off plotting at 5e8 steps
    x_max = min(x_max, 5.0)
    interp_x = np.linspace(x_min, x_max, most_timesteps)
    interpolated = []
    # print(list(dfs[0].columns))
    # exit()
    for x, y, in zip(timesteps_totals, [df.episode_reward_mean for df in dfs]):
        interp_y = np.interp(interp_x, x, y, left=np.nan, right=np.nan)
        interpolated.append(interp_y)
    reward_plotdata = PlotData(
        [interp_x] * 5, interpolated, "Mean collective reward", model_name, color,
    )
    return reward_plotdata, env


def change_color_luminosity(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    Taken from https://stackoverflow.com/a/49601444
    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys

    c = mc.cnames[color] if color in mc.cnames else color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))


def getExperimentParameters(path):
    path_arr = path.split("_")
    rogue = {"flag": False, "nrogue":0, "deg_rogue": 0.0}

    for word in path_arr:
        if word[1:] == "teams":
            nteams = int(word[0])
        elif word[1:] == "agents":
            nagents = int(word[0])
        elif "rogue" in word:
            rogue = {"flag": True, "nrogue":int(word[0]), "deg_rogue": word[6:]}
    return nteams, nagents, rogue

def plot_separate_results(scenarios):

    scratch_base = '/scratch/ssd004/scratch/dtradke/ray_results/'


    # Plot separate experiment results
    # for category_folder in get_all_subdirs(ray_results_path):
    for category_folder in get_all_subdirs(scratch_base):
        # if category_folder.split("/")[-1] == "cleanup_baseline_PPO_2teams_6agents":
        # if category_folder.split("/")[-1] in scenarios:
        #     print("Plotting category folder: " + category_folder.split("/")[-1])
        #     path = category_folder.split("/")[-1]
        #     nteams = int(path.split("_")[-4][0])
        #     nagents = int(path.split("_")[-3][0])

        if category_folder.split("/")[-1] in scenarios:
            path = category_folder.split("/")[-1]

            nteams, nagents, rogue = getExperimentParameters(path)

            csvs = []
            experiment_folders = get_all_subdirs(category_folder)
            for experiment_folder in experiment_folders:
                csv_path = experiment_folder + "/progress.csv"
                if os.path.getsize(csv_path) > 0:
                    csvs.append(csv_path)
            plot_csvs_results(csvs, nteams, nagents, rogue)

def compare_losses():
    scratch_base = '/scratch/ssd004/scratch/dtradke/ray_results/'
    nteam_arr = [1, 2, 3, 6]
    df = pd.DataFrame(columns=['nteams', 'policy_loss', 'vf_loss', 'total_loss'])
    for nteam in nteam_arr:
        exp_str = "cleanup_baseline_PPO_"+str(nteam)+"teams_6agents_custom_metrics_rgb"
        for category_folder in get_all_subdirs(scratch_base):
            if category_folder.split("/")[-1] == exp_str:
                path = category_folder.split("/")[-1]

                nteams, nagents, rogue = getExperimentParameters(path)
                csvs = []
                experiment_folders = get_all_subdirs(category_folder)
                for experiment_folder in experiment_folders:
                    csv_path = experiment_folder + "/progress.csv"
                    if os.path.getsize(csv_path) > 0:
                        csvs.append(csv_path)
                loss_dict = collect_losses(csvs, nteams, nagents, rogue)
                row = [nteam, loss_dict['policy_loss'], loss_dict['vf_loss'], loss_dict['total_loss']]
                df.loc[df.shape[0]] = row
    print(df)
    exit()


def plot_combined_results(scenarios):
    # Plot combined experiment rewards per environment, for means per model
    env_rewards = {}
    for category_folder in get_all_subdirs(ray_results_path):
        # if category_folder.split("/")[-1] == "cleanup_baseline_PPO_2teams_6agents":
        if category_folder.split("/")[-1] in scenarios:
            path = category_folder.split("/")[-1]

            nteams, nagents, rogue = getExperimentParameters(path)

            csvs = []
            experiment_folders = get_all_subdirs(category_folder)
            for experiment_folder in experiment_folders:
                csv_path = experiment_folder + "/progress.csv"
                if os.path.getsize(csv_path) > 0:
                    csvs.append(csv_path)

            experiment_rewards, env = get_experiment_rewards(csvs)
            if env not in env_rewards:
                env_rewards[env] = []
            env_rewards[env].append(experiment_rewards)

        for env, experiment_rewards in env_rewards.items():
            print("Plotting collective plot for environment: " + env)

            def plot_fn():
                plot_multiple_category_result(experiment_rewards)

            # Add filler to path which will be removed
            collective_env_path = "collective/filler/"
            plot_and_save(plot_fn, collective_env_path, env + "_collective_reward", nteams, nagents, rogue, compare=True)



def plot_team_results(scenarios):
    '''
    This function plots the team reward for each team that is in the scenario listed inside the "scenarios" list.

    [scenarios] = list of full paths from the base to the experiment
            ex: '/h/dtradke/ray_results/cleanup_baseline_PPO_1teams_6agents_custom_metrics_rgb/'
    '''

    # scenarios = ["cleanup_baseline_PPO_3teams_6agents_custom_metrics"]

    # global ray_results_path

    # ray_results_path = ray_results_path + "6teams_6agents"
    # Plot combined experiment rewards per environment, for means per model
    env_rewards = {}
    rogue = {"flag": False}
    team_reward_arrays = []
    for category_folder in scenarios: #get_all_subdirs(ray_results_path):
    # for category_folder in get_all_subdirs(ray_results_path):
    #     if category_folder.split("/")[-1] in scenarios:
        path = category_folder.split("/")[-1]

        nteams, nagents, rogue = getExperimentParameters(path)

        
        experiment_folders = get_all_subdirs(category_folder)
        experiment_folders = natsort.natsorted(experiment_folders)
        # experiment_folders = [experiment_folders[0]]

        all_csvs = []
        for trial_num, experiment_folder in enumerate(experiment_folders):
            inner_env_rewards = {}
            inner_team_reward_arrays = []
            csvs = []
            e = [experiment_folder]

            for experiment_folder in e:
                csv_path = experiment_folder + "/progress.csv"
                if os.path.getsize(csv_path) > 0:
                    csvs.append(csv_path)
                    all_csvs.append(csv_path)

            for team_count in range(nteams):
                team_rewards, env, team_reward_array, _ = get_team_rewards(csvs, nagents, nteams, team_count)
                inner_team_reward_arrays.append(team_reward_array[0])
                if env not in inner_env_rewards:
                    inner_env_rewards[env] = []
                inner_env_rewards[env].append(team_rewards)
            

            for env, team_rewards in inner_env_rewards.items():
                print("Plotting collective plot for environment: " + env)

                def plot_fn():
                    plot_multiple_category_result(team_rewards)

                # Add filler to path which will be removed
                collective_env_path = "team_results/filler/"
                # plot_and_save(plot_fn, collective_env_path, env + "_team_reward_"+str(nteams)+"teams_"+str(nagents)+"agents", nteams, nagents, rogue)
                # plot_and_save(plot_fn, collective_env_path, "TEST_"+env + "_rgb_team_reward_credo_"+str(nteams)+"teams_"+str(nagents)+"agents", nteams, nagents, rogue)
                plot_and_save(plot_fn, collective_env_path, env + "_team_reward_"+str(nteams)+"teams_"+str(nagents)+"agents", nteams, nagents, rogue, trial=trial_num)

        ''' to plot all trials together '''
        # for team_count in range(nteams):
        #     team_rewards, env, team_reward_array, _ = get_team_rewards(all_csvs, nagents, nteams, team_count)
        #     team_reward_arrays.append(team_reward_array[0])
        #     if env not in env_rewards:
        #         env_rewards[env] = []
        #     env_rewards[env].append(team_rewards)

        # for env, team_rewards in env_rewards.items():
        #     print("Plotting collective plot for environment: " + env)

        #     def plot_fn():
        #         plot_multiple_category_result(team_rewards)

        #     # Add filler to path which will be removed
        #     collective_env_path = "team_results/filler/"
        #     plot_and_save(plot_fn, collective_env_path, env + "_team_reward_"+str(nteams)+"teams_"+str(nagents)+"agents", nteams, nagents, rogue)




def plot_team_compare_ginis(category_folders):
    '''
    This function compares all of the ginis for experiments in the "scenarios" list on the same graph
    '''
    # global ray_results_path

    # scenarios = ["cleanup_baseline_PPO_6teams_6agents_collective", "cleanup_baseline_PPO_6teams_6agents", "cleanup_baseline_PPO_3teams_6agents", "cleanup_baseline_PPO_2teams_6agents", "cleanup_baseline_PPO_1teams_6agents"]
    # scenarios = ["cleanup_baseline_PPO_1teams_6agents_custom_metrics_rgb", "cleanup_baseline_PPO_2teams_6agents_custom_metrics_rgb", "cleanup_baseline_PPO_3teams_6agents_custom_metrics_rgb", "cleanup_baseline_PPO_6teams_6agents_custom_metrics_rgb"]
    # scenarios = ["cleanup_baseline_PPO_1teams_6agents_moreapples0-2_rgb", "cleanup_baseline_PPO_2teams_6agents_moreapples0-2_rgb", "cleanup_baseline_PPO_3teams_6agents_moreapples0-2_rgb", "cleanup_baseline_PPO_6teams_6agents_moreapples0-2_rgb"]


    env_rewards = {}
    rogue = {"flag": False}
    # for category_folder in get_all_subdirs(ray_results_path):
    #     if category_folder.split("/")[-1] in scenarios:
    for category_folder in category_folders:
        path = category_folder.split("/")[-1]

        nteams, nagents, rogue = getExperimentParameters(path)
        graph_label = str(nteams)+"/"+str(int(nagents/nteams))
        collective = False
        
        csvs = []
        experiment_folders = get_all_subdirs(category_folder)
        
        for experiment_folder in experiment_folders:
            csv_path = experiment_folder + "/progress.csv"
            if os.path.getsize(csv_path) > 0:
                csvs.append(csv_path)


        experiment_rewards, env = get_experiment_gini_label(csvs, nteams, nagents, graph_label, collective)
        if env not in env_rewards:
            env_rewards[env] = []
        env_rewards[env].append(experiment_rewards)


        for env, rewards in env_rewards.items():
            # print("Plotting collective plot for environment: " + env)

            def plot_fn():
                plot_multiple_category_result(rewards)

            # Add filler to path which will be removed
            collective_env_path = "gini_results/filler/"
            plot_and_save(plot_fn, collective_env_path, env+ "_compare_team_ginis_"+str(nagents)+"agents", nteams, nagents, rogue, compare=True)

            # plot_and_save(plot_fn, collective_env_path, "TEST_"+env+ "_compare_team_rewards_"+str(nagents)+"agents", nteams, nagents, rogue, compare=True)
            # exit()


def plot_triangle_credo_custom_metrics(met, category_folders):
    '''
    Plots the population reward of different credos in the triangle plots
    '''
    scale=5
    # category_folders = get_all_subdirs(ray_results_path)
    # scenarios, scratch_base = getCredoPaths()
    # category_folders = category_folders + get_all_subdirs(scratch_base)

    d = dict()
    env_rewards = {}
    rogue = {"flag": False}
    for category_folder in category_folders: #get_all_subdirs(ray_results_path):
        # if category_folder.split("/")[-1] in scenarios:
        path = category_folder.split("/")[-1]

        nteams, nagents, rogue = getExperimentParameters(path)
        graph_label = str(nteams)+"/"+str(int(nagents/nteams))
        collective = False

        graph_label = getCredoVals(path, nteams)
        i = int(float(graph_label.split("-")[0])*scale)
        j = int(float(graph_label.split("-")[1])*scale)
        k = int(float(graph_label.split("-")[2])*scale)

        csvs = []
        experiment_folders = get_all_subdirs(category_folder)
        
        for experiment_folder in experiment_folders:
            csv_path = experiment_folder + "/progress.csv"
            if os.path.getsize(csv_path) > 0:
                csvs.append(csv_path)

        d[(i,j)] = get_credo_experiment_custom_metrics(csvs, met)

        print(i/5, ", ", j/5, ", ", k/5, ": ", d[(i,j)])

    figure, tax = ternary.figure(scale=scale)
    ax = tax.heatmap(d, style="h", cmap='coolwarm', vmin=0, vmax=4)

    axes_colors = {'b': 'k',
            'l': 'k',
            'r': 'k'}

    tax.left_axis_label("Team ($\phi$)", fontsize=24, offset=0.23, color=axes_colors['l'])
    tax.right_axis_label("Self ($\psi$)", fontsize=24, offset=0.23, color=axes_colors['r'])
    tax.bottom_axis_label("System ($\omega$)", fontsize=24, offset=0.23, color=axes_colors['b'])

    ticks = [i / float(scale) for i in range(scale+1)]
    tax.ticks(ticks=ticks, axis='rlb', linewidth=1, clockwise=True, fontsize=21,
            axes_colors=axes_colors, offset=0.04, tick_formats="%0.1f")

    tax.gridlines(multiple=1, linewidth=4,
            horizontal_kwargs={'color': axes_colors['l']},
            left_kwargs={'color': axes_colors['r']},
            right_kwargs={'color': axes_colors['b']},
            alpha=0.7)

    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=20)

    plt.axis('off')

    if met == 'apples':
        title_label = "Apple Pickers"
    elif met == 'cleaning':
        title_label = "River Cleaners"
    tax.set_title(title_label, fontsize=24, y=1.1)

    path_split = path.split("/")[-1]
    pngpath2 = '../results/'+path_split.split("_")[0]+'/team_rewards_rgb/'
    if rogue["flag"]:
        pngpath2 = pngpath2 + "rogue/"+str(rogue["nrogue"])+"rogue/deg"+rogue["deg_rogue"]+"/"
    pngfile2 = pngpath2 + "credo-"+met+"count_3teams_2agents.png"
    if not os.path.exists(pngpath2):
        os.makedirs(pngpath2)

    plt.savefig(pngfile2,bbox_inches='tight', dpi=300)
    plt.close()
    print("saved: ", pngfile2)


def plot_credo_agent_custom_metrics(metric_type, trial, category_folder, reward):
    '''
    This function plots the policies of agents for "apples" or "cleaning" given a specific credo, trial number, and metric (apple/clean)
    '''
    scale=5
    

    d = dict()
    env_rewards = {}
    rogue = {"flag": False}
    # for category_folder in category_folders: #get_all_subdirs(ray_results_path):
    #     if category_folder.split("/")[-1] in scenarios:
    path = category_folder.split("/")[-2]


    nteams, nagents, rogue = getExperimentParameters(path)
    graph_label = str(nteams)+"/"+str(int(nagents/nteams))
    collective = False

    graph_label = getCredoVals(path, nteams)
    i = int(float(graph_label.split("-")[0])*scale)
    j = int(float(graph_label.split("-")[1])*scale)
    k = int(float(graph_label.split("-")[2])*scale)

    csvs = []
    # experiment_folders = get_all_subdirs(category_folder)
    # # grabs one trial
    # experiment_folders = natsort.natsorted(experiment_folders)
    experiment_folders = [category_folder] #[experiment_folders[trial]]
    
    for experiment_folder in experiment_folders:
        csv_path = experiment_folder + "/progress.csv"
        if os.path.getsize(csv_path) > 0:
            csvs.append(csv_path)

    for agent_num in range(nagents):
        agent_metric, env = get_custom_metrics(csvs, nagents, nteams, agent_num, metric_type, credo=True) #(csvs, nagents, nteams, agent_num, moi)
        if env not in env_rewards:
            env_rewards[env] = []
        env_rewards[env].append(agent_metric)

    for env, agent_metric in env_rewards.items():
        print("Plotting collective plot for environment: " + env)

        def plot_fn():
            plot_multiple_category_result(agent_metric)

        # Add filler to path which will be removed
        collective_env_path = "collective_results/filler/"
        # plot_and_save(plot_fn, collective_env_path, env+"_gini_specialization_"+str(nteams)+"teams_"+str(nagents)+"agents", nteams, nagents, rogue)
        plot_and_save(plot_fn, collective_env_path, "credo_"+ env+ "_"+graph_label+"_"+metric_type+"_"+str(nagents)+"agents", nteams, nagents, rogue, compare=True, trial=trial, credo=True, reward=reward)
    # exit()


def plot_credo_agent_custom_metrics_fillbetween(metric_type, category_folder):
    '''
    This function plots the policies of agents for "apples" or "cleaning" given a specific credo and metric (apple/clean)

    trying to make it a heatmap
    '''
    scale=5
    

    d = dict()
    env_rewards = {}
    rogue = {"flag": False}
    # for category_folder in category_folders: #get_all_subdirs(ray_results_path):
    #     if category_folder.split("/")[-1] in scenarios:
    path = category_folder.split("/")[-1]

    nteams, nagents, rogue = getExperimentParameters(path)
    graph_label = str(nteams)+"/"+str(int(nagents/nteams))
    collective = False

    graph_label = getCredoVals(path, nteams)
    i = int(float(graph_label.split("-")[0])*scale)
    j = int(float(graph_label.split("-")[1])*scale)
    k = int(float(graph_label.split("-")[2])*scale)

    csvs = []
    experiment_folders = get_all_subdirs(category_folder)

    # grabs one trial
    # experiment_folders = natsort.natsorted(experiment_folders)
    # experiment_folders = [experiment_folders[trial]]
    
    for experiment_folder in experiment_folders:
        csv_path = experiment_folder + "/progress.csv"
        if os.path.getsize(csv_path) > 0:
            csvs.append(csv_path)


    for agent_num in range(nagents):
        for csv in csvs:
            # agent_metric, env = get_custom_metrics([csv], nagents, nteams, agent_num, metric_type)
            x_vals, y_vals, y_axis_label, env = get_custom_metrics2([csv], nagents, nteams, agent_num, metric_type)
            if env not in env_rewards:
                env_rewards[env] = []
            env_rewards[env].append((x_vals, y_vals))

    for env, agent_metric in env_rewards.items():
        print("Plotting collective plot for environment: " + env)

        def plot_fn():
            plot_multiple_category_result_fill(agent_metric, y_axis_label)

        # Add filler to path which will be removed
        collective_env_path = "collective_results/filler/"
        # plot_and_save(plot_fn, collective_env_path, env+"_gini_specialization_"+str(nteams)+"teams_"+str(nagents)+"agents", nteams, nagents, rogue)
        plot_and_save(plot_fn, collective_env_path, "credo_"+ env+ "_"+graph_label+"_"+metric_type+"_fillbtwn"+str(nagents)+"agents", nteams, nagents, rogue, compare=True, credo=True)
    # exit()


def plot_triangle_credo_reward_lines(result, category_folders):
    '''
    Plots the population reward of different credos in the triangle plots
    '''
    scale=5
    category_folders = get_all_subdirs(ray_results_path)
    scenarios, scratch_base = getCredoPaths()
    category_folders = category_folders + get_all_subdirs(scratch_base)

    d = dict()
    conf_int = dict()
    env_rewards = {}
    rogue = {"flag": False}
    for category_folder in category_folders: #get_all_subdirs(ray_results_path):
        # if category_folder.split("/")[-1] in scenarios:
        path = category_folder.split("/")[-1]

        nteams, nagents, rogue = getExperimentParameters(path)
        graph_label = str(nteams)+"/"+str(int(nagents/nteams))
        collective = False

        graph_label = getCredoVals(path, nteams)
        i = int(float(graph_label.split("-")[0])*scale)
        j = int(float(graph_label.split("-")[1])*scale)
        k = int(float(graph_label.split("-")[2])*scale)

        csvs = []
        experiment_folders = get_all_subdirs(category_folder)
        
        for experiment_folder in experiment_folders:
            csv_path = experiment_folder + "/progress.csv"
            if os.path.getsize(csv_path) > 0:
                csvs.append(csv_path)

        if result == "Reward":
            reward, conf = get_credo_experiment_rewards(csvs, nagents)
            d[(i,j)] = reward
            conf_int[(i,j)] = conf
            fname_add = 'reward'
        else:
            d[(i,j)] = get_credo_experiment_gini(csvs, nagents)
            fname_add = 'gini'

        print(i/5, ", ", j/5, ", ", k/5, ": ", d[(i,j)])

    to_plot_all, to_plot_all_confs = [], []
    for j in range(6):
        to_plot, confs = [], []
        for i in range(6):
            if (i,j) in list(d.keys()):
                to_plot.append(d[(i,j)])
                confs.append(conf_int[(i,j)])
        to_plot_all.append(to_plot)
        to_plot_all_confs.append(confs)

    plt.style.use('seaborn-whitegrid')

    colors = ['b', 'g', 'r', 'c', 'm', 'tab:olive']

    for idx, to_plot in enumerate(to_plot_all[:-1]):
        plt.errorbar(np.linspace(0,(len(to_plot)-1)/5, len(to_plot)), to_plot, color=colors[idx], fmt='o', yerr=to_plot_all_confs[idx], capsize=6, label="Team-Focus="+str((6-len(to_plot))/5))
        plt.plot(np.linspace(0,(len(to_plot)-1)/5, len(to_plot)), to_plot, color=colors[idx])
    # plt.scatter(0, to_plot_all[-1], label="Team-Focus=1.0")
    plt.errorbar(0, to_plot_all[-1], fmt='o', yerr=to_plot_all_confs[-1], capsize=6, label="Team-Focus=1.0")
    
    plt.xlabel("Self-Focus", fontsize=22)
    plt.ylabel("Reward", fontsize=22)
    # plt.title("Team-Focus: "+str(j/5))
    plt.legend(loc='upper right', fontsize=12, frameon=True)

    path_split = path.split("/")[-1]
    pngpath2 = '../results/'+path_split.split("_")[0]+'/team_rewards_rgb/'
    pngfile2 = pngpath2 + "credo_"+fname_add+"_3teams_2agents_0.0teamw.png"
    if not os.path.exists(pngpath2):
        os.makedirs(pngpath2)

    plt.savefig(pngfile2,bbox_inches='tight', dpi=300)
    plt.close()
    print("saved: ", pngfile2)
    exit()



def plot_triangle_credo_reward(result, category_folders):
    '''
    Plots the population reward of different credos in the triangle plots
    '''

    scale=5
    d = dict()
    rogue = {"flag": False}
    for category_folder in category_folders: #get_all_subdirs(ray_results_path):
        # if category_folder.split("/")[-1] in scenarios:
        path = category_folder.split("/")[-1]
        

        nteams, nagents, rogue = getExperimentParameters(path)
        graph_label = str(nteams)+"/"+str(int(nagents/nteams))
        collective = False

        graph_label = getCredoVals(path, nteams)
        i = int(float(graph_label.split("-")[0])*scale)
        j = int(float(graph_label.split("-")[1])*scale)
        k = int(float(graph_label.split("-")[2])*scale)


        csvs = []
        experiment_folders = get_all_subdirs(category_folder)
        
        for experiment_folder in experiment_folders:
            csv_path = experiment_folder + "/progress.csv"
            if os.path.getsize(csv_path) > 0:
                csvs.append(csv_path)

        if result == "Reward":
            d[(i,j)], conf = get_credo_experiment_rewards(csvs, nagents)
            fname_add = 'reward'
        else:
            d[(i,j)] = get_credo_experiment_gini(csvs, nagents)
            fname_add = 'gini'

        print(i/5, ", ", j/5, ", ", k/5, ": ", d[(i,j)])

    figure, tax = ternary.figure(scale=scale)

    if fname_add == 'gini': bax = tax.heatmap(d, style="h", cmap='Reds', vmin=0, vmax=1) # coolwarm
    else: bax = tax.heatmap(d, style="h", cmap='Reds', vmin=0) #, vmin=0, vmax=4

    axes_colors = {'b': 'k',
            'l': 'k',
            'r': 'k'}

    tax.left_axis_label("Team ($\phi$)", fontsize=24, offset=0.23, color=axes_colors['l'])
    tax.right_axis_label("Self ($\psi$)", fontsize=24, offset=0.23, color=axes_colors['r'])
    tax.bottom_axis_label("System ($\omega$)", fontsize=24, offset=0.23, color=axes_colors['b'])

    ticks = [i / float(scale) for i in range(scale+1)]
    tax.ticks(ticks=ticks, axis='rlb', linewidth=1, clockwise=True, fontsize=21,
            axes_colors=axes_colors, offset=0.04, tick_formats="%0.1f")

    tax.gridlines(multiple=1, linewidth=4,
            horizontal_kwargs={'color': axes_colors['l']},
            left_kwargs={'color': axes_colors['r']},
            right_kwargs={'color': axes_colors['b']},
            alpha=0.7)

    y_flag = False
    # d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}
    for key in list(d.keys()):
        pt = [[key[0], key[1]]]
        if key[0] == 1 and key[1] == 0:
            tax.scatter(pt, marker='*', color='lime', ec='k',s=500, label='Scenario 1', zorder=10)
        elif key[0] == 1 and key[1] == 4:
            tax.scatter(pt, marker='*', color='yellow', ec='k',s=500, label='Scenario 2', zorder=10)
        elif key[0] == 0 and key[1] == 4:
            tax.scatter(pt, marker='*', color='yellow', ec='k',s=500, zorder=10)
        elif key[0] == 1 and key[1] == 3:
            tax.scatter(pt, marker='*', color='yellow', ec='k',s=500, zorder=10)
        elif key[0] == 0 and key[1] == 5:
            tax.scatter(pt, marker='*', color='yellow', ec='k',s=500, zorder=10)
        # else:
        #     if not y_flag: 
        #         y_flag = True
        #         tax.scatter(pt, marker='*', color='yellow', ec='k',s=250, label='Scenario 2', zorder=10)
        #     else:
        #         tax.scatter(pt, marker='*', color='yellow', ec='k',s=250, zorder=10)

    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=20)

    plt.axis('off')

    if fname_add == 'gini': tax.set_title("Mean "+result, fontsize=24, y=1.1)
    else: tax.set_title("Mean "+result+r' ($\overline{R^{\mathbf{cr}}}$)', fontsize=24, y=1.1)
    
    plt.legend(loc='upper left', fontsize=15, bbox_to_anchor=(-0.22, 1.1))

    path_split = path.split("/")[-1]
    pngpath2 = '../results/'+path_split.split("_")[0]+'/team_rewards_rgb/'
    if rogue["flag"]:
        pngpath2 = pngpath2 + "rogue/"+str(rogue["nrogue"])+"rogue/deg"+rogue["deg_rogue"]+"/"
    pngfile2 = pngpath2 + "credo_"+fname_add+"_3teams_2agents_stars_NEWEXPS_reds.png"
    if not os.path.exists(pngpath2):
        os.makedirs(pngpath2)

    plt.savefig(pngfile2,bbox_inches='tight', dpi=300)
    plt.close()
    print("saved: ", pngfile2)


def plot_scatter_reward_cleaning(scenarios):
    '''
    Plots the a scatterplot cross-referencing cleaning and reward of the agent
    '''

    scale=5
    d = dict()
    rogue = {"flag": False}
    rewards, apples, cleaning = np.array([]), np.array([]), np.array([])
    for category_folder in category_folders: #get_all_subdirs(ray_results_path):
        # if category_folder.split("/")[-1] in scenarios:
        path = category_folder.split("/")[-1]

        nteams, nagents, rogue = getExperimentParameters(path)
        graph_label = str(nteams)+"/"+str(int(nagents/nteams))
        collective = False

        graph_label = getCredoVals(path, nteams)
        i = int(float(graph_label.split("-")[0])*scale)
        j = int(float(graph_label.split("-")[1])*scale)
        k = int(float(graph_label.split("-")[2])*scale)


        csvs = []
        experiment_folders = get_all_subdirs(category_folder)
        
        for experiment_folder in experiment_folders:
            csv_path = experiment_folder + "/progress.csv"
            if os.path.getsize(csv_path) > 0:
                csvs.append(csv_path)

        reward_mtx = get_credo_experiment_matrix(csvs, nagents, metric='policy_reward_mean/agent-')
        cleaning_mtx = get_credo_experiment_matrix(csvs, nagents, metric='custom_metrics/cleaning_beam_agent-')
        apple_mtx = get_credo_experiment_matrix(csvs, nagents, metric='custom_metrics/apples_agent-')

        rewards = np.concatenate((rewards, reward_mtx.flatten()))
        cleaning = np.concatenate((cleaning, cleaning_mtx.flatten()))
        apples = np.concatenate((apples, apple_mtx.flatten()))

        # color = np.array([i/5, j/5, k/5])

        # plt.scatter(reward_mtx.flatten(), apple_mtx.flatten(), c=color.reshape(1,-1), label=graph_label)

        # print(i/5, ", ", j/5, ", ", k/5)

    path_split = path.split("/")[-1]
    pngpath2 = '../results/'+path_split.split("_")[0]+'/team_rewards_rgb/'
    pngfile2 = pngpath2 + "credo_scatter-apple-reward_3teams_2agents.png"

    np.save(pngpath2+"rewards.npy", rewards)
    np.save(pngpath2+"apples.npy", apples)
    np.save(pngpath2+"cleaning.npy", cleaning)
    exit()
    


    plt.xlabel("Rewards", fontsize=22)
    # plt.ylabel("Cleaning", fontsize=22)
    plt.ylabel("Apples", fontsize=22)

    # plt.legend(ncol=2)
    plt.legend(fontsize=13,ncol=2, loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)

    path_split = path.split("/")[-1]
    pngpath2 = '../results/'+path_split.split("_")[0]+'/team_rewards_rgb/'
    pngfile2 = pngpath2 + "credo_scatter-apple-reward_3teams_2agents.png"
    if not os.path.exists(pngpath2):
        os.makedirs(pngpath2)

    plt.savefig(pngfile2,bbox_inches='tight', dpi=300)
    plt.close()
    print("saved: ", pngfile2)
    exit()


def getCredoPaths():

    scale=5

    credo_addition = 'credo_3teams_6agents/'
    ray_results_addition = 'ray_results/'
    scratch_base = '/scratch/ssd004/scratch/dtradke/'
    paths = []
    d = dict()
    for (i,j,k) in simplex_iterator(scale):
        # if i == 5:
        #     path = scratch_base+ray_results_addition+"/cleanup_baseline_PPO_6teams_6agents_custom_metrics_rgb"
        if j == 5:
            path = scratch_base+ray_results_addition+"/cleanup_baseline_PPO_3teams_6agents_custom_metrics_rgb"
        # elif k == 5:
        #     path = scratch_base+ray_results_addition+"/cleanup_baseline_PPO_1teams_6agents_custom_metrics_rgb"
        else:
            path = scratch_base+credo_addition+"/cleanup_baseline_PPO_3teams_6agents_" + str(i/5) + "-" + str(j/5) + "-" + str(k/5) + "_rgb"
        paths.append(path)
    return paths, scratch_base

def getCredoVals(path, nteams):
    # 0.8-0.2-0.0_rgb
    if path.split("_")[-2][0] == "0":
        return path.split("_")[-2]
    # elif nteams == 1:
    if path.split("_")[-2][-3] == "1":
        return "0.0-0.0-1.0"
    # elif nteams == 6:
    elif path.split("_")[-2][0] == "1":
        return "1.0-0.0-0.0"
    else:
        return "0.0-1.0-0.0"


def plot_team_compare_results(scenarios=None, diff_sizes=False, credo=False):
    '''
    This function compares all of the experiments in the "scenarios" list on the same graph

    if 'diff_sizes' = True, it means I'm plotting for example: 3 scenarios with 1 team (different sized teams)... i.e., 1/3, 1/4, and 1/5
    '''
    

    if credo and scenarios is not None:
        # global ray_results_path
        category_folders = get_all_subdirs(ray_results_path)
        scenarios, scratch_base = getCredoPaths()
        category_folders = category_folders + get_all_subdirs(scratch_base)
    else:
        category_folders = scenarios
        scenarios = [s.split("/")[-1] for s in scenarios]
        # scenarios = ["cleanup_baseline_PPO_6teams_6agents_collective", "cleanup_baseline_PPO_6teams_6agents", "cleanup_baseline_PPO_3teams_6agents", "cleanup_baseline_PPO_2teams_6agents", "cleanup_baseline_PPO_1teams_6agents"]
        # scenarios = ["cleanup_baseline_PPO_1teams_6agents_custom_metrics_rgb", "cleanup_baseline_PPO_2teams_6agents_custom_metrics_rgb", "cleanup_baseline_PPO_3teams_6agents_custom_metrics_rgb", "cleanup_baseline_PPO_6teams_6agents_custom_metrics_rgb"]
        # scenarios = ["cleanup_baseline_PPO_1teams_6agents_moreapples0-2_rgb", "cleanup_baseline_PPO_2teams_6agents_moreapples0-2_rgb", "cleanup_baseline_PPO_3teams_6agents_moreapples0-2_rgb", "cleanup_baseline_PPO_6teams_6agents_moreapples0-2_rgb"]

    # print(category_folders)
    # exit()

    env_rewards = {}
    rogue = {"flag": False}
    for category_folder in category_folders: #get_all_subdirs(ray_results_path):
        if category_folder.split("/")[-1] in scenarios:
            path = category_folder.split("/")[-1]

            nteams, nagents, rogue = getExperimentParameters(path)
            graph_label = str(nteams)+"/"+str(int(nagents/nteams))
            collective = False

            if credo:
                graph_label = getCredoVals(path, nteams)

            if '1.0' not in graph_label.split("-"):
                if '0.0' in graph_label.split("-"):
                # if graph_label.split("-")[0] != '0.0':
                    continue
            
            csvs = []
            experiment_folders = get_all_subdirs(category_folder)
            experiment_folders = [natsort.natsorted(experiment_folders)[0]]
            
            for experiment_folder in experiment_folders:
                csv_path = experiment_folder + "/progress.csv"
                if os.path.getsize(csv_path) > 0:
                    csvs.append(csv_path)

            experiment_rewards, env = get_experiment_rewards_label(csvs, nteams, nagents, graph_label, collective, credo, diff_sizes)
            if env not in env_rewards:
                env_rewards[env] = []
            env_rewards[env].append(experiment_rewards)

        for env, rewards in env_rewards.items():
            # print("Plotting collective plot for environment: " + env)

            def plot_fn():
                plot_multiple_category_result(rewards)

            # Add filler to path which will be removed
            collective_env_path = "compare_team_structures/filler/"
            if credo:
                plot_and_save(plot_fn, collective_env_path, "credo_"+ env+ "_no-zeros_"+str(nagents)+"agents", nteams, nagents, rogue, compare=True, credo=True)
            else:
                if diff_sizes:
                    plot_and_save(plot_fn, collective_env_path, env+ "_compare_team_rewards_"+str(len(scenarios))+"diffsizeteams", nteams, nagents, rogue, compare=True)
                else:
                    plot_and_save(plot_fn, collective_env_path, env+ "_compare_team_rewards_"+str(nagents)+"agents", nteams, nagents, rogue, compare=True)

                # plot_and_save(plot_fn, collective_env_path, "TEST_"+env+ "_compare_team_rewards_"+str(nagents)+"agents", nteams, nagents, rogue, compare=True)
                # exit()

def plot_agent_custom_metrics_gini(scenarios):
    '''
    This function plots the gini between "apples" and "cleaning" for each agent to determine an agent's specialization
    during a specific scenario. The plot will have num_agents lines representing how specialized each agent is.

    Higher values: more specialization (higher gini between apples and cleaning)
    '''

    moi = ["cleaning", "apples"]

    # scenarios = ["cleanup_baseline_PPO_3teams_6agents_custom_metrics"]

    # Plot combined experiment rewards per environment, for means per model
    env_rewards = {}
    rogue = {"flag": False}
    for category_folder in get_all_subdirs(ray_results_path):
        if category_folder.split("/")[-1] in scenarios:
            path = category_folder.split("/")[-1]
            # nteams = int(path.split("_")[-4][0])
            # nagents = int(path.split("_")[-3][0])
            # if "rogue" in path:
            #     rogue_flag = True
            #     nrogue = int(path.split("_")[-5][0])
            #     deg_rogue = path.split("_")[-5][6:]
            #     rogue = {"flag": rogue_flag, "nrogue":nrogue, "deg_rogue": deg_rogue}

            nteams, nagents, rogue = getExperimentParameters(path)

            
            csvs = []
            experiment_folders = get_all_subdirs(category_folder)
            experiment_folders = natsort.natsorted(experiment_folders)
            experiment_folders = [experiment_folders[-1]]

            for experiment_folder in experiment_folders:
                csv_path = experiment_folder + "/progress.csv"
                if os.path.getsize(csv_path) > 0:
                    csvs.append(csv_path)
            
            for agent_num in range(nagents):
                agent_metric, env = get_custom_metrics_gini(csvs, nagents, nteams, agent_num, moi)
                if env not in env_rewards:
                    env_rewards[env] = []
                env_rewards[env].append(agent_metric)

        for env, agent_metric in env_rewards.items():
            print("Plotting collective plot for environment: " + env)

            def plot_fn():
                plot_multiple_category_result(agent_metric)

            # Add filler to path which will be removed
            collective_env_path = "gini_results/filler/"
            plot_and_save(plot_fn, collective_env_path, env+"_gini_specialization_"+str(nteams)+"teams_"+str(nagents)+"agents", nteams, nagents, rogue)



def plot_agent_custom_metrics(metric_type, category_folder, trial):
    '''
    This function plots the custom metrics for each agent on the same plot (default is 6 agents)

    metric type one of: "cleaning", "fire", or "apples"
    '''


    # scenarios = ["cleanup_baseline_PPO_3teams_6agents_custom_metrics"]

    # global ray_results_path

    # ray_results_path2 = ray_results_path + "6teams_6agents"

    # Plot combined experiment rewards per environment, for means per model
    env_rewards = {}
    rogue = {"flag": False}
    # for category_folder in get_all_subdirs(ray_results_path):
    #     if category_folder.split("/")[-1] in scenarios:
    path = category_folder.split("/")[-1]
    nteams, nagents, rogue = getExperimentParameters(path) 
    csvs = []
    experiment_folders = get_all_subdirs(category_folder)


    experiment_folders = natsort.natsorted(experiment_folders)
    experiment_folders = [experiment_folders[trial]]        # does index which experiment

    for experiment_folder in experiment_folders:
        csv_path = experiment_folder + "/progress.csv"
        if os.path.getsize(csv_path) > 0:
            csvs.append(csv_path)
    
    for agent_num in range(nagents):
        agent_metric, env = get_custom_metrics(csvs, nagents, nteams, agent_num, metric_type)
        if env not in env_rewards:
            env_rewards[env] = []
        env_rewards[env].append(agent_metric)

    for env, agent_metric in env_rewards.items():
        print("Plotting collective plot for environment: " + env)

        def plot_fn():
            plot_multiple_category_result(agent_metric)

        # Add filler to path which will be removed
        collective_env_path = "custom_metrics/filler/"
        plot_and_save(plot_fn, collective_env_path, env+"_"+str(nteams)+"teams_"+str(nagents)+"agents_"+metric_type, nteams, nagents, rogue, trial=trial)

        # plot_and_save(plot_fn, collective_env_path, "TEST_"+env+"_moreapples0-2_"+str(nteams)+"_teams_"+str(nagents)+"agents_"+metric_type, nteams, nagents, rogue)


def get_reward_from_experiment_folders(experiment_folders):
    '''
    This function plots the policies of agents for "apples" or "cleaning" given a specific credo, trial number, and metric (apple/clean)
    '''
    
    path = experiment_folders[0].split("/")[-2]

    nteams, nagents, rogue = getExperimentParameters(path)
    csvs = []
    
    for experiment_folder in experiment_folders:
        csv_path = experiment_folder + "/progress.csv"
        if os.path.getsize(csv_path) > 0:
            csvs.append(csv_path)

    reward, conf = get_credo_experiment_rewards(csvs, nagents)
    return reward

def plot_indiv_agent_metric_policies(category_folders, metrics):

    # todo_folders = ['/scratch/ssd004/scratch/dtradke/credo_3teams_6agents//cleanup_baseline_PPO_3teams_6agents_0.4-0.6-0.0_rgb',
    #                 '/scratch/ssd004/scratch/dtradke/credo_3teams_6agents//cleanup_baseline_PPO_3teams_6agents_0.2-0.8-0.0_rgb',
    #                 '/scratch/ssd004/scratch/dtradke/credo_3teams_6agents//cleanup_baseline_PPO_3teams_6agents_0.2-0.0-0.8_rgb',
    #                 '/scratch/ssd004/scratch/dtradke/credo_3teams_6agents//cleanup_baseline_PPO_3teams_6agents_0.0-0.0-1.0_rgb']

    for cat_count, category_folder in enumerate(category_folders):
        # print(category_folder, ": ", category_folder in todo_folders)
        # if category_folder in todo_folders:
        cat_folders = get_all_subdirs(category_folder)
        experiment_folders = natsort.natsorted(cat_folders)
        exp_reward = get_reward_from_experiment_folders(experiment_folders)
        for trial_num, exp_dir in enumerate(experiment_folders):
            for met in metrics:
                plot_credo_agent_custom_metrics(met, trial_num, exp_dir, exp_reward)
    exit()

def plot_learning_metrics(category_folders):

    # loss_types = ['policy_loss', 'vf_loss', 'total_loss']

    learning_metrics = None
    for cat_count, category_folder in enumerate(category_folders):
        # if category_folder in todo_folders:
        cat_folders = get_all_subdirs(category_folder)
        experiment_folders = natsort.natsorted(cat_folders)
        exp_reward = get_reward_from_experiment_folders(experiment_folders)
        credo = True if 'credo' in category_folder else False
        for trial_num, exp_dir in enumerate(experiment_folders):
            learning_metrics = getPathForColumnNames(exp_dir)
            # for lt in loss_types:
            #     plotTrialLosses(lt, trial_num, exp_dir, exp_reward, credo)
            for learning_metric in learning_metrics:
                plotTrialLearningMetrics(learning_metric, trial_num, exp_dir, exp_reward, credo)
    exit()

if __name__ == "__main__":

    # compare_losses()

    nteam_arr = [1, 2, 3, 6]
    # for nteam in nteam_arr:
    #     exp_str = "cleanup_baseline_PPO_"+str(nteam)+"teams_6agents_custom_metrics_rgb"
    #     scenarios = [exp_str]
    #     # print("Plotting separate results..")
    #     plot_separate_results(scenarios)
    #     # print("Plotting combined results..")
    #     # plot_combined_results()
    #     compare_losses(scenarios)


    nteam_arr = [1]
    nagent_arr = [1, 2, 3, 4, 5, 6]

    category_folders = []
    for nteam in nteam_arr:
        for nagent in nagent_arr:
            category_folders.append("/scratch/ssd004/scratch/dtradke/ray_results/cleanup_baseline_PPO_"+str(nteam)+"teams_"+str(nagent)+"agents_custom_metrics_rgb")
            # category_folders.append("../../ray_results"+str(nteam)+"teams_"+str(nagent)+"agents/cleanup_baseline_PPO_"+str(nteam)+"teams_"+str(nagent)+"agents_custom_metrics_rgb")


    # plot_team_results(category_folders)                   # graphs teams in one scenario/team size separately... pass in list of paths to experiments to plot separately
    plot_team_compare_results(category_folders, diff_sizes=True)           # graphs all teams on same plot (DIFF TEAM SIZES)... pass in list of paths to experiments to compare
    # # plot_team_compare_ginis(category_folders)             # plots gini all teams on same plot (DIFF TEAM SIZES)... pass in list of paths to experiments to compare
    # # exit()

    # metrics = ["apples", "cleaning"]
    # for category_folder in category_folders:
    #     cat_folders = get_all_subdirs(category_folder)
    #     experiment_folders = natsort.natsorted(cat_folders)
    #     for trial_num, exp_dir in enumerate(experiment_folders):
    #         for met in metrics:
    #             plot_agent_custom_metrics(met, category_folder, trial_num) # metric type one of: "cleaning", "fire", or "apples"

    # plot_learning_metrics(category_folders)
    exit()




    

    ''' CREDO BELOW '''
    # category_folders = get_all_subdirs(ray_results_path)
    # scenarios, scratch_base = getCredoPaths()
    # category_folders = scenarios

    '''             triangle plots for credo paper              '''
    # plot_triangle_credo_reward("Reward", scenarios) # "Reward", "Inv. Gini"              plotting reward or gini in triangle
    # plot_triangle_credo_reward("Inverse Gini", scenarios)
    # plot_triangle_credo_reward_lines("Reward", scenarios)
    # plot_scatter_reward_cleaning(scenarios)

    metrics = ["apples", "cleaning"]
    # for met in metrics:
    #     plot_triangle_credo_custom_metrics(met, scenarios)                 # plotting pickers or cleaners in triangle for credo

    # plot_indiv_agent_metric_policies(category_folders, metrics)             # plots apple/cleaning curves for agents with all credo


    # for cat_count, category_folder in enumerate(category_folders):
    #     for met in metrics:
    #         plot_credo_agent_custom_metrics_fillbetween(met, category_folder)       # trying to make heatmap plots... not very nice
    
