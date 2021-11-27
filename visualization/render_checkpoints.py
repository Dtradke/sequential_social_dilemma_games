import ray
from ray.rllib import _register_all
import natsort


from utility_funcs import get_all_subdirs
from visualization.plot_results import ray_results_path
from visualization.visualizer_rllib import create_parser, run


def create_args(checkpoint_file, video_dir, video_filename):
    args = list()
    args.append(checkpoint_file)
    args.append("--run")
    args.append("PPO")
    args.append("--video-dir")
    args.append(video_dir)
    args.append("--video-filename")
    args.append(video_filename)
    args.append("--steps")
    args.append("1000")

    # print(checkpoint_file)
    # print(video_dir)
    # print(video_filename)
    # exit()
    return args


render_dirs = "cleanup_baseline_PPO_6teams_6agents"


def render():
    video_base_path = ray_results_path + "_video/"
    category_folders = get_all_subdirs(ray_results_path)
    for category_folder in category_folders:
        if category_folder.split("/")[-1] in render_dirs:
            experiment_folders = get_all_subdirs(category_folder)
            for i, experiment_folder in enumerate(experiment_folders):
                print("Rendering experiment" + str(i + 1) + "/" + str(len(experiment_folders)))
                checkpoint_folders = get_all_subdirs(experiment_folder)
                video_path = video_base_path + experiment_folder.split("/")[-1]
                checkpoint_folders = natsort.natsorted(checkpoint_folders)
                checkpoints = [checkpoint_folders[0], checkpoint_folders[-1]]

                for j, checkpoint_folder in enumerate(checkpoints):
                    print("Rendering checkpoint" + str(j + 1) + "/" + str(len(checkpoint_folders)), " --> ", checkpoint_folder)
                    checkpoint_number = checkpoint_folder.split("_")[-1]
                    checkpoint_file = checkpoint_folder + "/checkpoint-" + str(checkpoint_number)
                    print(checkpoint_file)
                    video_filename = str(checkpoint_number)
                    # print(video_filename)
                    # print(checkpoint_folder)
                    # exit()
                    args = create_args(checkpoint_file, video_path, video_filename)
                    # exit()
                    parser = create_parser()
                    parsed_args = parser.parse_args(args)
                    run(parsed_args, parser)
                    ray.shutdown()

                    # Register_all due to this bug: https://github.com/ray-project/ray/issues/8205
                    _register_all()
            exit()


if __name__ == "__main__":
    render()
