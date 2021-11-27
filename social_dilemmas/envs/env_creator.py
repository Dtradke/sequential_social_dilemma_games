from social_dilemmas.envs.cleanup import CleanupEnv
from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.switch import SwitchEnv


def get_env_creator(env, num_agents, num_teams, credo, args):
    if env == "harvest":

        def env_creator(_):
            return HarvestEnv(
                num_agents=num_agents,
                num_teams=num_teams,
                credo=credo,
                rogue=args.rogue,
                num_rogue=args.num_rogue,
                rogue_deg=args.rogue_deg,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
            )

    elif env == "cleanup":

        def env_creator(_):
            return CleanupEnv(
                num_agents=num_agents,
                num_teams=num_teams,
                credo=credo,
                rogue=args.rogue,
                num_rogue=args.num_rogue,
                rogue_deg=args.rogue_deg,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
            )

    elif env == "switch":

        def env_creator(_):
            return SwitchEnv(num_agents=num_agents, num_teams=num_teams, args=args)

    return env_creator
