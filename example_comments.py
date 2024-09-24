import warnings
from functools import partial as bind

import dreamerv3
import embodied

warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')


def main():
    """
    Main function to configure and run the DreamerV3 agent.
    Sets up the configuration, logging, environment, and replay buffer,
    then initiates the training process.
    """
    config = embodied.Config(dreamerv3.Agent.configs['defaults'])
    config = config.update({
        **dreamerv3.Agent.configs['size100m'],
        'logdir': f'~/logdir/{embodied.timestamp()}-example',
        'run.train_ratio': 32,
    })
    config = embodied.Flags(config).parse()

    print('Logdir:', config.logdir)
    logdir = embodied.Path(config.logdir)
    logdir.mkdir()
    config.save(logdir / 'config.yaml')

    def make_agent(config: embodied.Config) -> dreamerv3.Agent:
        """
        Create an instance of the DreamerV3 agent.
        Args:
            config: Configuration object for the agent and environment.
        Returns:
            An instance of the DreamerV3 agent.
        """
        env = make_env(config)
        agent = dreamerv3.Agent(env.obs_space, env.act_space, config)
        env.close()
        return agent

    def make_logger(config: embodied.Config) -> embodied.Logger:
        """
        Create a logger for tracking training metrics.
        Args:
            config: Configuration object containing logger settings.
        Returns:
            An instance of the embodied.Logger with different outputs (terminal, JSON, TensorBoard).
        """
        logdir = embodied.Path(config.logdir)
        return embodied.Logger(embodied.Counter(), [
            embodied.logger.TerminalOutput(config.filter),
            embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
            embodied.logger.TensorBoardOutput(logdir),
            # embodied.logger.WandbOutput(logdir.name, config=config),
        ])

    def make_replay(config):
        return embodied.replay.Replay(
            length=config.batch_length,
            capacity=config.replay.size,
            directory=embodied.Path(config.logdir) / 'replay',
            online=config.replay.online)

    def make_env(config: embodied.Config, env_id: int = 0) -> embodied.envs.Environment:
        """
        Create and configure the environment for the agent.
        Args:
            config: Configuration object containing environment settings.
            env_id: Optional environment ID for differentiating between multiple environments.

        Returns:
            A wrapped environment ready for interaction with the agent.
        """
        import crafter
        from embodied.envs import from_gym
        env = crafter.Env()
        env = from_gym.FromGym(env)
        env = dreamerv3.wrap_env(env, config)
        return env

    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        batch_length_eval=config.batch_length_eval,
        replay_context=config.replay_context,
    )

    embodied.run.train(
        bind(make_agent, config),
        bind(make_replay, config),
        bind(make_env, config),
        bind(make_logger, config), args)

if __name__ == '__main__':
    main()
