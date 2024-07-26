from tensorforce import Runner
from tensorforce import Agent, Environment, Runner

def main():
    # OpenAI-Gym environment specification
    #environment = dict(environment='gym', level='CartPole-v1')
    environment = Environment.create(environment='gym', level='CartPole-v1', max_episode_timesteps=500)

    if 0:
        # PPO agent specification
        agent = dict(
            agent='ppo',
            # Automatically configured network
            network='auto',
            # PPO optimization parameters
            batch_size=10, update_frequency=2, learning_rate=3e-4, multi_step=10,
            subsampling_fraction=0.33,
            # Reward estimation
            likelihood_ratio_clipping=0.2, discount=0.99, predict_terminal_values=False,
            reward_processing=None,
            # Baseline network and optimizer
            baseline=dict(type='auto', size=32, depth=1),
            baseline_optimizer=dict(optimizer='adam', learning_rate=1e-3, multi_step=10),
            # Regularization
            l2_regularization=0.0, entropy_regularization=0.0,
            # Preprocessing
            state_preprocessing='linear_normalization',
            # Exploration
            exploration=0.0, variable_noise=0.0,
            # Default additional config values
            config=None,
            # Save agent every 10 updates and keep the 5 most recent checkpoints
            saver=dict(directory='model', frequency=10, max_checkpoints=5),
            # Log all available Tensorboard summaries
            summarizer=dict(directory='summaries', summaries='all'),
            # Do not record agent-environment interaction trace
            recorder=None
        )
        # or: Agent.create(agent='ppo', environment=environment, ...)
        # with additional argument "environment" and, if applicable, "parallel_interactions"

        # Initialize the runner
        runner = Runner(agent=agent, environment=environment, max_episode_timesteps=500)

        # Train for 200 episodes
        runner.run(num_episodes=100)
        runner.close()

    # plus agent.close() and environment.close() if created separately
    if 0: #测试SAVE and Load
        agent = Agent.load(directory='model', format='checkpoint', environment=environment)
        runner = Runner(agent=agent, environment=environment,max_episode_timesteps=500)
        runner.run(num_episodes=100, evaluation=True)

    
    
    # Evaluate for 1 episodes
    agent = Agent.load(directory='model', format='checkpoint', environment=environment)
    sum_rewards = 0.0
    for episodes in range(1):
        print('episodes:',episodes)
        states = environment.reset()
        internals = agent.initial_internals()
        terminal = False
        while not terminal:
            actions, internals = agent.act(
                states=states, internals=internals, independent=True, deterministic=True
            )
            states, terminal, reward = environment.execute(actions=actions)
            sum_rewards += reward
            print('reward,sum_rewards:',reward,sum_rewards)
    print('states:',states)   
    print('Mean evaluation return:', sum_rewards)

    # Close agent and environment
    agent.close()
    environment.close()

if __name__ == '__main__':
    main()