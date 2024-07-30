from tensorforce import Runner
from tensorforce import Agent, Environment, Runner
from Veh2CrashEnv import Veh2CrashEnv
import numpy as np
import imageio
import matplotlib.pyplot as plt
from tensorforce.agents import PPOAgent

def main():
    # OpenAI-Gym environment specification
    #environment = dict(environment='gym', level='Veh2CrashEnv')
    environment = Environment.create(environment='gym', level='Veh2CrashEnv-v1', max_episode_timesteps=500)
    
    network_spec = [
      dict(type='dense', size=128, activation='relu'),
      dict(type='dense', size=128, activation='relu')
            ]

    if 1:
        # PPO agent specification
        agent1 = dict(
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
            baseline=dict(type='auto', size=64, depth=2),
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
        
     
        agent2 = dict(
            agent='ac',
            # Automatically configured network
            network=dict(type='auto', size=128, depth=2), # PPO optimization parameters
            batch_size=10, update_frequency=2, learning_rate=3e-4,
            
            # Reward estimation
            discount=0.95, predict_terminal_values=True,
            reward_processing=None,
            # Preprocessing
            state_preprocessing='linear_normalization',
            # Exploration
            exploration=0.01, variable_noise=0.05,
            # Default additional config values
            config=None,
            # Save agent every 10 updates and keep the 5 most recent checkpoints
            saver=dict(directory='model', frequency=20, max_checkpoints=2,append ="episodes"),
            # Log all available Tensorboard summaries
            summarizer=dict(directory='summaries', summaries='all'),
            # Do not record agent-environment interaction trace
            recorder=None
        )
                
       
        # or: Agent.create(agent='ppo', environment=environment, ...)
        # with additional argument "environment" and, if applicable, "parallel_interactions"

        # Initialize the runner
        runner = Runner(agent=agent1, environment=environment, max_episode_timesteps=500)

        # Train for 200 episodes
        runner.run(num_episodes=2000)
        runner.close()

    # plus agent.close() and environment.close() if created separately
    if 0: #测试SAVE and Load
        agent = Agent.load(directory='model', format='checkpoint', environment=environment)
        runner = Runner(agent=agent, environment=environment,max_episode_timesteps=500)
        runner.run(num_episodes=100, evaluation=True)

    
    
    # Evaluate for 1 episodes
    agent = Agent.load(directory='model', format='checkpoint', environment=environment)
    sum_rewards = 0.0
    recordStates = []
    recordReward = []
    
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
            print('actions:',actions)
            print('states:{}'.format(np.round(states,2)))
            print('reward:%.2f,sum_rewards:%.2f' %(reward,sum_rewards))
            recordStates.append(states)
            recordReward.append(reward) 
            
    #print('states:',states)   
    #print('Mean evaluation return:', sum_rewards)

    # Close agent and environment
    agent.close()
    environment.close()
    
    
    ########统计
    recordStates1 = np.array(recordStates)
    x1 = recordStates1[:,0]
    y1 = recordStates1[:,3]
    x2 = recordStates1[:,4]
    y2 = recordStates1[:,6]
    dist = (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2) 
    dist = np.sqrt(dist)
    
    with imageio.get_writer('./veh2CrashEnv_tensorforce.gif', mode='I',fps =2) as writer:

        for step in range(x1.shape[0]):
            #print(step)
            x1pts = x1[step]
            y1pts = y1[step]
            x2pts = x2[step]
            y2pts = y2[step]
            distpts = dist[step]
            plt.plot(x1pts,y1pts,'rx', label='v1')
            plt.plot(x2pts,y2pts,'b.', label='v2')
            plt.plot(x1pts,distpts,'y.', label='v1')
            title = "step:{},time:{:.2f}".format(step,step*0.1)
            plt.title(title)
            plt.savefig('./tmp_tensorforce.jpg')
            img_array = imageio.v2.imread('./tmp_tensorforce.jpg')
            #print(img_array.shape)
            #frames.append(img_array)
            writer.append_data(img_array)

if __name__ == '__main__':
    main()