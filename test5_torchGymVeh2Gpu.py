'''
测试5，是否pytorch+自定义gym-Veh2Crash+GPU(环境torch24PY38Gpu)能用
'''

import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())  #输出为True，则安装无误

import gymnasium as gym
import gym_examples
import os

from stable_baselines3 import A2C
import imageio
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy   
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
class SaveOnBestTrainingRewardCallback(BaseCallback):

    def __init__(self, check_freq: int, log_dir: str, verbose: int = 0):
       
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        
    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 2 episodes
              mean_reward = np.mean(y[-1:])
              
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)
                  self.model.save("a2c_Veh2CrashEnv-v1")
                  print(f"Best mean reward: {mean_reward:.2f},Num timesteps: {self.num_timesteps}")

        return True
    
    
    
if __name__ == "__main__":
    log_dir = "./tmp/"
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    

    
    env = gym.make("gym_examples/Veh2CrashEnv-v1")
    env = Monitor(env, log_dir)

    model = A2C("MlpPolicy", env, verbose=1)
    
    callback = SaveOnBestTrainingRewardCallback(check_freq=1, log_dir=log_dir,verbose = 0)
    
    model.learn(total_timesteps=10000, callback=callback)
    
    
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=None)
   
    print("\n\n mean_reward:",mean_reward,"std_reward:",std_reward)
    
    model.load("a2c_Veh2CrashEnv-v1")
    model.load("./tmp/best_model.zip")

    #####################################################################################
    # execute only if run as a script
    
    vec_env = model.get_env()
    obs = vec_env.reset()
    episode_reward = 0

    recordStates = []
    recordReward = []
    
    recordStates.append(obs)
    recordReward.append(0)
    while True:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        
     
        recordStates.append(list(obs)) 
        recordReward.append(reward[0]) 
        
        episode_reward += reward 
        print('actions:',action)
        print('states:{}'.format(np.round(obs,2)))
        print('reward:%.2f,episode_reward:%.3f' %(reward,episode_reward))
        if done:
            break
                
    recordStates1 = np.array(recordStates)
    recordStates1 = np.squeeze(recordStates1)
    print(recordStates1.shape)
    print(recordStates1[0:2])
    
    x1 = recordStates1[:,0]
    y1 = recordStates1[:,3]
    x2 = recordStates1[:,4]
    y2 = recordStates1[:,6]
    dist = (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2) 
    dist = np.sqrt(dist)
    
   
    plt.plot(x1,y1,'rx', label='v1')
    plt.plot(x2,y2,'b.', label='v2')
    y = recordReward
    plt.plot(x1,y,'g.', label='reward')
    
    dist = (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2) 
    dist = np.sqrt(dist)
    plt.plot(x1,dist,'y.', label='dist')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    plt.close()
    with imageio.get_writer('./veh2CrashEnv_pytorchGPU.gif', mode='I',fps =2) as writer:

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
            plt.show()
            plt.savefig('./tmp_pytorchGPU.jpg')
            img_array = imageio.imread('./tmp_pytorchGPU.jpg')
            #print(img_array.shape)
            #frames.append(img_array)
            writer.append_data(img_array)

    print('ended')