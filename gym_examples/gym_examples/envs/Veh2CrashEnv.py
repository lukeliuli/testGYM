import numpy as np

import gymnasium as gym
from gymnasium import spaces
import math

class Veh2CrashEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, initEnv=None):
        
        self.runTimes = 0
        self.stepNum = 0

        self.runState = "init"
        self.action_space = spaces.Discrete(4)#0:[0,0].1:[0 1],2:[1,0],3:[1,1,]
        #self.action_space = spaces.MultiBinary(2)
        tmp= np.array([1,1,1,1,1,1,1,1])
        self.observation_space = spaces.Box(tmp*(-1000), tmp*1000,dtype=np.float32)
        self.info = {}

        self.deltaT = 0.1

        assert initEnv is None or len(initEnv) == 8
        if initEnv == None:
            self.envStateInit = [0,20/3.6,0,0,   13,20/3.6/2,-6.5,0]
        else:
            self.envStateInit = initEnv

        
        self.envState = self.envStateInit

        v1_xpos,v1_xvel,v1_ypos,v1_yvel,v2_xpos,v2_xvel,v2_ypos,v2_yvel = self.envState

        self.v1_xpos0 = v1_xpos
        self.v1_ypos0 = v1_ypos
        self.v2_xpos0 = v2_xpos 
        self.v2_ypos0 = v2_ypos
        self.v1_xvel0 = self.envStateInit[1]
        self.v2_xvel0 = self.envStateInit[5]
        self.A = 6.5
    




   

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
       


    def _get_obs(self):
        tmp = np.array(self.envState)
        return tmp
        

    def _get_info(self):
        info = {"runState":self.envState,"stepNum":self.stepNum,"Time":self.stepNum*self.deltaT}
        return info

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

   
        self.stepNum = 0
        self.runstate = "init"
        
     
        if options == None or  options['initEnv'] == None:
            self.envStateInit = [0,20/3.6,0,0,   13,20/3.6/2,-6.5,0]
        else:
            self.envStateInit = options['initEnv']
        
        self.envState = self.envStateInit
        
        v1_xpos,v1_xvel,v1_ypos,v1_yvel,v2_xpos,v2_xvel,v2_ypos,v2_yvel = self.envState

        self.v1_xpos0 = v1_xpos
        self.v1_ypos0 = v1_ypos
        self.v2_xpos0 = v2_xpos 
        self.v2_ypos0 = v2_ypos

        info = {"runState":self.envState,"stepNum":self.stepNum,"Time":self.stepNum*self.deltaT}
        #print('states:{}'.format(np.round(self.envState,2)))

  
        reward = 0
        terminated = 0
        return np.array(self.envState), info

    def render(self):
        pass

    def _render_frame(self):
        pass
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def step(self,action):  
        
        terminated = 0

        if self.runState == "init":
            self.runState = "running"
            self.runTimes = self.runTimes+1


        if self.runState == "running":
            self.stepNum += 1

        if self.runState == "closed":
            #print('Error:setp and self.runState == closed')
            info = {"runState":self.runState,"stepNum":self.stepNum,"Time":self.stepNum*self.deltaT}
            return np.array(self.envState), None, None,info


        
        v1_xpos,v1_xvel,v1_ypos,v1_yvel,v2_xpos,v2_xvel,v2_ypos,v2_yvel = self.envState
        
        if action  ==  0:
            v1_action = 0
            v2_action = 0
        if action  ==  1:
            v1_action = 0
            v2_action = 1
        if action  ==  2:
            v1_action = 1
            v2_action = 0    
        if action  ==  3:
            v1_action = 1
            v2_action = 1
       
        if v1_action ==  0:
            v1_envState = [v1_xpos,0,v1_ypos,0] # 简单的ACTION 为0 停止，为1以固定X速度向前走
        if v2_action ==  0:
            v2_envState = [v2_xpos,0,v2_ypos,0]

        if v1_action ==  1:
            v1_xvel = self.v1_xvel0
            v1_envState = [v1_xpos+v1_xvel*self.deltaT,v1_xvel,v1_ypos+v1_yvel*self.deltaT,v1_yvel] # 简单的ACTION 为0 停止，为1以固定X速度向前走

        if v2_action ==  1:
            v2_xvel = self.v2_xvel0
            v2_xpos= v2_xpos+v2_xvel*self.deltaT
            #A = 6.5,v2_ypos = A-A*exp(-v2_xpos)
            #self.v2_xpos0 = 13   
            #y = -6.5*exp(-x+13)
            v2_ypos = -self.A*math.exp(-v2_xpos+self.v2_xpos0)
            v2_envState = [v2_xpos,v2_xvel,v2_ypos,v2_yvel] # 简单的ACTION 为0 停止，为1以固定X速度向前走
       
        v1_envState.extend(v2_envState)
        self.envState =  v1_envState

        reward = v1_action+v2_action
        dist = np.sqrt((v1_xpos -v2_xpos)*(v1_xpos -v2_xpos)+(v1_ypos -v2_ypos)*(v1_ypos -v2_ypos))
        #reward = reward+(math.tanh(2*(dist-7))-1)
        #rewardGood1 = 2*reward+math.tanh(v1_xpos/20)+math.tanh(v2_xpos/20)+math.tanh(v2_ypos/6.5)
        #rewardGood2 = 3*reward+math.tanh(v1_xpos/20)+math.tanh(v2_xpos/20)+math.tanh(v2_ypos/6.5)
        reward = 2*reward+math.tanh(v1_xpos/20)+math.tanh(v2_xpos/20)+math.tanh(v2_ypos/6.5)
      
        
        if dist <7 or dist >20:
            terminated = 1
            reward = reward-100
        
        if dist>7 and v2_xpos>20 and v1_xpos>20:
            terminated = 1
            reward = reward+1000 
            
        if  self.stepNum *self.deltaT >10:
            terminated = 1
           

        info = {"runState":self.runState,"stepNum":self.stepNum,"Time":self.stepNum*self.deltaT}
        truncated = 0 #gym版本不一样
        return np.array(self.envState), reward, terminated,truncated,info