import numpy as np
import math
import gym
from gym import spaces
import matplotlib.pyplot as plt
import imageio
class Veh2CrashEnv(gym.Env):

    def __init__(self):


        self.runTimes = 0
        self.stepNum = 0

        self.runState = "init"
        self.action_space = spaces.Discrete(4)#0:[0,0].1:[0 1],2:[1,0],3:[1,1,]
        #self.action_space = spaces.MultiBinary(2)
        tmp= np.array([1,1,1,1,1,1,1,1])
        self.observation_space = spaces.Box(tmp*(-1000), tmp*1000)
        self.info = {}

        self.deltaT = 0.1
        self.envStateInit = [0,20/3.6,0,0,   13,20/3.6/2,-6.5,0]
        self.envState = self.envStateInit

        v1_xpos,v1_xvel,v1_ypos,v1_yvel,v2_xpos,v2_xvel,v2_ypos,v2_yvel = self.envState

        self.v1_xpos0 = v1_xpos
        self.v1_ypos0 = v1_ypos
        self.v2_xpos0 = v2_xpos 
        self.v2_ypos0 = v2_ypos
        self.v1_xvel0 = self.envStateInit[1]
        self.v2_xvel0 = self.envStateInit[5]
        self.A = 6.5
        #print('states:{}'.format(np.round(self.envState,2)))


    def setInitEnv(self,envInit):
        self.envStateInit  = envInit
        self.envState = self.envStateInit
        
        #print(' setInitEnv:',self.envState)
        v1_xpos,v1_xvel,v1_ypos,v1_yvel,v2_xpos,v2_xvel,v2_ypos,v2_yvel = self.envState
        #print('v1_xpos,v1_xvel,v1_ypos,v1_yvel,v2_xpos,v2_xvel,v2_ypos,v2_yvel')
        #print(v1_xpos,v1_xvel,v1_ypos,v1_yvel,v2_xpos,v2_xvel,v2_ypos,v2_yvel)
        self.v1_xpos0 = v1_xpos
        self.v1_ypos0 = v1_ypos
        self.v2_xpos0 = v2_xpos 
        self.v2_ypos0 = v2_ypos
        self.v1_xvel0 = self.envStateInit[1]
        self.v2_xvel0 = self.envStateInit[5]
        

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
        #reward = reward+v1_xpos/15+v2_xpos/15
        if dist <7:
            reward = -1
        
        if dist <5:
            terminated = 1
            reward = -1
       
        if self.stepNum *self.deltaT >20:
            terminated = 1

        info = {"runState":self.runState,"stepNum":self.stepNum,"Time":self.stepNum*self.deltaT}
        return np.array(self.envState), reward, terminated,info

    def getState(self):
        tmp = np.array(self.envState)
        return tmp


    def close(self):
        
        if self.runState == "running" or self.runState == "close":
            self.runState = "close"
            info = {"runState":self.envState,"stepNum":self.stepNum,"Time":self.stepNum*self.deltaT}
       
        return np.array(self.envState), info

    def reset(self):


        self.stepNum = 0
        self.runstate = "init"
        self.envState = self.envStateInit

        v1_xpos,v1_xvel,v1_ypos,v1_yvel,v2_xpos,v2_xvel,v2_ypos,v2_yvel = self.envState

        self.v1_xpos0 = v1_xpos
        self.v1_ypos0 = v1_ypos
        self.v2_xpos0 = v2_xpos 
        self.v2_ypos0 = v2_ypos

        info = {"runState":self.envState,"stepNum":self.stepNum,"Time":self.stepNum*self.deltaT}
        #print('states:{}'.format(np.round(self.envState,2)))
        return np.array(self.envState)

    def render(self,mode):
        pass
        
  




if __name__ == "__main__":
    # execute only if run as a script
    myEnv = Veh2CrashEnv()
    envInit = [0,5.56,0,0,13,5.56/2,-6.5,0]
    myEnv.setInitEnv(envInit)
   
    action = 3
    recordStates = []
    recordReward = []
    
    recordStates.append(envInit)
    recordReward.append(0)
    while True:
        state, reward, terminated,info = myEnv.step(action)
        recordStates.append(list(state)) 
        recordReward.append(reward) 
        
        if terminated  ==  1:
                break
                
    recordStates1 = np.array(recordStates)
    print(recordStates1.shape)
    print(recordStates1[0:2])
    
    x1 = recordStates1[:,0]
    y1 = recordStates1[:,3]
    x2 = recordStates1[:,4]
    y2 = recordStates1[:,6]
    
   
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
    
    with imageio.get_writer('./veh2CrashEnv.gif', mode='I',fps =2) as writer:

        for step in range(x1.shape[0]):
            print(step)
            x1pts = x1[step]
            y1pts = y1[step]
            x2pts = x2[step]
            y2pts = y2[step]
            distpts = dist[step]
            plt.plot(x1pts,y1pts,'rx', label='v1')
            plt.plot(x2pts,y2pts,'b.', label='v2')
            plt.plot(x1pts,distpts,'y.', label='v1')
            plt.savefig('./tmp.jpg')
            img_array = imageio.v2.imread('./tmp.jpg')
            #print(img_array.shape)
            #frames.append(img_array)
            writer.append_data(img_array)
 
 
        