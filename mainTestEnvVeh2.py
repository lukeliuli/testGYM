from Veh2CrashEnv import Veh2CrashEnv
import numpy as np
import matplotlib.pyplot as plt
import imageio

if __name__ == "__main__":
    # execute only if run as a script
    myEnv = Veh2CrashEnv()
    envInit = [0,5.56,0,0,13,5.56,-6.5,0]
    myEnv.setInitEnv([0,5.56,0,0,13,5.56/2,-6.5,0])
   
    action = [1,1]
    recordStates = []
    recordReward = []
    
    recordStates.append([0,5.56,0,0,13,5.56,-6.5,0])
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
 
 
        