'''
测试4，是否tensorflow-keras+自定义gym-Veh2Crash+GPU(环境tensor23py36gpu)能用
参考
https://blog.csdn.net/2401_84495872/article/details/139591541
'''

import gym # 导入Gym库，用于开发和比较强化学习算法
import numpy as np # 导入NumPy库，用于进行科学计算
import tensorflow as tf # 导入TensorFlow库
from tensorflow import keras # 导入keras模块，这是一个高级神经网络API
from tensorflow.keras import layers # 导入keras中的layers模块，用于创建神经网络层
import imageio

import matplotlib.pyplot as plt

seed = 42 # 设定随机种子，用于复现实验结果
gamma = 0.99 # 定义折扣率，用于计算未来奖励的现值
max_steps_per_episode = 10000 # 设定每个 episode 的最大步数
env = gym.make("Veh2CrashEnv-v1") # 创建 CartPole-v0 环境实例

eps = np.finfo(np.float32).eps.item() # 获取 float32 数据类型的误差最小值 epsilon 

#########################
num_inputs = 8 # 状态空间的维度，即输入层的节点数
num_actions = 4 # 行为空间的维度，即输出层的节点数
num_hidden = 128 # 隐藏层的节点数

inputs = layers.Input(shape=(num_inputs,)) # 创建输入层，指定输入的形状
common = layers.Dense(num_hidden, activation="relu")(inputs) # 创建一个全连接层，包含num_hidden 个神经元，使用 ReLU 作为激活函数
commonEnd = layers.Dense(num_hidden, activation="relu")(common) # 创建一个全连接层，包含num_hidden 个神经元，使用 ReLU 作为激活函数
action = layers.Dense(num_actions, activation="softmax")(commonEnd) # 创建一个全连接层，包含 num_actions 个神经元，使用 softmax 作为激活函数
critic = layers.Dense(1)(commonEnd) # 创建一个全连接层，包含1个神经元

model = keras.Model(inputs=inputs, outputs=[action, critic]) # 创建一个 Keras 模型，包含输入层、共享的隐藏层和两个输出层


#########################

    
action_probs =np.array([0.25,0.25,0.25,0.25])    
env.reset() 
frames = []
for t in range(max_steps_per_episode):
    

    observation, reward, done, info, _ = env.step(np.random.choice(num_actions,p=np.squeeze(action_probs)))
    if done:
        break


        
        

#########################

optimizer = keras.optimizers.Adam(learning_rate=0.01) # 创建 Adam 优化器实例，设置学习率为 0.01
huber_loss = keras.losses.Huber() # 创建损失函数实例
action_probs_history = [] # 创建一个列表，用于保存 action 网络在每个步骤中采取各个行动的概率
critic_value_history = [] # 创建一个列表，用于保存 critic 网络在每个步骤中对应的值
rewards_history = [] # 创建一个列表，用于保存每个步骤的奖励值
running_reward = 0 # 初始化运行过程中的每轮奖励
episode_count = 0 # 初始化 episode 计数器

max_reward_model = model
max_reward  = 0
while True:  
    stateInit= env.reset()  # 新一轮游戏开始，重置环境
    state = stateInit
    episode_reward = 0  # 记录本轮游戏的总奖励值
   
    with tf.GradientTape() as tape:  # 构建 GradientTape 用于计算梯度
        for timestep in range(1, max_steps_per_episode): # 本轮游戏如果一切正常会进行 max_steps_per_episode 步
            
            state = tf.convert_to_tensor(state)  # 将状态转换为张量
            
            state = tf.expand_dims(state, 0)  # 扩展维度，以适应模型的输入形状
            
            action_probs, critic_value = model(state)  # 前向传播，得到 action 网络输出的动作空间的概率分布，和 critic 网络预测的奖励值
            critic_value_history.append(critic_value[0, 0])  # 将上面 critic 预测的奖励值记录在 critic_value_history 列表中

            action = np.random.choice(num_actions, p=np.squeeze(action_probs))  # 依据概率分布抽样某个动作，当然了某个动作概率越大越容易被抽中，同时也保留了一定的随机性
            action_probs_history.append(tf.math.log(action_probs[0, action]))  # 将使用该动作的对数概率值记录在 action_probs_history 列表中
        
            state, reward, done, info, _ = env.step(action)  # 游戏环境使用选中的动作去执行，得到下一个游戏状态、奖励、是否终止和其他信息
            rewards_history.append(reward)  # 将该时刻的奖励记录在 rewards_history 列表中
            episode_reward += reward  # 累加本轮游戏的总奖励值

            if done:  # 如果到达终止状态，则结束循环
                break

        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward  # 计算平均奖励

        returns = []  # 存储折扣回报
        discounted_sum = 0
        for r in rewards_history[::-1]:  # 从后往前遍历奖励的历史值
            discounted_sum = r + gamma * discounted_sum  # 计算折扣回报
            returns.insert(0, discounted_sum)  # 将折扣回报插入列表的开头，最后形成的还是从前往后的折扣奖励列表

        returns = np.array(returns)  # 将折扣回报转换为数组
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)  # 归一化折扣回报
        returns = returns.tolist()  # 将折扣回报转换为列表形式

        history = zip(action_probs_history, critic_value_history, returns)  # 将三个列表进行 zip 压缩
        actor_losses = []  # 存储 action 网络的损失
        critic_losses = []  # 存储 critic 网络的损失

        for log_prob, value, ret in history:
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # 计算 actor 的损失函数

            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0)) # 计算 critic 的损失函数
            )

        loss_value = sum(actor_losses) + sum(critic_losses) # 计算总损失函数
        grads = tape.gradient(loss_value, model.trainable_variables) # 计算梯度
        optimizer.apply_gradients(zip(grads, model.trainable_variables)) # 更新模型参数

        action_probs_history.clear() # 清空之前的历史记录
        critic_value_history.clear() # 清空之前的历史记录
        rewards_history.clear() # 清空之前的历史记录

    episode_count += 1 # 当一轮游戏结束时， episode 加一
    if episode_count % 10 == 0: # 每训练 10 个 episode ，输出当前的平均奖励
        template = "在第 {} 轮游戏中获得奖励: {:.2f} 分"
        print(template.format(episode_count, running_reward))

    if episode_count > 1000:  # 
        print("episode_count > 1000 ，训练结束")
        break
        
    if max(max_reward,running_reward) > max_reward:
        max_reward_model = model
        max_reward = running_reward
        print('max_reward_model:{},max_reward:{:.3f}'.format(episode_count,max_reward))

        
###################

stateInit= env.reset()  # 新一轮游戏开始，重置环境
state = stateInit
recordStates = []
recordReward = []

episode_reward = 0
for t in range(max_steps_per_episode):
    state = tf.convert_to_tensor(state)  # 将状态转换为张量
    state = tf.expand_dims(state, 0)  # 扩展维度，以适应模型的输入形状
    
    action_probs, _ = max_reward_model(state)
    action = np.random.choice(num_actions, p=np.squeeze(action_probs))
    state, reward, done,info, _ = env.step(action)
    recordStates.append(state)
    recordReward.append(reward)
    
    episode_reward += reward 
    print('actions:',action)
    print('states:{}'.format(np.round(state,2)))
    print('reward:%.2f,episode_reward:%.3f' %(reward,episode_reward))
    if done:
        break

recordStates1 = np.array(recordStates)
x1 = recordStates1[:,0]
y1 = recordStates1[:,3]
x2 = recordStates1[:,4]
y2 = recordStates1[:,6]
dist = (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2) 
dist = np.sqrt(dist)
    
with imageio.get_writer('./veh2CrashEnv_tensor23py36Gpu.gif', mode='I',fps =2) as writer:

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
        plt.savefig('./tmp_tensor23py36Gpu.jpg')
        img_array = imageio.imread('./tmp_tensor23py36Gpu.jpg')
        #print(img_array.shape)
        #frames.append(img_array)
        writer.append_data(img_array)
        
print('ended')



