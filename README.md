## testGYM
1. testGYM.ipynb,用于测试GYM和加强算法,所有程序的主入口，实现+环境的细节描述
2. testTensorforce1.py,testTensorforce2.py,testTensorforce3.py(主要)的测试tensorforce算法，conda环境为tensorforcc
3. test4_GymVeh2Gpu.py 为tensorflow-keras+自定义gym-Veh2Crash+GPU能用，conda环境tensor23py36gpu
4. test5_torchGymVeh2Gpu.py，为pytorch+stable_baselines3+自定义gym-Veh2Crash+GPU，conda,conda环境torch24PY38Gpu
5. Veh2CrashEnv+????????.py,自定义GYM环境的主程序文件
6. gym_example 为test5_torchGymVeh2Gpu的gym环境的主文件
6. 现阶段最好的最好的结果test5_torchGymVeh2Gpu，模型文件为a2c_Veh2CrashEnv-v1-best.zip，结果文件为veh2CrashEnv_pytorchGPU-best.gif

## 注意1
1. 大量的自有代码已经设计加强算法
+ Hierarchical-Actor-Critc-HAC-
+ learningOpenAI
+ myRLExamples1，matlab版本
+ Reinforcement-learning-with-tensorflow
+ sumoLinuxPythonRL1，重要建立了自己的环境

## 注意2 本项目，用于GYM下面的微分博弈
1. 与differentialEquSamples相关联
2. 采用testGYM 运行在tensor23py36gpu的环境中，其中测试tensorforce运行在tensorforce环境中
3. 已经下载了tensorforce的代码，需要pip安装相关环境和支持包 
4. 测试stable_baselines3，需要允许在rl_sb3环境中，参考https://zhuanlan.zhihu.com/p/659089157

# 注意3，收集

* gym: OpenAI提供的一个用于开发和比较强化学习算法的工具包，可以用于各种任务，包括但不限于环境交互、强化学习等。

* TensorForce: 一个用于强化学习的库，它提供了基于深度强化学习的算法，并且可以与TensorFlow集成。

* OpenAI Gym: 一个用于开发和比较强化学习算法的工具包，可以用于各种任务，包括但不限于环境交互、强化学习等。

* Baselines: 一个提供各种强化学习算法的库，包括DQN、PPO、A2C等。

* Dopamine: 一个用于研究复杂机器学习系统的开源库，主要关注强化学习。

* RLlib: 用于分布式强化学习的工具包，可以在大型集群上训练A3C算法。

* Coach: 由百度提供的，用于强化学习研究的开源框架。

* 强化学习算法库 stable_baselines3
https://zhuanlan.zhihu.com/p/659089157

* openAI GYM优化库，gymnasium.env