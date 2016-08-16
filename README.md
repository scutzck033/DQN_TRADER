# DQN_TRAINER

##在运行项目之前请先转至DQN目录，运行命令：
##./install_dependencies.sh
##安装相关的torch库

1.运行：
A.运行train_sin_data.lua文件，对模型进行训练，训练完成后的模型会以文件形式存在dqn目录下。
命令： ./run_gpu breakout (第一次运行需要修改run_gpu中最后一行train_sin_data.lua的路径)
B.运行test_sin_data.lua文件，对训练的模型进行测试
命令：./test_gpu breakout 模型名 (第一次运行需要修改test_gpu中最后一行test_sin_data.lua的路径)

2.train_sin_data中，65-170行为自己写的代码(具体参数设置请看代码注释)
--测试模型的时候，可以改变test_sin_data.lua第70行的参数loop
--训练模型的次数由run_gpu中的steps决定

3.重要代码区间：
getSinValue(sin_index,dt)--获取第i个点的sin值
Step(action)--根据action返回对应的状态,reward,terminal
getState()--返回初始状态，初始reward,初始terminal

4.结果可视化部分代码(见test_sin_data.lua文件229-240行)：
gnuplot.pngfigure('/home/qxm/result/plot.png')
--下面代码依次画出：
--价格走势
--对应所采取的动作
--总回报(own=total_reward/max)，除以系数max是为了结果展示方便
gnuplot.plot({torch.Tensor(sindex), torch.Tensor(price)},{torch.Tensor(action_index), torch.Tensor(shb)} , {torch.Tensor(action_index),torch.Tensor(own)})

print(#sindex)
print(#price)
print(#shb)
print(#own)
gnuplot.plotflush()
