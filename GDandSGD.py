import numpy as np
import matplotlib.pyplot as plt
from random import choice


### 获取两位数据集和，共计2000个
def getData(sampleNumber):

	xs = np.random.uniform(0, 1, sampleNumber)
	ys = np.random.uniform(0, 1, sampleNumber)
	data = list(zip(xs, ys) );

	return data;

### 梯度下降算法
### xi yi 表示起始位点，learning_rate表示学习率 epoch 表示最大循环迭代次数 data表示输入的二维数据集
### sampleNumber 表示样本数目
def GradientDescent(xi, yi, learning_rate, epoch, data, sampleNumber):

	PointList = [];
	PointList.append((xi, yi)) ### 表示梯度下降过程中经过的点

	for i in range(epoch):

		total = 0;
		gradientX = 0;
		gradientY = 0;

		for j in range(sampleNumber):

			total += (xi - data[j][0]) ** 2 + (yi - data[j][1]) ** 2; ### 损失函数计算，及梯度计算
			gradientX += (xi - data[j][0]);
			gradientY += (yi - data[j][1]);

		print("epoch " + str(i)+ " cost : " + str(total / 2.0 / sampleNumber)); #打印损失函数

		xi = xi - learning_rate * gradientX / sampleNumber;
		yi = yi - learning_rate * gradientY / sampleNumber; ### xi yi 更新

		PointList.append((xi, yi))

	return PointList### 表示梯度下降过程中经过的点

### 随机梯度下降算法
def StochasticGradientDescent(xi, yi, learning_rate, epoch, data, sampleNumber):

	PointList = [];
	PointList.append((xi, yi))

	for i in range(epoch):

		total = 0;

		pointPre = choice(data); ###随机选择数据集中的一个点进行模型更新
		gradientX = (xi - pointPre[0]);
		gradientY = (yi - pointPre[1]);

		for j in range(sampleNumber):

			total += (xi - data[j][0]) ** 2 + (yi - data[j][1]) ** 2;

		print("epoch " + str(i)+ " cost : " + str(total / 2.0 / sampleNumber)); #打印损失函数

		xi = xi - learning_rate * gradientX;
		yi = yi - learning_rate * gradientY;### xi yi 更新

		PointList.append((xi, yi))

	return PointList### 表示梯度下降过程中经过的点


def main():
	data = getData(2000);
	PointList = GradientDescent(0, 1, 0.01, 1000, data, 2000) ### 梯度下降
	#PointList = StochasticGradientDescent(0, 1, 0.01, 1000, data, 2000) ### 随机梯度下降
	xs = [a[0] for a in data]
	ys = [b[1] for b in data]

	plt.scatter(xs, ys, color = 'red')

	pointXS = [x[0] for x in PointList]
	pointYS = [y[1] for y in PointList]

	plt.plot(pointXS, pointYS, color = 'blue') ### 画图

	plt.show()

main()
