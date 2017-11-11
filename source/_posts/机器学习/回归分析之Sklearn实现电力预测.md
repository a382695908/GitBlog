---
title: 回归分析之Sklearn实现电力预测
date: 2017-11-07 13:39:15
tags: [回归分析,sklearn,pandas,交叉验证]
categories: 技术篇
---

参考原文：http://www.cnblogs.com/pinard/p/6016029.html
这里进行了手动实现，增强记忆。
<!--More-->
# 1：数据集介绍
使用的数据是UCI大学公开的机器学习数据

数据的介绍在这： http://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant

数据的下载地址在这：http://archive.ics.uci.edu/ml/machine-learning-databases/00294/

里面是一个循环发电场的数据，共有9568个样本数据，每个数据有5列，分别是:AT（温度）, V（压力）, AP（湿度）, RH（压强）, PE（输出电力)。我们不用纠结于每项具体的意思。

我们的问题是得到一个线性的关系，对应PE是样本输出，而AT/V/AP/RH这4个是样本特征， 机器学习的目的就是得到一个线性回归模型，即:

$$
PE = \theta _{0} + \theta _{0} * AT + \theta _{0} * V +\theta _{0} * AP +\theta _{0}*RH
$$

而需要学习的，就是θ0,θ1,θ2,θ3,θ4这5个参数。

---
# 2：准备数据
下载源数据之后，解压会得到一个xlsx的文件，打开另存为csv文件，数据已经整理好，没有非法数据，但是数据并没有进行归一化，不过这里我们可以使用sklearn来帮我处理

sklearn的归一化处理参考：http://blog.csdn.net/gamer_gyt/article/details/77761884

---

# 3：使用pandas来进行数据的读取

```
import pandas as pd
# pandas 读取数据
data = pd.read_csv("Folds5x2_pp.csv")
data.head()
```
然后会看到如下结果，说明数据读取成功：

```
	AT	V	AP	RH	PE
0	8.34	40.77	1010.84	90.01	480.48
1	23.64	58.49	1011.40	74.20	445.75
2	29.74	56.90	1007.15	41.91	438.76
3	19.07	49.69	1007.22	76.79	453.09
4	11.80	40.66	1017.13	97.20	464.43
```

---

# 4：准备运行算法的数据
```
X = data[["AT","V","AP","RH"]]
print X.shape
y = data[["PE"]]
print y.shape
```

```
(9568, 4)
(9568, 1)
```

说明有9658条数据，其中"AT","V","AP","RH" 四列作为样本特征，"PE"列作为样本输出。

---
# 5：划分训练集和测试集

```
from sklearn.cross_validation import train_test_split

# 划分训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)
print X_train.shape
print y_train.shape
print X_test.shape
print y_test.shape
```
```
(7176, 4)
(7176, 1)
(2392, 4)
(2392, 1)
```
75%的数据被划分为训练集，25的数据划分为测试集。

---
# 6：运行sklearn 线性模型
```
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
linreg.fit(X_train,y_train)

# 训练模型完毕，查看结果
print linreg.intercept_
print linreg.coef_
```

```
[ 447.06297099]
[[-1.97376045 -0.23229086  0.0693515  -0.15806957]]
```

即我们得到的模型结果为：
$$
PE = 447.06297099 - 1.97376045*AT - 0.23229086*V + 0.0693515*AP -0.15806957*RH
$$

---
# 7：模型评价
我们需要评价模型的好坏，通常对于线性回归来讲，我么一般使用均方差（MSE，Mean Squared Error）或者均方根差（RMSE，Root Mean Squared Error）来评价模型的好坏

```
y_pred = linreg.predict(X_test)
from sklearn import metrics

# 使用sklearn来计算mse和Rmse
print "MSE:",metrics.mean_squared_error(y_test, y_pred)
print "RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred))
```

```
MSE: 20.0804012021
RMSE: 4.48111606657
```
得到了MSE或者RMSE，如果我们用其他方法得到了不同的系数，需要选择模型时，就用MSE小的时候对应的参数。

---
# 8：交叉验证

我们可以通过交叉验证来持续优化模型，代码如下，我们采用10折交叉验证，即cross_val_predict中的cv参数为10：

```
# 交叉验证
from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(linreg,X,y,cv=10)
print "MSE:",metrics.mean_squared_error(y, predicted)
print "RMSE:",np.sqrt(metrics.mean_squared_error(y, predicted))
```

```
MSE: 20.7955974619
RMSE: 4.56021901469
```

可以看出，采用交叉验证模型的MSE比第6节的大，主要原因是我们这里是对所有折的样本做测试集对应的预测值的MSE，而第6节仅仅对25%的测试集做了MSE。两者的先决条件并不同。

---
# 9：画图查看结果
```
# 画图查看结果
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
```
![这里写图片描述](http://img.blog.csdn.net/20171107133222238?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvR2FtZXJfZ3l0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
