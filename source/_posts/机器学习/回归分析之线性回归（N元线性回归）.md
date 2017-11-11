---
title: 回归分析之线性回归（N元线性回归）
date: 2017-09-29 16:45:14
tags: [回归分析, 二元线性回归,多元线性回归]
categories: 技术篇
---

在上一篇文章中我们介绍了 [ 回归分析之理论篇][1]，在其中我们有聊到线性回归和非线性回归，包括广义线性回归，这一篇文章我们来聊下回归分析中的线性回归。

<!--More-->
# 一元线性回归
预测房价：
输入编号	| 平方米	| 价格
-|-|-
1 |	150 |	6450
2	| 200	| 7450
3|	250	|8450
4|	300	|9450
5|	350	|11450
6|	400	|15450
7|	600|	18450

针对上边这种一元数据来讲，我们可以构建的一元线性回归函数为
$$
H(x) = k*x + b
$$
其中H(x)为平方米价格表，k是一元回归系数，b为常数。最小二乘法的公式：
$$
k =\frac{ \sum_{1}^{n} (x_{i} - \bar{x} )(y_{i} - \bar{y}) } { \sum_{1}^{n}(x_{i}-\bar{x})^{2} }
$$
自己使用python代码实现为：
```
def leastsq(x,y):
    """
    x,y分别是要拟合的数据的自变量列表和因变量列表
    """
    meanX = sum(x) * 1.0 / len(x)      # 求x的平均值
    meanY = sum(y) * 1.0 / len(y)     # 求y的平均值

    xSum = 0.0
    ySum = 0.0

    for i in range(len(x)):
        xSum += (x[i] - meanX) * (y[i] - meanY)
        ySum += (x[i] - meanX) ** 2

    k = ySum/xSum
    b = ySum - k * meanX

    return k,b
```

使用python的scipy包进行计算:
```
leastsq(func, x0, args=(), Dfun=None, full_output=0, col_deriv=0, ftol=1.49012e-08, xtol=1.49012e-08, gtol=0.0, maxfev=0, epsfcn=None, factor=100, diag=None)

from scipy.optimize import leastsq
import numpy as np

def fun(p, x):
    """
    定义想要拟合的函数
    """
    k,b = p    # 从参数p获得拟合的参数
    return k*x + b

def err(p, x, y):
    return fun(p,x) - y

#定义起始的参数 即从 y = 1*x+1 开始，其实这个值可以随便设，只不过会影响到找到最优解的时间
p0 = [1,1]

#将list类型转换为 numpy.ndarray 类型，最初我直接使用
#list 类型,结果 leastsq函数报错，后来在别的blog上看到了，原来要将类型转
#换为numpy的类型

x1 = np.array([150,200,250,300,350,400,600])
y1 = np.array([6450,7450,8450,9450,11450,15450,18450])

xishu = leastsq(err, p0, args=(x1,y1))

print xishu[0]

```

当然python的leastsq函数不仅仅局限于一元一次的应用，也可以应用到一元二次，二元二次，多元多次等，具体可以看下这篇博客：http://www.cnblogs.com/NanShan2016/p/5493429.html

# 多元线性回归
总之：我们可以用python leastsq函数解决几乎所有的线性回归的问题了，比如说
$$y = a * x^2 + b * x + c$$
$$y = a * x_1^2 + b * x_1 + c * x_2 + d$$
$$y = a * x_1^3 + b * x_1^2 + c * x_1 + d$$
在使用时只需把参数列表和 fun 函数中的return 换一下，拿以下函数举例
$$y = a * x_1^2 + b * x_1 + c * x_2 + d$$

对应的python 代码是：
```
from scipy.optimize import leastsq
import numpy as np


def fun(p, x1, x2):
    """
    定义想要拟合的函数
    """
    a,b,c,d = p    # 从参数p获得拟合的参数
    return a * (x1**2) + b * x1 + c * x2 + d

def err(p, x1, x2, y):
    return fun(p,x1,x2) - y

#定义起始的参数 即从 y = 1*x+1 开始，其实这个值可以随便设，只不过会影响到找到最优解的时间
p0 = [1,1,1,1]

#将list类型转换为 numpy.ndarray 类型，最初我直接使用
#list 类型,结果 leastsq函数报错，后来在别的blog上看到了，原来要将类型转
#换为numpy的类型

x1 = np.array([150,200,250,300,350,400,600])    # 面积
x2 = np.array([4,2,7,9,12,14,15])               # 楼层
y1 = np.array([6450,7450,8450,9450,11450,15450,18450])   # 价格/平方米

xishu = leastsq(err, p0, args=(x1,x2,y1))

print xishu[0]
```

# sklearn中的线性回归应用
## 普通最小二乘回归
这里我们使用的是sklearn中的linear_model来模拟$$y=a * x_1 + b * x_2 + c$$

```
In [1]: from sklearn.linear_model import LinearRegression

In [2]: linreg = LinearRegression()

In [3]: linreg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])

In [4]: linreg.coef_
Out[4]: array([ 0.5,  0.5])

In [5]: linreg.intercept_
Out[5]: 1.1102230246251565e-16

In [6]: linreg.predict([4,4])
Out[6]: array([ 4.])

In [7]: zip(["x1","x2"], linreg.coef_)
Out[7]: [('x1', 0.5), ('x2', 0.49999999999999989)]
```
所以可得$$ y = 0.5 * x_1 + 0.5 * x_2 + 1.11e-16$$

linreg.coef_  为系数 a,b

linreg.intercept_ 为截距 c

缺点：因为系数矩阵x与它的转置矩阵相乘得到的矩阵不能求逆，导致最小二乘法得到的回归系数不稳定，方差很大。


## 多项式回归：基函数扩展线性模型
机器学习中一种常见的模式是使用线性模型训练数据的非线性函数。这种方法保持了一般快速的线性方法的性能，同时允许它们适应更广泛的数据范围。

例如，可以通过构造系数的多项式特征来扩展一个简单的线性回归。在标准线性回归的情况下，你可能有一个类似于二维数据的模型：
$$
y(w,x) = w_{0} + w_{1} x_{1} + w_{2} x_{2}
$$

如果我们想把抛物面拟合成数据而不是平面，我们可以结合二阶多项式的特征，使模型看起来像这样:
$$
y(w,x) = w_{0} + w_{1} x_{1} + w_{2} x_{2} + w_{3} x_{1}x_{2} + w_{4} x_{1}^2 + w_{5} x_{2}^2
$$

我们发现，这仍然是一个线性模型，想象着创建一个新变量：
$$
z = [x_{1},x_{2},x_{1} x_{2},x_{1}^2,x_{2}^2]
$$

可以把线性回归模型写成下边这种形式：
$$
y(w,x) = w_{0} + w_{1} z_{1} + w_{2} z_{2} + w_{3} z_{3} + w_{4} z_{4} + w_{5} z_{5}
$$
我们看到，所得的多项式回归与我们上面所考虑的线性模型相同（即模型在W中是线性的），可以用同样的方法来求解。通过考虑在用这些基函数建立的高维空间中的线性拟合，该模型具有灵活性，可以适应更广泛的数据范围。

使用如下代码，将二维数据进行二元转换,转换规则为：
$$
[x_1, x_2] => [1, x_1, x_2, x_1^2, x_1 x_2, x_2^2]
$$

```
In [15]: from sklearn.preprocessing import PolynomialFeatures

In [16]: import numpy as np

In [17]: X = np.arange(6).reshape(3,2)

In [18]: X
Out[18]: 
array([[0, 1],
       [2, 3],
       [4, 5]])

In [19]: poly = PolynomialFeatures(degree=2)

In [20]: poly.fit_transform(X)
Out[20]: 
array([[  1.,   0.,   1.,   0.,   0.,   1.],
       [  1.,   2.,   3.,   4.,   6.,   9.],
       [  1.,   4.,   5.,  16.,  20.,  25.]])
```

验证：
```
In [38]: from sklearn.preprocessing import PolynomialFeatures

In [39]: from sklearn.linear_model import LinearRegression

In [40]: from sklearn.pipeline import Pipeline

In [41]: import numpy as np

In [42]: 

In [42]: model = Pipeline( [ ("poly",PolynomialFeatures(degree=3)),("linear",LinearRegression(fit_intercept=False)) ] )

In [43]: model
Out[43]: Pipeline(steps=[('poly', PolynomialFeatures(degree=3, include_bias=True, interaction_only=False)), ('linear', LinearRegression(copy_X=True, fit_intercept=False, n_jobs=1, normalize=False))])

In [44]: x = np.arange(5)

In [45]: y = 3 - 2 * x + x ** 2 - x ** 3

In [46]: y
Out[46]: array([  3,   1,  -5, -21, -53])

In [47]: model = model.fit(x[:,np.newaxis],y)

In [48]: model.named_steps['linear'].coef_
Out[48]: array([ 3., -2.,  1., -1.])
```
我们可以看出最后求出的参数和一元三次方程是一致的。

这里如果把degree改为2，y的方程也换一下，结果也是一致的
```
In [51]: from sklearn.linear_model import LinearRegression

In [52]: from sklearn.preprocessing import PolynomialFeatures

In [53]: from sklearn.pipeline import Pipeline

In [54]: import numpy as np

In [55]: model = Pipeline( [ ("poly",PolynomialFeatures(degree=2)),("linear",LinearRegression(fit_intercept=False)) ] )

In [56]: x = np.arange(5)

In [57]: y = 3 + 2 * x + x ** 2

In [58]: model = model.fit(x[:, np.newaxis], y)

In [59]: model.named_steps['linear'].coef_
Out[59]: array([ 3., 2.,  1.])
```


## 线性回归的评测
在[上一篇文章](http://note.youdao.com/)中我们聊到了回归模型的评测方法，解下来我们详细聊聊如何来评价一个回归模型的好坏。

这里我们定义预测值和真实值分别为：
```
true = [10, 5, 3, 2]
pred = [9, 5, 5, 3]
```

1: 平均绝对误差（Mean Absolute Error, MAE）
$$
\frac{1}{N}(\sum_{1}^{n} |y_i - \bar{y}|)
$$

2: 均方误差（Mean Squared Error, MSE）
$$
\frac{1}{N}\sum_{1}^{n}(y_i - \bar{y})^2
$$

3: 均方根误差（Root Mean Squared Error, RMSE）
$$
\frac{1}{N} \sqrt{ \sum_{1}^{n}(y_i - \bar{y})^2 }
$$

```
In [80]: from sklearn import metrics

In [81]: import numpy as np

In [82]: true = [10, 5, 3, 2]

In [83]: pred = [9, 5, 5, 3]

In [84]: print("MAE: ", metrics.mean_absolute_error(true,pred))
('MAE: ', 1.0)

In [85]: print("MAE By Hand: ", (1+0+2+1)/4.)
('MAE By Hand: ', 1.0)

In [86]: print("MSE: ", metrics.mean_squared_error(true,pred))
('MSE: ', 1.5)

In [87]: print("MSE By Hand: ", (1 ** 2 + 0 ** 2 + 2 ** 2 + 1 ** 2 ) / 4.)
('MSE By Hand: ', 1.5)

In [88]: print("RMSE: ", np.sqrt(metrics.mean_squared_error(true,pred)))
('RMSE: ', 1.2247448713915889)

In [89]: print("RMSE By Hand: ", np.sqrt((1 ** 2 + 0 ** 2 + 2 ** 2 + 1 ** 2 ) / 4.))
('RMSE By Hand: ', 1.2247448713915889)
```

---
# 总结
线性回归在现实中还是可以解决很多问题的，但是并不是万能的，后续我会继续整理逻辑回归，岭回归等相关回归的知识，如果你感觉有用，欢迎分享！

  [1]: http://blog.csdn.net/gamer_gyt/article/details/78008144