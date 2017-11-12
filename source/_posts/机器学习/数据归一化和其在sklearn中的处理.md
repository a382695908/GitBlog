---
title: 数据归一化和其在sklearn中的处理
date: 2017-09-01 11:33:50
tags: [数据归一化,sklearn]
categories: 技术篇
---

# 一：数据归一化

数据归一化（标准化）处理是数据挖掘的一项基础工作，不同评价指标往往具有不同的量纲和量纲单位，这样的情况会影响到数据分析的结果，为了消除指标之间的量纲影响，需要进行数据标准化处理，以解决数据指标之间的可比性。原始数据经过数据标准化处理后，各指标处于同一数量级，适合进行综合对比评价。
<!--More-->
归一化方法有两种形式，一种是把数变为（0，1）之间的小数，一种是把有量纲表达式变为无量纲表达式。在机器学习中我们更关注的把数据变到0～1之间，接下来我们讨论的也是第一种形式。

## 1）min-max标准化
min-max标准化也叫做离差标准化，是对原始数据的线性变换，使结果落到[0,1]区间，其对应的数学公式如下：

$$
X_{scale} = \frac{x-min}{max-min}
$$

对应的python实现为
```
# x为数据 比如说 [1,2,1,3,2,4,1]
def Normalization(x):
    return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]
```

如果要将数据转换到[-1,1]之间，可以修改其数学公式为：

$$
X_{scale} = \frac{x-x_{mean}}{max-min}
$$
x_mean 表示平均值。

对应的python实现为
```
import numpy as np

# x为数据 比如说 [1,2,1,3,2,4,1]
def Normalization(x):
    return [(float(i)-np.mean(x))/float(max(x)-min(x)) for i in x]
```

其中max为样本数据的最大值，min为样本数据的最小值。这种方法有个缺陷就是当有新数据加入时，可能导致max和min的变化，需要重新定义。

该标准化方法有一个缺点就是，如果数据中有一些偏离正常数据的异常点，就会导致标准化结果的不准确性。比如说一个公司员工（A，B，C，D）的薪水为6k,8k,7k,10w,这种情况下进行归一化对每个员工来讲都是不合理的。

当然还有一些其他的办法也能实现数据的标准化。

## 2）z-score标准化
z-score标准化也叫标准差标准化，代表的是分值偏离均值的程度，经过处理的数据符合标准正态分布，即均值为0，标准差为1。其转化函数为

$$
X_{scale} = \frac{x-\mu }{\sigma }
$$

其中μ为所有样本数据的均值，σ为所有样本数据的标准差。

其对应的python实现为：
```
import numpy as np

#x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
def z_score(x):
    return (x - np.mean(x) )/np.std(x, ddof = 1)
```
z-score标准化方法同样对于离群异常值的影响。接下来看一种改进的z-score标准化方法。

## 3）改进的z-score标准化

将标准分公式中的均值改为中位数，将标准差改为绝对偏差。

$$
X_{scale} = \frac{x-x_{center} }{\sigma_{1} }$$
中位数是指将所有数据进行排序，取中间的那个值，如数据量是偶数，则取中间两个数据的平均值。

σ1为所有样本数据的绝对偏差,其计算公式为：
$$
\frac{1}{N} \sum_{1}^{n}|x_{i} - x_{center}|
$$

----
# 二：sklearn中的归一化

sklearn.preprocessing 提供了一些实用的函数 用来处理数据的维度，以供算法使用。

## 1）均值-标准差缩放

即我们上边对应的z-score标准化。
在sklearn的学习中，数据集的标准化是很多机器学习模型算法的常见要求。如果个别特征看起来不是很符合正态分布，那么他们可能为表现不好。

实际上，我们经常忽略分布的形状，只是通过减去整组数据的平均值，使之更靠近数据中心分布，然后通过将非连续数特征除以其标准偏差进行分类。


例如，用于学习算法（例如支持向量机的RBF内核或线性模型的l1和l2正则化器）的目标函数中使用的许多元素假设所有特征都以零为中心并且具有相同顺序的方差。如果特征的方差大于其他数量级，则可能主导目标函数，使估计器无法按预期正确地学习其他特征。

例子：
```
>>> from sklearn import preprocessing
>>> import numpy as np
>>> X_train = np.array([[ 1., -1.,  2.],
...                     [ 2.,  0.,  0.],
...                     [ 0.,  1., -1.]])
>>> X_scaled = preprocessing.scale(X_train)
>>> X_scaled
array([[ 0.        , -1.22474487,  1.33630621],
       [ 1.22474487,  0.        , -0.26726124],
       [-1.22474487,  1.22474487, -1.06904497]])
```
标准化后的数据符合标准正太分布
```
>>> X_scaled.mean(axis=0)
array([ 0.,  0.,  0.])
>>> X_scaled.std(axis=0)
array([ 1.,  1.,  1.])
```

预处理模块还提供了一个实用程序级StandardScaler，它实现了Transformer API来计算训练集上的平均值和标准偏差，以便能够稍后在测试集上重新应用相同的变换。
```
>>> scaler = preprocessing.StandardScaler().fit(X_train)
>>> scaler
StandardScaler(copy=True, with_mean=True, with_std=True)
>>> scaler.mean_
array([ 1.        ,  0.        ,  0.33333333])
>>> scaler.scale_
array([ 0.81649658,  0.81649658,  1.24721913])
>>> scaler.transform(X_train)
array([[ 0.        , -1.22474487,  1.33630621],
       [ 1.22474487,  0.        , -0.26726124],
       [-1.22474487,  1.22474487, -1.06904497]])
```

使用转换器可以对新数据进行转换
```
>>> X_test = [[-1., 1., 0.]]
>>> scaler.transform(X_test)
array([[-2.44948974,  1.22474487, -0.26726124]])
```

## 2）min-max标准化

X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

```

>>> X_train = np.array([[ 1., -1.,  2.],
...                      [ 2.,  0.,  0.],
...                      [ 0.,  1., -1.]])
>>> min_max_scaler = preprocessing.MinMaxScaler()
>>> X_train_minmax = min_max_scaler.fit_transform(X_train)
>>> X_train_minmax
array([[ 0.5       ,  0.        ,  1.        ],
       [ 1.        ,  0.5       ,  0.33333333],
       [ 0.        ,  1.        ,  0.        ]])
```
上边我们创建的min_max_scaler 同样适用于新的测试数据
```
>>> X_test = np.array([[ -3., -1.,  4.]])
>>> X_test_minmax = min_max_scaler.transform(X_test)
>>> X_test_minmax
array([[-1.5       ,  0.        ,  1.66666667]])
```
可以通过scale_和min方法查看标准差和最小值
```
>>> min_max_scaler.scale_ 
array([ 0.5       ,  0.5       ,  0.33333333])
>>> min_max_scaler.min_
array([ 0.        ,  0.5       ,  0.33333333])
```

## 3）最大值标准化

对于每个数值／每个维度的最大值

```
>>> X_train
array([[ 1., -1.,  2.],
       [ 2.,  0.,  0.],
       [ 0.,  1., -1.]])
>>> max_abs_scaler = preprocessing.MaxAbsScaler()
>>> X_train_maxabs = max_abs_scaler.fit_transform(X_train)
>>> X_train_maxabs
array([[ 0.5, -1. ,  1. ],
       [ 1. ,  0. ,  0. ],
       [ 0. ,  1. , -0.5]])
>>> X_test = np.array([[ -3., -1.,  4.]])
>>> X_test_maxabs = max_abs_scaler.transform(X_test)
>>> X_test_maxabs                 
array([[-1.5, -1. ,  2. ]])
>>> max_abs_scaler.scale_         
array([ 2.,  1.,  2.])
```

## 4）规范化
规范化是文本分类和聚类中向量空间模型的基础

```
>>> X = [[ 1., -1.,  2.],
...      [ 2.,  0.,  0.],
...      [ 0.,  1., -1.]]
>>> X_normalized = preprocessing.normalize(X, norm='l2')
>>> X_normalized
array([[ 0.40824829, -0.40824829,  0.81649658],
       [ 1.        ,  0.        ,  0.        ],
       [ 0.        ,  0.70710678, -0.70710678]])
```

解释：norm 该参数是可选的，默认值是l2（向量各元素的平方和然后求平方根），用来规范化每个非零向量，如果axis参数设置为0，则表示的是规范化每个非零的特征维度。

机器学习中的范数规则：[点击阅读](http://blog.csdn.net/zouxy09/article/details/24971995/)<br>
其他对应参数：[点击查看](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html#sklearn.preprocessing.normalize)


preprocessing模块提供了训练种子的功能，我们可通过以下方式得到一个新的种子，并对新数据进行规范化处理。
```
>>> normalizer = preprocessing.Normalizer().fit(X)
>>> normalizer
Normalizer(copy=True, norm='l2')
>>> normalizer.transform(X)
array([[ 0.40824829, -0.40824829,  0.81649658],
       [ 1.        ,  0.        ,  0.        ],
       [ 0.        ,  0.70710678, -0.70710678]])
>>> normalizer.transform([[-1,1,0]])
array([[-0.70710678,  0.70710678,  0.        ]])
```

## 5）二值化
将数据转换到0-1 之间
```
>>> X
[[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]]
>>> binarizer = preprocessing.Binarizer().fit(X)
>>> binarizer
Binarizer(copy=True, threshold=0.0)
>>> binarizer.transform(X)
array([[ 1.,  0.,  1.],
       [ 1.,  0.,  0.],
       [ 0.,  1.,  0.]])
```
可以调整二值化的门阀
```
>>> binarizer = preprocessing.Binarizer(threshold=1.1)
>>> binarizer.transform(X)
array([[ 0.,  0.,  1.],
       [ 1.,  0.,  0.],
       [ 0.,  0.,  0.]])
```

## 6）编码的分类特征

通常情况下，特征不是作为连续值给定的。例如一个人可以有
```
["male", "female"], ["from Europe", "from US", "from Asia"], ["uses Firefox", "uses Chrome", "uses Safari", "uses Internet Explorer"]
```
这些特征可以被有效的编码为整数，例如
```
["male", "from US", "uses Internet Explorer"] => [0, 1, 3]
["female", "from Asia", "uses Chrome"] would be [1, 2, 1].
```
这样的整数不应该直接应用到scikit的算法中，可以通过one-of-k或者独热编码（OneHotEncorder），该种处理方式会把每个分类特征的m中可能值转换成m个二进制值。

```
>>> enc = preprocessing.OneHotEncoder()
>>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
OneHotEncoder(categorical_features='all', dtype=<class 'numpy.float64'>,
       handle_unknown='error', n_values='auto', sparse=True)
>>> enc.transform([[0,1,3]]).toarray()
array([[ 1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.]])
```
默认情况下，从数据集中自动推断出每个特征可以带多少个值。可以明确指定使用的参数n_values。在我们的数据集中有两种性别，三种可能的大陆和四种Web浏览器。然后，我们拟合估计量，并转换一个数据点。在结果中，前两个数字编码性别，下一组三个数字的大陆和最后四个Web浏览器。
```
>>> enc = preprocessing.OneHotEncoder(n_values=[2,3,4])
>>> enc.fit([[1,2,3],[0,2,0]])
OneHotEncoder(categorical_features='all', dtype=<class 'numpy.float64'>,
       handle_unknown='error', n_values=[2, 3, 4], sparse=True)
>>> enc.transform([[1,0,0]]).toarray()
array([[ 0.,  1.,  1.,  0.,  0.,  1.,  0.,  0.,  0.]])
```

## 7）填补缺失值
由于各种原因，真实数据中存在大量的空白值，这样的数据集，显然是不符合scikit的要求的，那么preprocessing模块提供这样一个功能，利用已知的数据来填补这些空白。
```
>>> import numpy as np
>>> from sklearn.preprocessing import Imputer
>>> imp = Imputer(missing_values='NaN',strategy='mean',verbose=0)
>>> imp.fit([[1, 2], [np.nan, 3], [7, 6]])
Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)
>>> X = [[np.nan, 2], [6, np.nan], [7, 6]]
>>> print(imp.transform(X))                           
[[ 4.          2.        ]
 [ 6.          3.66666667]
 [ 7.          6.        ]]
```

Imputer同样支持稀疏矩阵
```
>>> import scipy.sparse as sp
>>> X = sp.csc_matrix([[1,2],[0,3],[7,6]])
>>> imp = Imputer(missing_values=0,strategy='mean',axis=0)
>>> imp.fit(X)
Imputer(axis=0, copy=True, missing_values=0, strategy='mean', verbose=0)
>>> X_test = sp.csc
sp.csc          sp.csc_matrix(  
>>> X_test = sp.csc_matrix([[0,2],[6,0],[7,6]])
>>> print(imp.transform(X_test))
[[ 4.          2.        ]
 [ 6.          3.66666667]
 [ 7.          6.        ]]
```

## 8）生成多项式特征
通常，通过考虑输入数据的非线性特征来增加模型的复杂度是很有用的。一个简单而常用的方法是多项式特征，它可以得到特征的高阶和相互作用项。

其遵循的原则是 
$$ 
(X_1, X_2) -> (1, X_1, X_2, X_1^2, X_1X_2, X_2^2)
$$
```
>>> import numpy as np
>>> from sklearn.preprocessing import PolynomialFeatures
>>> X = np.arange(6).reshape(3, 2)
>>> X                                                 
array([[0, 1],
       [2, 3],
       [4, 5]])
>>> poly = PolynomialFeatures(2)
>>> poly.fit_transform(X)                             
array([[  1.,   0.,   1.,   0.,   0.,   1.],
       [  1.,   2.,   3.,   4.,   6.,   9.],
       [  1.,   4.,   5.,  16.,  20.,  25.]])
```

有些情况下，有相互关系的标签才是必须的，这个时候可以通过设置 interaction_only=True 来进行多项式特征的生成
```
>>> X = np.arange(9).reshape(3, 3)
>>> X                                                 
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])
>>> poly = PolynomialFeatures(degree=3, interaction_only=True)
>>> poly.fit_transform(X)                             
array([[   1.,    0.,    1.,    2.,    0.,    0.,    2.,    0.],
       [   1.,    3.,    4.,    5.,   12.,   15.,   20.,   60.],
       [   1.,    6.,    7.,    8.,   42.,   48.,   56.,  336.]])
```
其遵循的规则是：
$$
(X_1, X_2, X_3) -> (1, X_1, X_2, X_3, X_1X_2, X_1X_3, X_2X_3, X_1X_2X_3)
$$

---

对应的scikit-learn资料为： http://scikit-learn.org/stable/modules/preprocessing.html