---
title: 异常检测之指数平滑（利用elasticsearch来实现）
date: 2017-11-20 17:18:54
tags: [ELK,异常检测,ES]
categories: 技术篇
---

指数平滑法是一种特殊的加权平均法，加权的特点是对离预测值较近的历史数据给予较大的权数，对离预测期较远的历史数据给予较小的权数，权数由近到远按指数规律递减，所以，这种预测方法被称为指数平滑法。它可分为一次指数平滑法、二次指数平滑法及更高次指数平滑法。
<!--More-->
# 关于指数平滑的得相关资料：

- ES API接口：
> https://github.com/IBBD/IBBD.github.io/blob/master/elk/aggregations-pipeline.md
<br><br>
https://www.elastic.co/guide/en/elasticsearch/reference/current/search-aggregations-pipeline-movavg-aggregation.html

- 理论概念
> http://blog.sina.com.cn/s/blog_4b9acb5201016nkd.html

# ES移动平均聚合：Moving Average的四种模型
## simple
就是使用窗口内的值的和除于窗口值，通常窗口值越大，最后的结果越平滑: (a1 + a2 + ... + an) / n
```
curl -XPOST 'localhost:9200/_search?pretty' -H 'Content-Type: application/json' -d'
{
    "size": 0,
    "aggs": {
        "my_date_histo":{
            "date_histogram":{
                "field":"date",
                "interval":"1M"
            },
            "aggs":{
                "the_sum":{
                    "sum":{ "field": "price" }
                },
                "the_movavg":{
                    "moving_avg":{
                        "buckets_path": "the_sum",
                        "window" : 30,
                        "model" : "simple"
                    }
                }
            }
        }
    }
}
'
```

## 线性模型：Linear
对窗口内的值先做线性变换处理，再求平均：(a1 * 1 + a2 * 2 + ... + an * n) / (1 + 2 + ... + n)

```
curl -XPOST 'localhost:9200/_search?pretty' -H 'Content-Type: application/json' -d'
{
    "size": 0,
    "aggs": {
        "my_date_histo":{
            "date_histogram":{
                "field":"date",
                "interval":"1M"
            },
            "aggs":{
                "the_sum":{
                    "sum":{ "field": "price" }
                },
                "the_movavg": {
                    "moving_avg":{
                        "buckets_path": "the_sum",
                        "window" : 30,
                        "model" : "linear"
                    }
                }
            }
        }
    }
}
'
```

## 指数平滑模型
### 指数模型：EWMA (Exponentially Weighted)
即： 一次指数平滑模型

EWMA模型通常也成为单指数模型（single-exponential）, 和线性模型的思路类似，离当前点越远的点，重要性越低，具体化为数值的指数下降，对应的参数是alpha。 alpha值越小，下降越慢。（估计是用1 - alpha去计算的）默认的alpha=0.3

计算模型：s2 = α * x2 + (1 - α) * s1

其中α是平滑系数，si是之前i个数据的平滑值，α取值为[0,1]，越接近1，平滑后的值越接近当前时间的数据值，数据越不平滑，α越接近0，平滑后的值越接近前i个数据的平滑值，数据越平滑，α的值通常可以多尝试几次以达到最佳效果。 一次指数平滑算法进行预测的公式为：xi+h=si，其中i为当前最后的一个数据记录的坐标，亦即预测的时间序列为一条直线，不能反映时间序列的趋势和季节性。

```
curl -XPOST 'localhost:9200/_search?pretty' -H 'Content-Type: application/json' -d'
{
    "size": 0,
    "aggs": {
        "my_date_histo":{
            "date_histogram":{
                "field":"date",
                "interval":"1M"
            },
            "aggs":{
                "the_sum":{
                    "sum":{ "field": "price" }
                },
                "the_movavg": {
                    "moving_avg":{
                        "buckets_path": "the_sum",
                        "window" : 30,
                        "model" : "ewma",
                        "settings" : {
                            "alpha" : 0.5
                        }
                    }
                }
            }
        }
    }
}
'
```

### 二次指数平滑模型: Holt-Linear
计算模型：

s2 = α * x2 + (1 - α) * (s1 + t1)

t2 = ß * (s2 - s1) + (1 - ß) * t1

默认alpha = 0.3 and beta = 0.1

二次指数平滑保留了趋势的信息，使得预测的时间序列可以包含之前数据的趋势。二次指数平滑的预测公式为 xi+h=si+hti 二次指数平滑的预测结果是一条斜的直线。

```
curl -XPOST 'localhost:9200/_search?pretty' -H 'Content-Type: application/json' -d'
{
    "size": 0,
    "aggs": {
        "my_date_histo":{
            "date_histogram":{
                "field":"date",
                "interval":"1M"
            },
            "aggs":{
                "the_sum":{
                    "sum":{ "field": "price" }
                },
                "the_movavg": {
                    "moving_avg":{
                        "buckets_path": "the_sum",
                        "window" : 30,
                        "model" : "holt",
                        "settings" : {
                            "alpha" : 0.5,
                            "beta" : 0.5
                        }
                    }
                }
            }
        }
    }
}
'
```
### 三次指数平滑模型：Holt-Winters无季节模型
三次指数平滑在二次指数平滑的基础上保留了季节性的信息，使得其可以预测带有季节性的时间序列。三次指数平滑添加了一个新的参数p来表示平滑后的趋势。

1: Additive Holt-Winters：Holt-Winters加法模型

下面是累加的三次指数平滑
```
si=α(xi-pi-k)+(1-α)(si-1+ti-1)
ti=ß(si-si-1)+(1-ß)ti-1
pi=γ(xi-si)+(1-γ)pi-k
```
其中k为周期

累加三次指数平滑的预测公式为： xi+h=si+hti+pi-k+(h mod k)

```
curl -XPOST 'localhost:9200/_search?pretty' -H 'Content-Type: application/json' -d'
{
    "size": 0,
    "aggs": {
        "my_date_histo":{
            "date_histogram":{
                "field":"date",
                "interval":"1M"
            },
            "aggs":{
                "the_sum":{
                    "sum":{ "field": "price" }
                },
                "the_movavg": {
                    "moving_avg":{
                        "buckets_path": "the_sum",
                        "window" : 30,
                        "model" : "holt_winters",
                        "settings" : {
                            "type" : "add",
                            "alpha" : 0.5,
                            "beta" : 0.5,
                            "gamma" : 0.5,
                            "period" : 7
                        }
                    }
                }
            }
        }
    }
}
'
```

2: Multiplicative Holt-Winters：Holt-Winters乘法模型

下式为累乘的三次指数平滑：
```
si=αxi/pi-k+(1-α)(si-1+ti-1)
ti=ß(si-si-1)+(1-ß)ti-1
pi=γxi/si+(1-γ)pi-k  其中k为周期
```
累乘三次指数平滑的预测公式为： xi+h=(si+hti)pi-k+(h mod k)

α，ß，γ的值都位于[0,1]之间，可以多试验几次以达到最佳效果。

s,t,p初始值的选取对于算法整体的影响不是特别大，通常的取值为s0=x0,t0=x1-x0,累加时p=0,累乘时p=1.

```
curl -XPOST 'localhost:9200/_search?pretty' -H 'Content-Type: application/json' -d'
{
    "size": 0,
    "aggs": {
        "my_date_histo":{
            "date_histogram":{
                "field":"date",
                "interval":"1M"
            },
            "aggs":{
                "the_sum":{
                    "sum":{ "field": "price" }
                },
                "the_movavg": {
                    "moving_avg":{
                        "buckets_path": "the_sum",
                        "window" : 30,
                        "model" : "holt_winters",
                        "settings" : {
                            "type" : "mult",
                            "alpha" : 0.5,
                            "beta" : 0.5,
                            "gamma" : 0.5,
                            "period" : 7,
                            "pad" : true
                        }
                    }
                }
            }
        }
    }
}
'
```
## 预测模型：Prediction
使用当前值减去前一个值，其实就是环比增长

```
curl -XPOST 'localhost:9200/_search?pretty' -H 'Content-Type: application/json' -d'
{
    "size": 0,
    "aggs": {
        "my_date_histo":{
            "date_histogram":{
                "field":"date",
                "interval":"1M"
            },
            "aggs":{
                "the_sum":{
                    "sum":{ "field": "price" }
                },
                "the_movavg": {
                    "moving_avg":{
                        "buckets_path": "the_sum",
                        "window" : 30,
                        "model" : "simple",
                        "predict" : 10
                    }
                }
            }
        }
    }
}
'
```

## 最小化：Minimization
某些模型（EWMA，Holt-Linear，Holt-Winters）需要配置一个或多个参数。参数选择可能会非常棘手，有时不直观。此外，这些参数的小偏差有时会对输出移动平均线产生剧烈的影响。

出于这个原因，三个“可调”模型可以在算法上最小化。最小化是一个参数调整的过程，直到模型生成的预测与输出数据紧密匹配为止。最小化并不是完全防护的，并且可能容易过度配合，但是它往往比手动调整有更好的结果。

ewma和holt_linear默认情况下禁用最小化，而holt_winters默认启用最小化。 Holt-Winters最小化是最有用的，因为它有助于提高预测的准确性。 EWMA和Holt-Linear不是很好的预测指标，主要用于平滑数据，所以最小化对于这些模型来说不太有用。

通过最小化参数启用/禁用最小化："minimize" : true

# 原始数据
数据为SSH login数据其中 IP／user已处理
```
{
    "_index": "logstash-sshlogin-others-success-2017-10",
    "_type": "sshlogin",
    "_id": "AV-weLF8c2nHCDojUbat",
    "_version": 2,
    "_score": 1,
    "_source": {
        "srcip": "222.221.238.162",
        "dstport": "",
        "pid": "20604",
        "program": "sshd",
        "message": "dwasw-ibb01:Oct 19 23:38:02 176.231.228.130 sshd[20604]: Accepted publickey for nmuser from 222.221.238.162 port 49484 ssh2",
        "type": "zhongcai-sshlogin",
        "ssh_type": "ssh_successful_login",
        "forwarded": "false",
        "manufacturer": "others",
        "IndexTime": "2017-10",
        "path": "/home/logstash/log/logstash_data/audit10/sshlogin/11.txt",
        "number": 1,
        "hostname": "176.231.228.130",
        "protocol": "ssh2",
        "@timestamp": "2017-10-19T15:38:02.000Z",
        "ssh_method": "publickey",
        "_hostname": "dwasw-ibb01",
        "@version": "1",
        "host": "localhost",
        "srcport": "49484",
        "dstip": "",
        "category": "sshlogin",
        "user": "nmuser"
    }
}
```

# 利用ES API接口去调用查询数据

"interval": "hour": hour为单位，这里可以是分钟，小时，天，周，月

"format": "yyyy-MM-dd HH": 聚合结果得日期格式

```
"the_sum": {
    "sum": {
        "field": "number"
    }
}
```
number为要聚合得字段

```
curl -POST  'localhost:9200/logstash-sshlogin-others-success-2017-10/sshlogin/_search?pretty' -H 'Content-Type: application/json' -d'
{
  "size": 0,
  "query": {
    "term": {
      "ssh_type": "ssh_successful_login"
    }
  },
  "aggs": {
    "hour_sum": {
      "date_histogram": {
        "field": "@timestamp",
        "interval": "hour",
        "format": "yyyy-MM-dd HH"
      },
      "aggs": {
        "the_sum": {
          "sum": {
            "field": "number"
          }
        },
        "the_movavg": {
          "moving_avg": {
            "buckets_path": "the_sum",
            "window": 30,
            "model": "holt",
            "settings": {
              "alpha": 0.5,
              "beta": 0.7
            }
          }
        }
      }
    }
  }
}'
```
得到的结果形式为：
```
{
  "took" : 35,
  "timed_out" : false,
  "_shards" : {
    "total" : 1,
    "successful" : 1,
    "failed" : 0
  },
  "hits" : {
    "total" : 206821,
    "max_score" : 0.0,
    "hits" : [ ]
  },
  "aggregations" : {
    "hour_sum" : {
      "buckets" : [
        {
          "key_as_string" : "2017-09-30 16",
          "key" : 1506787200000,
          "doc_count" : 227,
          "the_sum" : {
            "value" : 227.0
          }
        },
        {
          "key_as_string" : "2017-09-30 17",
          "key" : 1506790800000,
          "doc_count" : 210,
          "the_sum" : {
            "value" : 210.0
          },
          "the_movavg" : {
            "value" : 113.5
          }
        },
        {
          "key_as_string" : "2017-09-30 18",
          "key" : 1506794400000,
          "doc_count" : 365,
          "the_sum" : {
            "value" : 365.0
          },
          "the_movavg" : {
            "value" : 210.0
          }
        },
    ...
    }
}
```

# 对应得python代码（查询数据到画图）
```
# coding: utf-8
from elasticsearch import Elasticsearch
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontManager, FontProperties

class Smooth:
    def __init__(self,index):
        self.es = Elasticsearch(['localhost:9200'])
        self.index = index
        
    # 处理mac中文编码错误
    def getChineseFont(self):
        return FontProperties(fname='/System/Library/Fonts/PingFang.ttc')
    
    # 对index进行聚合
    def agg(self):
        # "format": "yyyy-MM-dd HH:mm:SS"
        dsl = '''
                {
                  "size": 0,
                  "query": {
                    "term": {
                      "ssh_type": "ssh_successful_login"
                    }
                  },
                  "aggs": {
                    "hour_sum": {
                      "date_histogram": {
                        "field": "@timestamp",
                        "interval": "day",
                        "format": "dd"
                      },
                      "aggs": {
                        "the_sum": {
                          "sum": {
                            "field": "number"
                          }
                        },
                        "the_movavg": {
                          "moving_avg": {
                            "buckets_path": "the_sum",
                            "window": 30,
                            "model": "holt_winters",
                            "settings": {
                              "alpha": 0.5,
                              "beta": 0.7
                            }
                          }
                        }
                      }
                    }
                  }
                }
                '''
        res = self.es.search(index=self.index, body=dsl)
        return res['aggregations']['hour_sum']['buckets']
    
    # 画图
    def draw(self):
        x,y_true,y_pred = [],[],[]
        for one in self.agg():
            x.append(one['key_as_string'])
            y_true.append(one['the_sum']['value'])
            if 'the_movavg' in one.keys():       # 前几条数据没有 the_movavg 字段，故将真实值赋值给pred值
                y_pred.append(one['the_movavg']['value'])
            else:
                y_pred.append(one['the_sum']['value'])
        
        x_line = range(len(x))
        
        plt.figure(figsize=(10,5))
        plt.plot(x_line,y_true,color="r")
        plt.plot(x_line,y_pred,color="g")
        
        plt.xlabel(u"每单位时间",fontproperties=self.getChineseFont()) #X轴标签 
        plt.xticks(range(len(x)), x)
        plt.ylabel(u"聚合结果",fontproperties=self.getChineseFont()) #Y轴标签  
        plt.title(u"10月份 SSH 主机登录成功聚合图",fontproperties=self.getChineseFont()) # 标题
        plt.legend([u"True value",u"Predict value"])
        plt.show()

smooth = Smooth("logstash-sshlogin-others-success-2017-10")
print smooth.draw()
```
结果图示为：
![这里写图片描述](http://img.blog.csdn.net/20171120171404972?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvR2FtZXJfZ3l0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
