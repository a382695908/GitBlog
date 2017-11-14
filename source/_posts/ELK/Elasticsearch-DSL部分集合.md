---
title: Elasticsearch-DSL部分集合
date: 2017-11-14 17:26:48
tags: [ELK,ES]
categories: 技术篇
---

ELK是日志收集分析神器，在这篇文章中将会介绍一些ES的常用命令。

点击阅读：[ELK Stack 从入门到放弃](http://blog.csdn.net/column/details/13079.html)
<!--More-->
# DSL中遇到的错误及解决办法
## 分片限制错误
```
Trying to query 2632 shards, which is over the limit of 1000. This limit exists because querying many shards at the same time can make the job of the coordinating node very CPU and/or memory intensive. It is usually a better idea to have a smaller number of larger shards. Update [action.search.shard_count.limit] to a greater value if you really want to query that many shards at the same time.
```

解决办法：
```
修改该限制数目

curl -k -u admin:admin -XPUT 'http://localhost:9200/_cluster/settings' -H 'Content-Type: application/json' -d' 
{
    "persistent" : {
        "action.search.shard_count.limit" : "5000"
    }
}
'

-k -u admin:admin 表述如果有权限保护的话可以加上
```

## Fileddate 错误
```
Fielddata is disabled on text fields by default. Set fielddata=true on [make] in order to load fielddata in memory by uninverting the inverted index. Note that this can however use significant memory.
```

解决办法：
```
cars: 索引名
transactions：索引对应的类型
color：字段

curl -XPUT -k -u admin:admin 'localhost:9200/cars/_mapping/transactions?pretty' -H 'Content-Type: application/json' -d'
{
  "properties": {
    "color": { 
      "type":     "text",
      "fielddata": true
    }
  }
}
'
```

# 指定关键词查询，排序和函数统计
## 指定关键词

from 为首个偏移量，size为返回数据的条数
```
http://10.10.11.139:9200/logstash-nginx-access-*/nginx-access/_search?pretty

{
    "from":0,size":1000,
    "query" : {
        "term" : { 
        	"major" : "55"
        }
    }
}
```

## 添加排序

(需要进行mapping设置，asc 为升序  desc为降序)
```
{
    "from":0,"size":1000,
    "sort":[
        {"offset":"desc"}
    ],
    "query" : {
        "term" : { 
            "major" : "55"
        }
    }
}
```

## mode 方法
mode方法包括 min／max／avg／sum／median

假如现在要对price字段进行排序，但是price字段有多个值，这个时候就可以使用mode 方法了。

```
{
   "query" : {
      "term" : { "product" : "chocolate" }
   },
   "sort" : [
      {"price" : {"order" : "asc", "mode" : "avg"}}
   ]
}
```

# IP范围和网段查询
## IP range 搜索

错误：
```
Fielddata is disabled on text fields by default. Set fielddata=true on [clientip] in order to load fielddata in memory by uninverting the inverted index. Note that this can however use significant memory.
```

解决办法：
```
curl -k -u admin:admin -XPUT '10.10.11.139:9200/logstash-nginx-access-*/_mapping/nginx-access?pretty' -H 'Content-Type: application/json' -d'
{
  "properties": {
    "clientip": { 
      "type":     "text",
      "fielddata": true,
      "norms": false
    }
  }
}
'
```

查看某个索引的mapping
```
curl -k -u admin:admin -XGET http://10.10.11.139:9200/logstash-nginx-access-*/_mapping?pretty
```

(当IP为不可解析使就会出现错误)
```
http://10.10.11.139:9200/logstash-sshlogin-others-success-*/zhongcai/_search?pretty
{
    "size":100,
    "aggs" : {
        "ip_ranges" : {
            "ip_range" : {
                "field" : "clientip",
                "ranges" : [
                    { "to" : "40.77.167.73" },
                    { "from" : "40.77.167.75" }
                ]
            }
        }
    }
}
```

## 网段查询
```
http://10.10.11.139:9200/logstash-sshlogin-others-success-*/zhongcai/_search?pretty
{
    "aggs" : {
        "ip_ranges" : {
            "ip_range" : {
                "field" : "ip",
                "ranges" : [
                    { "mask" : "172.21.202.0/24" },
                    { "mask" : "172.21.202.0/24" }
                ]
            }
        }
    }
}
```

# 关于索引的操作

## 删除某个索引
-k -u admin:admin 为用户名：密码
```
curl -XDELETE  -k -u admin:admin 'http://localhost:9200/my_index'
```


## 查看某个索引的Mapping
```
curl -XGET "http://127.0.0.1:9200/my_index/_mapping?pretty"
```

## 索引数据迁移

Es索引reindex(从ip_remote上迁移到本地)
```
curl -XPOST 'localhost:9200/_reindex?pretty' -H 'Content-Type: application/json' -d'
{
  "source": {
    "remote": {
      "host": "http://ip_remote:9200",
      "username": "username",
      "password": "passwd"
    },
    "index": "old_index"
  },
  "dest": {
    "index": "new_index"
  }
}
'
```

## 为某个索引添加字段
添加number字段：
### 唯一ID
```
curl -POST 'http://127.0.0.1:9200/my_idnex/my_index_type/id/_update?pretty' -H 'Content-Type: application/json' -d'
{
   "doc" : {
      "number" : 1
   }
}
'
```
### 批量操作
```
curl -XPOST 'localhost:9200/logstash-sshlogin-others-success-2017-*/_update_by_query?pretty' -H 'Content-Type: application/json' -d'
{
  "script": {
    "inline": "ctx._source.number=1",
    "lang": "painless"
  },
  "query": {
    "match_all": {
    }
  }
}
'

```

# 根据指定条件进行聚合
每小时成功登录的次数进行聚合
```
curl -POST 'http://127.0.0.1:9200/logstash-sshlogin-others-success-2017-*/zhongcai-sshlogin/_search?pretty' -H 'Content-Type: application/json' -d'
{
  "query": {
    "term": {
      "ssh_type": "ssh_successful_login"
    }
  },
  "aggs": {
    "sums": {
      "date_histogram": {
        "field": "@timestamp",
        "interval": "hour",
        "format": "yyyy-MM-dd HH"
      }
    }
  }
}
'
```