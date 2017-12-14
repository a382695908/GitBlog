---
title: Hexo-Yilia加入相册功能
date: 2017-12-14 17:55:29
tags: [hexo]
categories: 随手记
---
参考：[点击查看](http://maker997.com/2017/07/01/hexo-Yilia-%E4%B8%BB%E9%A2%98%E5%A6%82%E4%BD%95%E6%B7%BB%E5%8A%A0%E7%9B%B8%E5%86%8C%E5%8A%9F%E8%83%BD)

但是其中有一些小问题，自己便重新整理了一下（本文适用于使用github存放照片）

<!--More-->
# 主页新建相册链接

主题_config.json文件的menu 中加入 相册和对应的链接
```
themes/yilia/_config.json

menu:
  主页: /
  ... ...
  相册: /photos
```

# 新建目录并拷贝相应文件
使用的是litten 大神的博客 photos文件夹，对应的路径为：
https://github.com/litten/BlogBackup/tree/master/source/photos

自己的项目根目录下的source文件夹下新建photos文件夹，将下载的几个文件放在该文件夹中，或者不用新建，直接将下载的photos文件夹放在source目录下。

# 文件修改

 1. 修改 ins.js 文件的 render()函数
 这个函数是用来渲染数据的
修改图片的路径地址.minSrc 小图的路径. src 大图的路径.修改为自己的图片路径(github的路径)
例如我的为：
```
var minSrc = 'https://raw.githubusercontent.com/Thinkgamer/GitBlog/master/min_photos/' + data.link[i] + '.min.jpg';
var src = 'https://raw.githubusercontent.com/Thinkgamer/GitBlog/master/photos/' + data.link[i];

```
# 生成json
1：下载相应python工具文件

- tools.py
- ImageProcess.py

下载地址：https://github.com/Thinkgamer/GitBlog

2：新建photos和min_photos文件夹
在项目根目录下创建，用来存放照片和压缩后的照片
```
mkdir photos
mkdir min_photos
```
3：py文件和文件夹都放在项目根目录下

4：生成json
执行
```
python tools.py
```
如果提示：
```
Traceback (most recent call last):
  File "tools.py", line 13, in <module>
    from PIL import Image
ImportError: No module named PIL
```
说明你没有安装pillow，执行以下命令安装即可
```
pip install pillow
```

如果报错：
```
ValueError: time data 'DSC' does not match format '%Y-%m-%d'
```
说明你照片的命名方式不合格，这里必须命名为以下这样的格式（当然时间是随意的）
```
2016-10-12_xxx.jpg/png
```
ok，至此会在min_photos文件夹下生成同名的文件，但是大小会小很多

# 本地预览和部署
## 本地预览
项目根目录下执行
```
hexo s
```
浏览器4000端口访问，按照上边的方式进行配置，正常情况下你是看不到图片的，通过调试可以发现图片的url中后缀变成了 xxx.jpg.jpg，所以我们要去掉一个jpg

改正方法
ins.js/render 函数
```
var minSrc = 'https://raw.githubusercontent.com/Thinkgamer/GitBlog/master/min_photos/' + data.link[i] + '.min.jpg';

换成

var minSrc = 'https://raw.githubusercontent.com/Thinkgamer/GitBlog/master/min_photos/' + data.link[i];

注释掉该行：
src += '.jpg'; 
```

到这里没完，路径都对了，但是在浏览器中还是不能看到图片，调试发现，下载大神的photos文件夹的ins.js中有一行代码，饮用了一张图片，默认情况下，在你的项目中，这张图片是不存在的，改正办法就是在对应目录下放一张图片，并修改相应的名字

```
src="/assets/img/empty.png
```

ok，至此刷新浏览器是可以看到图片的，如果还看不到，应该就是浏览器缓存问题了，如果还有问题，可以加我微信进行沟通：gyt13342445911