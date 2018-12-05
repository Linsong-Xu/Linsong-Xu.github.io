---
title: 配置服务器cuda+cudnn+tensorflow-gpu
date: 2018-12-01 00:00:00
categories: Engineer
tags: [Configuration]
---

配置学院服务器GPU，解决权限不足，无法将`cudnn`和`cuda`合并的问题。

<!--more-->



# 配置服务器cuda+cudnn+tensorflow-gpu

------
## 问题
权限不足，无法将`cudnn`和`cuda`合并。

> 确定需要安装的框架（以`tensorflow`为例）及其版本（以tensorflow_gpu-1.10.1-cp36-cp36mmanylinux1_
x86_64 为例），框架版本与`cuda` 和`cudnn`对应关系见[此](https://www.tensorflow.org/install/source#tested_source_configurations)

> 这篇教程使用下面链接里的[cuda+cudnn+tensorflow-gpu版本](https://pan.baidu.com/s/1yDKYuT6OZ0k59qn4IhqrbA),提取码为：km3n

------

## 解决方法

### 1. 以我的服务器为例：一开始`$HOME`目录下只有以下几个文件：

> Anaconda3-5.0.0-Linux-x86_64.sh

> cuda_9.0.176_384.81_linux-run

> cudnn-9.0-linux-x64-v7.tgz

> pip install tensorflow_gpu-1.10.1-cp36-cp36m-manylinux1_x86_64.whl

### 2. 创建cuda目录并安装cuda
创建自己的`cuda`目录，`cuda-9.0`(不要命名为cuda，因为后面cudnn解压后会有一个cuda重名文件)
```
mkdir cuda-9.0
```
更改文件权限
```
chmod +x cuda_9.0.176_384.81_linux-run
```
安装到自己的`cuda-9.0`目录下
```
./cuda _9.0.176_384.81_linux-run
```
这一步安装时会遇到以下几个问题：
> Do you accept the previously read EULA? 
**`accept`**

> Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 384.81?
**`n`**

> Install the CUDA 9.0 Toolkit
**`y`**

> Enter Toolkit Location
**`/home/[your_username]/cuda-9.0`**

> Do you want to install a symbolic link at /usr/local/cuda
**`n`**

> Install the CUDA 9.0 Samples?
**`n`**

### 3. 解压`cudnn`文件
```
tar -xvf cudnn-9.0-linux-x64-v7.tgz
```
### 4. 拷贝`cudnn`的文件到`cuda-9.0`的目录下并更改权限
```
cp cuda/include/cudnn.h cuda-9.0/include/
```
```
cp cuda/lib64/libcudnn* cuda-9.0/lib64
```
```
chmod a+r cuda-9.0/include/cudnn.h cuda-9.0/lib64/libcudnn*
```


### 5. 给自己的当前用户配置环境变量
编辑`.bashrc`
```
vim .bashrc
```
添加环境变量
```
export PATH=$HOME/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/cuda-9.0/lib64/
```
使更改的环境变量生效
```
source .bashrc
```

### 6. 安装tensorflow-gpu

```
pip install tensorflow_gpu-1.10.1-cp36-cp36m-manylinux1_x86_64.whl
```
### 7. 测试

``` python
import tensorflow as tf
 
with tf.device('/cpu:0'):
    a = tf.constant([1.0,2.0,3.0],shape=[3],name='a')
    b = tf.constant([1.0,2.0,3.0],shape=[3],name='b')
with tf.device('/gpu:1'):
    c = a+b
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
sess.run(tf.global_variables_initializer())
print(sess.run(c))
```

