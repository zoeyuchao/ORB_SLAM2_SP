# SP-SLAM使用说明

## 1. 安装

跟LF-SLAM一样，请移步。

## 2. 使用

- 下载数据集，[链接](链接：https://pan.baidu.com/s/1TIuS7voxUUHdXP6Jv8X8jA 
  )（提取码：2vfm ）

在***~/catkin_ws/src/ORB_SLAM2_SP***路径下运行下列语句:

[^]: 注意:运行前要先运行SP,将特征提前提取完毕(这一步已经做完,暂时不需要再运行),存储在/home/zoe/data/rgbd_dataset_freiburg1_room/SP文件里,如果改动了SP-SLAM的代码,先在build下进行编译.

```bash
./Examples/RGB-D/rgbd_tum Vocabulary/LFNET500voc.txt Examples/RGB-D/TUM1.yaml /home/yuchao/Data/rgbd_dataset_freiburg1_room /home/yuchao/Data/rgbd_dataset_freiburg1_room/associate.txt
```

特别注意：

- rgb-d tum文件里有4个true或者false的选项需要在编译之前修改，分别控制的是viewer，localmap，loop和trackonly这4种模式（代码中改动的地方搜索zoe 20190513能找到）

## 3. 说明

1. 本程序基于ORB-SLAM2框架,将ORB特征全面替换为SuperPoint特征,文中所有改动均已加了注释//zoe time,可以全文搜索zoe,寻找改动的地方。
2. 由于SP和ORB特征不尽相同,所以level这个信息目前是全面屏蔽的.之后再进行改进。
3. SP的角度是在SLAM程序中算出来的，跟LF是有区别的，也是这版代码的一个改进。
4. 程序还有很多超参数,不知道应该怎么进行修改,目前就是暂时可以按照正常的程序运行,但是精度不如ORB本身,差了一个量级。
5. 字典生成代码在Vocabulary文件夹下的create_voc.bin,原始代码在src下的create_voc.cc文件,如果需要修改,请先修改源码,然后在build文件里进行编译,同时注意,有可能还要进一步修改DBOW2三方库下的文件,里面有FLFNET.cc和FLFNET.h文件,这两个文件是我新加的,适用于新的LFNET特征。
6. 根据不同的特征距离计算方式,可以有平方差及sqrt(平方差)两种方式,也可以有一张照片提取500个特征点或者1000个特征点,根据此,我生成了几个词袋模型。
      1. LFNET500voc.txt表示500个特征点,平方差.
      2. LFNET500sqrtvoc.txt表示500个特征点,sqrt(平方差).
      3. LFNET1000voc.txt表示1000个特征点,平方差.
7. **精度评价**
   在eval_script文件夹下,有2个sh脚本,直接运行即可,每次运行完会存在png和txt下，我都是拷贝出来的。他们的作用分别是;
       1. LFNET_Evaluate.sh:进行LF-SLAM的性能分析
           2. ORB_Evaluate.sh:进行ORB-SLAM2的性能分析
           3. SP_Evaluate.sh:进行SP-SLAM的性能分析
[^]: 运行脚本前请看脚本里的备注,注意路径/数据集名称是否需要修改,如果在此电脑运行,则不需要更改.
​	现在存了LFNET500和LFNET1000以及ORB SP四个文件夹,里面有对应的性能分析文件,供参考.

## 4.实验结果

- 20190513 把线程分离开，但是词袋模型没有删掉，实验结果[ORB链接](https://pan.baidu.com/s/1aWpuZQJuJV48nLr0-Jrx3g)(提取码：s556)    [SP链接](https://pan.baidu.com/s/1VkUAtfK-Mgn7d-30dLA6UQ)(提取码：mbwy)