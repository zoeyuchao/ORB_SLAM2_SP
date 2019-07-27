# SP-SLAM使用说明

## 1. 安装

- 先把ROS的**全版本**安装一下，确认有opencv和octomap。

- 执行下面的命令

  `mkdir && cd ~/catkin_ws/src/`

  `git clone https://github.com/zoeyuchao/ORB_SLAM2_SP.git -b add-orb-pytorch-caffe`

- 依赖库的安装步骤跟ORB_SLAM2一样，请移步官网，注意opencv就用ros自带的即可，自己安装的会有诸多奇怪的bug需要解决。

- 自己需要安装一个pcl库，用命令`sudo apt-get install libpcl-dev`默认安装1.7版本。

- cuda自行选择，然后下载caffe-ssd放在`~/`下，注意用make，不需要cmake。

  `cd ~/caffe-ssd`

  `make -j`

- 依赖库都安装好以后直接执行`./build.sh`

## 2. 使用

- 打开`Examples/ROS/ORB_SLAM2_SP`里面的`TUM1.yaml`,按照自己的需求修改,1表示启用，0表示不启用。
  
  `UseLocalMap: 1`
  
  `UseLoop: 1`
  
  `UseViewer: 1`
  
  `UseBoW: 1`
  
  `UseORB: 0`
  
  `UseExistFile: 0`
  
  `OnlyTracking: 0`

- 如果选择采用UseExistFile，则需要提前下载离线数据集，[链接](链接：https://pan.baidu.com/s/1TIuS7voxUUHdXP6Jv8X8jA 
)（提取码：2vfm ）放在`~/Data/`下，这里注意默认是1000个特征点，描述子长度是256。

- 测试用的是TUM.launch文件，执行命令是

  ```Shell
  source environment.txt
  roslaunch ORB_SLAM2_SP TUM.launch
  ```
