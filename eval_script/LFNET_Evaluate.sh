#! /bin/bash
#  使用说明
#  1. 更改Dataset的值为需要测评的数据集的根路径
#  2. 更改ImgPrefix的值为保存的png图片的前缀，Num是想要测评该数据的次数
#  3. 一定要记得！！！更换数据集后，把png txt文件夹下的文件全部拷出来单独存，尤其是txt文件，否则分不清是哪个数据集测评的结果了。由于记录下的txt文件里的数据之间都有空格作为分隔符，所以可以直接粘贴到libreOffice软件 的excel表格里面，选择分隔选项 空格 就可以完美整理到表格中啦 不过png 和txt文件夹不要删除哦，否则无法自动创建的

Dataset="~/Data/rgbd_dataset_freiburg1_room"
ImgPrefix="1_room"
Num=1

for i in $(seq $Num)
do
echo "start test"
roslaunch ORB_SLAM2_SP TUM.launch
python evaluate_rpe.py $Dataset/groundtruth.txt /home/yuchao/catkin_ws/src/ORB_SLAM2_SP/CameraTrajectory.txt --fixed_delta --plot png/${ImgPrefix}_RPE_SP_${i}.png --verbose
python evaluate_ate.py $Dataset/groundtruth.txt /home/yuchao/catkin_ws/src/ORB_SLAM2_SP/CameraTrajectory.txt --plot png/${ImgPrefix}_ATE_SP_${i}.png --verbose
done

