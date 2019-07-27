#! /bin/bash
# source environment.txt
Dataset="~/Data/rgbd_dataset_freiburg1_room"
ImgPrefix="1_room"
Num=1

for i in $(seq $Num)
do
echo "start ORB_SLAM2_SP"
roslaunch ORB_SLAM2_SP TUM.launch
done

