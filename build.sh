source environment.txt

echo "Configuring and building Thirdparty/DBoW2 ..."

cd Thirdparty/DBoW2
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

cd ../../g2o

echo "Configuring and building Thirdparty/g2o ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

cd ../../superpoint

echo "Configuring and building Thirdparty/superpoint ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make

cd ../../../

echo "Configuring and building ORB_SLAM2_SP ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

cd ../Examples/ROS/ORB_SLAM2_SP/

echo "Configuring and building ORB_SLAM2_SP_ROS ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
