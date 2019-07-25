/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "PointCloud.h"
#include <KeyFrame.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <thread>
#include "Converter.h"

#include "Eigen/Dense" 
#include "Eigen/Geometry"

#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/projection_matrix.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>

pcl::PointCloud<pcl::PointXYZRGBA> pcl_filter; 
ros::Publisher pcl_pub;
sensor_msgs::PointCloud2 pcl_point;

pcl::PointCloud<pcl::PointXYZRGBA> pcl_cloud;

// every frame cloud
sensor_msgs::PointCloud2 camera_point;
ros::Publisher camera_pub;
pcl::PointCloud<pcl::PointXYZRGBA> camera_cloud;

PointCloudMapping::PointCloudMapping(double resolution_)
{
    this->resolution = resolution_;
    voxel.setLeafSize( resolution, resolution, resolution);
    this->sor.setMeanK(50);                         
    this->sor.setStddevMulThresh(1.0);
    globalMap = boost::make_shared< PointCloud >();
    
    viewerThread = boost::make_shared<thread>( bind(&PointCloudMapping::viewer, this ) );
}

void PointCloudMapping::shutdown()
{
    {
        unique_lock<mutex> lck(shutDownMutex);  
        shutDownFlag = true;
        keyFrameUpdated.notify_one();
    }
    viewerThread->join();
}

void PointCloudMapping::insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)
{
    //cout<<"receive a keyframe, id = "<<kf->mnId<<endl;
    unique_lock<mutex> lck(keyframeMutex);
    keyframes.push_back( kf );
    colorImgs.push_back( color.clone() );
    depthImgs.push_back( depth.clone() );

    keyFrameUpdated.notify_one();
    
}

pcl::PointCloud< PointCloudMapping::PointT >::Ptr PointCloudMapping::generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)
{

    PointCloud::Ptr tmp( new PointCloud() );
    // point cloud is null ptr
    for ( int m=0; m<depth.rows; m+=3)
    {
        for ( int n=0; n<depth.cols; n+=3)
        {
            float d = depth.ptr<float>(m)[n];
            if (d < 0.01 || d > 10)
               continue;  	
	   
            PointT p;
            p.z = d;
            p.x = ( n - kf->cx) * p.z / kf->fx;
            p.y = ( m - kf->cy) * p.z / kf->fy;

			p.b = color.ptr<uchar>(m)[n*3];
			p.g = color.ptr<uchar>(m)[n*3+1];
			p.r = color.ptr<uchar>(m)[n*3+2]; 
			tmp->points.push_back(p);
        }
    }
    // add camera points
    ros::NodeHandle n;
    camera_pub = n.advertise<sensor_msgs::PointCloud2>("/ORB_SLAM2_SP/camera_PointCloud",100000);
    camera_cloud = *tmp;
    pcl::toROSMsg(camera_cloud, camera_point);
    camera_point.header.frame_id = "/map";
    camera_pub.publish(camera_point);

    Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat( kf->GetPose() );
    PointCloud::Ptr cloud(new PointCloud);
    pcl::transformPointCloud( *tmp, *cloud, T.inverse().matrix());
    cloud->is_dense = false;

    cout<<"generate point cloud for kf "<<kf->mnId<<", size="<<cloud->points.size()<<endl;
    return cloud;
}


void PointCloudMapping::viewer()
{
    ros::NodeHandle n;
    pcl_pub = n.advertise<sensor_msgs::PointCloud2>("/ORB_SLAM2_SP/PointCloud",100000);
    while(1)
    {
        
        {
            unique_lock<mutex> lck_shutdown( shutDownMutex );
            if (shutDownFlag)
            {
                break;
            }
        }
        
        {
            unique_lock<mutex> lck_keyframeUpdated( keyFrameUpdateMutex );
            keyFrameUpdated.wait( lck_keyframeUpdated );
        }
        
        // keyframe is updated
        size_t N=0;
        {
            unique_lock<mutex> lck( keyframeMutex );
            N = keyframes.size();
        }
        
        if(N==0)
        {
            cout<<"keyframes miss"<<endl;
            usleep(1000);
            continue;
        }
        
        for ( size_t i=lastKeyframeSize; i<N ; i++ )
        {
            PointCloud::Ptr p = generatePointCloud( keyframes[i], colorImgs[i], depthImgs[i]);
            *globalMap += *p;
        }
        //PointCloud::Ptr tmp1(new PointCloud());
        //voxel.setInputCloud( KfMap );
        //voxel.filter( *tmp1 );
        //KfMap->swap( *tmp1 );
        pcl_cloud = *globalMap;		
        Cloud_transform(pcl_cloud,pcl_filter);
        pcl::toROSMsg(pcl_filter, pcl_point);
        pcl_point.header.frame_id = "/map";
        pcl_pub.publish(pcl_point);//publish point cloud
        lastKeyframeSize = N;	
        
    }
}

void PointCloudMapping::Cloud_transform(pcl::PointCloud<pcl::PointXYZRGBA>& source, pcl::PointCloud<pcl::PointXYZRGBA>& out)
{
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered;

	Eigen::Matrix4f m;

	m<< 0,0,1,0,
	    -1,0,0,0,
		0,-1,0,0;
	Eigen::Affine3f transform(m);
	pcl::transformPointCloud (source, out, transform);
}
