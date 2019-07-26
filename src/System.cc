/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/



#include "System.h"
#include "Converter.h"
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>

namespace ORB_SLAM2
{
//zoe 20190513
System::System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor):mSensor(sensor), mpViewer(static_cast<Viewer*>(NULL)),mpLocalMapper(static_cast<LocalMapping*>(NULL)),mpLoopCloser(static_cast<LoopClosing*>(NULL)), mbReset(false),mbActivateLocalizationMode(false),
        mbDeactivateLocalizationMode(false), mbUseLocalMap(false), mbUseLoop(false), mbUseBoW(false), mbUseORB(false), mbUseExistFile(false), mbOnlyTracking(false), mbUseViewer(false)
{
    // Output welcome message
    cout << endl <<
    "ORB-SLAM2 Copyright (C) 2014-2016 Raul Mur-Artal, University of Zaragoza." << endl <<
    "This program comes with ABSOLUTELY NO WARRANTY;" << endl  <<
    "This is free software, and you are welcome to redistribute it" << endl <<
    "under certain conditions. See LICENSE.txt." << endl << endl;

    cout << "Input sensor was set to: ";

    if(mSensor==MONOCULAR)
        cout << "Monocular" << endl;
    else if(mSensor==STEREO)
        cout << "Stereo" << endl;
    else if(mSensor==RGBD)
        cout << "RGB-D" << endl;

    //Check settings file
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
       cerr << "Failed to open settings file at: " << strSettingsFile << endl;
       exit(-1);
    }

    int bUseLocalMap = fsSettings["UseLocalMap"];
    int bUseLoop = fsSettings["UseLoop"];
    int bUseViewer = fsSettings["UseViewer"];
    int bUseBoW = fsSettings["UseBoW"];
    int bUseORB = fsSettings["UseORB"];
    int bUseExistFile = fsSettings["UseExistFile"];
    int bOnlyTracking = fsSettings["OnlyTracking"];

    if (bUseLocalMap == 1)
        mbUseLocalMap = true;
    if (bUseLoop == 1)
        mbUseLoop = true;
    if (bUseViewer == 1)
        mbUseViewer = true;
    if (bUseBoW == 1)
        mbUseBoW = true;
    if (bUseORB == 1)
        mbUseORB = true;
    if (bUseExistFile == 1)
        mbUseExistFile = true;
    if (bOnlyTracking == 1)
        mbOnlyTracking = true;

    // turn off the loop or not //zoe 20190511 
    cout << "Loop Mapping is set: " << mbUseLocalMap << endl;
    cout << "Loop Closing is set: " << mbUseLoop << endl;
    cout << "Viewer is set: " << mbUseViewer << endl;
    cout << "Use BoW is set: " << mbUseBoW << endl;
    cout << "Use ORB is set: " << mbUseORB << endl;
    cout << "Use ExistFile is set: " << mbUseExistFile << endl;
    cout << "Only Track is set: " << mbOnlyTracking << endl;

    //Create the Map
    mpMap = new Map();

    //Create Drawers. These are used by the Viewer
    mpFrameDrawer = new FrameDrawer(mpMap);
    mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);
    float resolution = fsSettings["PointCloudMapping.Resolution"];
    mpPointCloudMapping = boost::make_shared<PointCloudMapping>(resolution);//zoe 20190711

    if (mbUseORB && mbUseExistFile)
    {
        cout << "ORB and ExistFile can not be true together, so use ORB is chosen first！" << endl;
    }

    if (mbUseLoop && !mbUseBoW)
    {
        cout << "Loop needs BoW, so Loop will be closed！" << endl;
    }
    /*
    if (!mbUseORB && !mbUseExistFile)
    {
        cout << endl << "Loading Pytorch Model..." << endl;

        mpModule = torch::jit::load("model/modelSP_fuse.pt");
        mpModule->to(at::kCUDA);
        assert(mpModule != nullptr);

        int Height = fsSettings["Camera.height"];//480;
        int Width = fsSettings["Camera.width"];//640;
        if(Height != 480 || Width != 640)
        {
            Height = 480;
            Width = 640;
            cout << "Detect different image size, check resize!" << endl;
        }
        mpImage = new float [1*1*Height*Width];
        
        std::ifstream inpfile("model/inp.qwe", std::ios::binary);
        inpfile.read((char*)mpImage, 1*1*Height*Width*sizeof(float));
        inpfile.close();

        cout << endl << "Pytorch Model loaded!" << endl;
    }
    */

    
    if (!mbUseORB && !mbUseExistFile)
    {
        cout << endl << "Loading Caffe Model..." << endl;
        
        string ModelPath = fsSettings["SP.ModelPath"];//"/home/yuchao/catkin_ws/src/ORB_SLAM2_SP/Model/superpoint.prototxt";
        string TrainedPath = fsSettings["SP.TrainedPath"];//"/home/yuchao/catkin_ws/src/ORB_SLAM2_SP/Model/superpoint.caffemodel";
        int SPNum = fsSettings["SP.nFeatures"];
        mpSuperPoint = new SuperPoint(ModelPath, TrainedPath, SPNum);
        cout << endl << "Caffe Model loaded!" << endl;
    }

    if (mbUseBoW)
    {
        //Load Vocabulary
        cout << endl << "Loading Vocabulary..." << endl;

        //zoe 20190719
        if (mbUseORB)
        {
            mpVocabularyORB = new ORBVocabulary();
            bool bVocLoadORB = mpVocabularyORB->loadFromTextFile(strVocFile);
            if(!bVocLoadORB)
            {
                cerr << "Wrong path to vocabulary. " << endl;
                cerr << "Falied to open at: " << strVocFile << endl;
                exit(-1);
            }
            cout << "ORB Vocabulary loaded!" << endl << endl;

            //Create KeyFrame Database
            mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabularyORB, mbUseORB);// zoe database名字没改
        
            mpTracker = new Tracking(this, mpVocabularyORB, mpFrameDrawer, mpMapDrawer,
                                mpMap, mpKeyFrameDatabase, mpPointCloudMapping, strSettingsFile, mSensor, mbOnlyTracking); //zoe 20190513 增加tracking参数
        }
        else
        {
            mpVocabularyLFNet = new LFNETVocabulary();
            bool bVocLoadLFNet = mpVocabularyLFNet->loadFromTextFile(strVocFile);
            if(!bVocLoadLFNet)
            {
                cerr << "Wrong path to vocabulary. " << endl;
                cerr << "Falied to open at: " << strVocFile << endl;
                exit(-1);
            }
            cout << "LFNET Vocabulary loaded!" << endl << endl;

            //Create KeyFrame Database
            mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabularyLFNet, mbUseORB);// zoe database名字没改

            if (mbUseExistFile)
                mpTracker = new Tracking(this, mpVocabularyLFNet, mpFrameDrawer, mpMapDrawer, 
                                mpMap, mpKeyFrameDatabase, mpPointCloudMapping, strSettingsFile, mSensor, mbOnlyTracking); //zoe 20190513 增加tracking参数
            else
                mpTracker = new Tracking(this, mpVocabularyLFNet, mpFrameDrawer, mpMapDrawer, mpMap, mpKeyFrameDatabase, mpPointCloudMapping, mpSuperPoint, strSettingsFile, mSensor, mbOnlyTracking); //zoe 2019724 增加pytorch参数
            
        }
    }
    else
    {
        if (mbUseExistFile)
            mpTracker = new Tracking(this, mpFrameDrawer, mpMapDrawer, mpMap, mpPointCloudMapping, strSettingsFile, mSensor, mbOnlyTracking, mbUseORB); //zoe 20190520
        else
            mpTracker = new Tracking(this, mpFrameDrawer, mpMapDrawer, mpMap, mpPointCloudMapping, mpSuperPoint, strSettingsFile, mSensor, mbOnlyTracking, mbUseORB); //zoe 20190724
        
    }
    //Initialize the Local Mapping thread and launch
    if (mbUseLocalMap)
    {
        mpLocalMapper = new LocalMapping(mpMap, mSensor==MONOCULAR, mbUseORB);
        mptLocalMapping = new thread(&ORB_SLAM2::LocalMapping::Run,mpLocalMapper);
    }

    //Initialize the Loop Closing thread and launch
    //zoe 20190511 关闭回环检测
    if (mbUseBoW && mbUseLoop)
    {   
        //zoe 20181016
        if (mbUseORB)
        {
            mpLoopCloser = new LoopClosing(mpMap, mpKeyFrameDatabase, mpVocabularyORB, mSensor!=MONOCULAR);
        }
        else
        {
            mpLoopCloser = new LoopClosing(mpMap, mpKeyFrameDatabase, mpVocabularyLFNet, mSensor!=MONOCULAR);
        }     
        mptLoopClosing = new thread(&ORB_SLAM2::LoopClosing::Run, mpLoopCloser);
    }

    //Initialize the Viewer thread and launch
    if(mbUseViewer)
    {
        mpViewer = new Viewer(this, mpFrameDrawer, mpMapDrawer, mpTracker, strSettingsFile, mbOnlyTracking);
        mptViewer = new thread(&Viewer::Run, mpViewer);
        mpTracker->SetViewer(mpViewer);
    }

    //Set pointers between threads
    //zoe 20190512
    if (mbUseLocalMap)
    {
        mpTracker->SetLocalMapper(mpLocalMapper);
        mpLocalMapper->SetTracker(mpTracker);
    }

    //zoe 20190511
    if (mbUseBoW && mbUseLoop)
    {
        mpTracker->SetLoopClosing(mpLoopCloser);   
        mpLocalMapper->SetLoopCloser(mpLoopCloser);
        mpLoopCloser->SetTracker(mpTracker);
        mpLoopCloser->SetLocalMapper(mpLocalMapper);

    }
}

cv::Mat System::TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp)
{
    if(mSensor!=STEREO)
    {
        cerr << "ERROR: you called TrackStereo but input sensor was not set to STEREO." << endl;
        exit(-1);
    }   

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            if(mpLocalMapper)
            {
                mpLocalMapper->RequestStop();

                // Wait until Local Mapping has effectively stopped
                while(!mpLocalMapper->isStopped())
                {
                    usleep(1000);
                }
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            if (mpLocalMapper)
            {
                mpLocalMapper->Release();
            }
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }

    cv::Mat Tcw = mpTracker->GrabImageStereo(imLeft,imRight,timestamp);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    
    if (mbUseORB)
        mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;// zoe
    else
        mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKptsUn;// zoe
    
    return Tcw;
}

cv::Mat System::TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp)
{
    if(mSensor!=RGBD)
    {
        cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << endl;
        exit(-1);
    }    

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            if(mpLocalMapper)
            {
                mpLocalMapper->RequestStop();

                // Wait until Local Mapping has effectively stopped
                while(!mpLocalMapper->isStopped())
                {
                    usleep(1000);
                }
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            if(mpLocalMapper)
            {
                mpLocalMapper->Release();
            }
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbReset)
        {
            mpTracker->Reset();
            mbReset = false;
        }
    }
    
    cv::Mat Tcw = mpTracker->GrabImageRGBD(im,depthmap,timestamp);
    
    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;

    if (mbUseORB)
        mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;// zoe mark
    else
        mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKptsUn;
    
    return Tcw;
}
//zoe 单目跟踪 暂时不用改
cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp)
{
    if(mSensor!=MONOCULAR)
    {
        cerr << "ERROR: you called TrackMonocular but input sensor was not set to Monocular." << endl;
        exit(-1);
    }

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }

    cv::Mat Tcw = mpTracker->GrabImageMonocular(im,timestamp);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    
    if (mbUseORB)
        mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;//zoe nochange
    else
        mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKptsUn;//zoe nochange

    return Tcw;
}

void System::ActivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbActivateLocalizationMode = true;
}

void System::DeactivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbDeactivateLocalizationMode = true;
}

bool System::MapChanged()
{
    static int n=0;
    int curn = mpMap->GetLastBigChangeIdx();
    if(n<curn)
    {
        n=curn;
        return true;
    }
    else
        return false;
}

void System::Reset()
{
    unique_lock<mutex> lock(mMutexReset);
    mbReset = true;
}

void System::Shutdown()
{
    if (mpLocalMapper)
        mpLocalMapper->RequestFinish();
    if (mpLoopCloser)
        mpLoopCloser->RequestFinish();
    if(mpViewer)
    {
        mpViewer->RequestFinish();
        while(!mpViewer->isFinished())
            usleep(5000);
    }

    // Wait until all thread have effectively stopped
    if (mpLocalMapper)
    {
        while(!mpLocalMapper->isFinished())
        {
            usleep(5000);
        }
    }

    if (mpLoopCloser)
    {
        while(!mpLoopCloser->isFinished() || mpLoopCloser->isRunningGBA())
        {
            usleep(5000);
        }
    }

    if(mpViewer)
        pangolin::BindToContext("ORB-SLAM2: Map Viewer");
}

void System::SaveTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << endl;
        return;
    }

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();
    for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        if(*lbL)
            continue;

        KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        while(pKF->isBad())
        {
            Trw = Trw*pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw*pKF->GetPose()*Two;

        cv::Mat Tcw = (*lit)*Trw;
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

        vector<float> q = Converter::toQuaternion(Rwc);

        f << setprecision(6) << *lT << " " <<  setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
    }
    f.close();
    cout << endl << "trajectory saved!" << endl;
}


void System::SaveKeyFrameTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    //cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];

       // pKF->SetPose(pKF->GetPose()*Two);

        if(pKF->isBad())
            continue;

        cv::Mat R = pKF->GetRotation().t();
        vector<float> q = Converter::toQuaternion(R);
        cv::Mat t = pKF->GetCameraCenter();
        f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

    }

    f.close();
    cout << endl << "trajectory saved!" << endl;
}

void System::SaveTrajectoryKITTI(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
        return;
    }
    
    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);
    
    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();
    

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    
    for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(), lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++)
    {
        ORB_SLAM2::KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        while(pKF->isBad())
        {
          //  cout << "bad parent" << endl;
            Trw = Trw*pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw*pKF->GetPose()*Two;

        cv::Mat Tcw = (*lit)*Trw;
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

        f << setprecision(9) << Rwc.at<float>(0,0) << " " << Rwc.at<float>(0,1)  << " " << Rwc.at<float>(0,2) << " "  << twc.at<float>(0) << " " <<
             Rwc.at<float>(1,0) << " " << Rwc.at<float>(1,1)  << " " << Rwc.at<float>(1,2) << " "  << twc.at<float>(1) << " " <<
             Rwc.at<float>(2,0) << " " << Rwc.at<float>(2,1)  << " " << Rwc.at<float>(2,2) << " "  << twc.at<float>(2) << endl;
    }
    f.close();
    cout << endl << "trajectory saved!" << endl;
}

int System::GetTrackingState()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackingState;
}

vector<MapPoint*> System::GetTrackedMapPoints()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedMapPoints;
}

vector<cv::KeyPoint> System::GetTrackedKeyPointsUn()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedKeyPointsUn;
}

} //namespace ORB_SLAM
