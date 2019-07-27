#include <SuperPoint.hpp>
#include <stdlib.h>
#include <chrono> 

using namespace std;

class SuperPoint;
class max_heap_t;
class min_heap_t;

SuperPoint::SuperPoint(const std::string& model_file, const std::string& trained_file, int keep_k_points):KEEP_K_POINTS(keep_k_points)
{
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    net_.reset(new caffe::Net<float>(model_file, caffe::TEST));
    net_->CopyTrainedLayersFrom(trained_file);
    caffe::Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
    width_ = input_layer->width();
    height_ = input_layer->height();
}

void SuperPoint::WrapInputLayer(std::vector<cv::Mat>* input_channels)
{

    caffe::Blob<float>* input_layer = net_->input_blobs()[0];

    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i)
    {
        cv::Mat channel(height_, width_, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width_ * height_;
    }
}

void SuperPoint::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels)
{
  // Convert the input image to the input image format of the network
    cv::Mat sample;

    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;
    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);
    sample_float /= 255.0;
    cv::split(sample_float, *input_channels);
}

void SuperPoint::TopK(std::vector<point>& input_arr, int32_t n, int32_t k) 
{
    // O(k)
    // we suppose the k element of the min heap if the default top k element
    min_heap_t min_heap(input_arr, k);
    min_heap.build_heap_from_bottom_to_top();
    
    for (int32_t i = k; i < n; ++i) {
        // compare each element with the min element of the min heap
        // if the element > the min element of the min heap
        // we think may be the element is one of what we wanna to find in the top k
        if (input_arr[i].semi > min_heap.arr[0].semi){
            // swap
            min_heap.arr[0] = input_arr[i];
            
            // heap adjust
            min_heap.heap_adjust_from_top_to_bottom(0, k - 1);
        }
    }
    
    input_arr.assign(min_heap.arr.begin(),min_heap.arr.end());
}

float SuperPoint::SP_Angle(const cv::Mat& image, cv::KeyPoint& kpt, const std::vector<int>& u_max)
{
    int m_01 = 0, m_10 = 0;
    const uchar* center = &image.at<uchar> (cvRound(kpt.pt.y), cvRound(kpt.pt.x));

    // Treat the center line differently, v=0
    for (int u = -HALF_PATCH_SIZE_SP; u <= HALF_PATCH_SIZE_SP; ++u)
        m_10 += u * center[u];

    // Go line by line in the circuI853lar patch
    int step = (int)image.step1();
	
    for (int v = 1; v <= HALF_PATCH_SIZE_SP; ++v)
    {
        // Proceed over the two lines
        int v_sum = 0;
        int d = u_max[v];
        for (int u = -d; u <= d; ++u)
        {
            int val_plus = center[u + v*step], val_minus = center[u - v*step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }
	
    return cv::fastAtan2((float)m_01, (float)m_10);
}

int SuperPoint::ExactSP(const cv::Mat& image, std::vector<cv::KeyPoint>& kpts, std::vector<std::vector<float> >& dspts)
{
    cv::Mat image_angle;
    image_angle = image.clone();
    //This is for orientation
    // pre-compute the end of a row in a circular patch
    std::vector<int> umax;
    umax.resize(HALF_PATCH_SIZE_SP + 1);

    int v, v0, vmax = cvFloor(HALF_PATCH_SIZE_SP * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(HALF_PATCH_SIZE_SP * sqrt(2.f) / 2);
    const double hp2 = HALF_PATCH_SIZE_SP*HALF_PATCH_SIZE_SP;
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt(hp2 - v * v));

    // Make sure we are symmetric
    for (v = HALF_PATCH_SIZE_SP, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }

    caffe::Blob<float>* input_layer = net_->input_blobs()[0];

    input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
    
    net_->Reshape();
    
    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);

    Preprocess(image, &input_channels);
    
    net_->Forward();
    
    std::vector< caffe::Blob<float>* > output_layers = net_->output_blobs();

    float* result_semi = output_layers[0]->mutable_cpu_data();
    float* result_desc = output_layers[1]->mutable_cpu_data();

    float semi[height_][width_];
    
    for(int i=0; i<height_/Cell; i++) 
    {
        for(int j=0; j<width_/Cell; j++)
        {
            for(int kh=0; kh<Cell; kh++) 
                for(int kw=0; kw<Cell; kw++)
	            if (( kh+i*Cell <= HALF_PATCH_SIZE_SP+1 ) || ( kh+i*Cell >= height_ - HALF_PATCH_SIZE_SP - 1) || ( kw+j*Cell <= HALF_PATCH_SIZE_SP+1 ) || ( kw+j*Cell >= width_ - HALF_PATCH_SIZE_SP - 1)){
			semi[kh+i*Cell][kw+j*Cell] = -1.;
		    }else{
                    	semi[kh+i*Cell][kw+j*Cell] = result_semi[kw+kh*Cell+j*Feature_Length+i*Feature_Length*width_/Cell];
		    }
        }
    }
    std::vector<point> tmp_point;
    //NMS
    for(int i=0; i<height_; i++) 
    {
        for(int j=0; j<width_; j++)
         {
            if(semi[i][j] != 0) 
            {
                float tmp_semi = semi[i][j];
                for(int kh=std::max(0,i-NMS_Threshold); kh<std::min(height_,i+NMS_Threshold+1); kh++)
                    for(int kw=std::max(0,j-NMS_Threshold); kw<std::min(width_,j+NMS_Threshold+1); kw++)
                        if(i!=kh||j!=kw) 
                        {
                            if(tmp_semi>=semi[kh][kw])
                                semi[kh][kw] = 0;
                            else
                                semi[i][j] = 0;
                        }
                if(semi[i][j]!=0)
                    tmp_point.push_back(point(i,j,semi[i][j]));
            }
        }
    }
    TopK(tmp_point,tmp_point.size(), KEEP_K_POINTS);

    dspts.clear();
    kpts.clear();
    kpts.resize(tmp_point.size());
    
    for(int i=0; i<tmp_point.size(); i++) 
    {
        kpts[i].pt.x = tmp_point[i].W;
        kpts[i].pt.y = tmp_point[i].H;

        kpts[i].angle = SP_Angle(image_angle, kpts[i], umax); //20190511 zoe
        kpts[i].octave = 0;

        std::vector<float> dspt;
        dspt.resize(D);
        int x1 = int(tmp_point[i].W / Cell);
        int x2 = int(tmp_point[i].H / Cell);
        for (int j = 0; j < D; j++)
        {
	        dspt[j]  = result_desc[x1*D + x2*(D*width_/Cell) +j];            
        }
        dspts.push_back(dspt);
    }
    return 0;
}
