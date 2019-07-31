#include <SuperPoint.hpp>
#include <stdlib.h>
#include <chrono> 

using namespace std;

class SuperPoint;
class max_heap_t;
class min_heap_t;

//xzl 190729
SuperPoint::SuperPoint(const std::string& model_file, const std::string& trained_file, int keep_k_points):KEEP_K_POINTS(keep_k_points)
{
    // caffe::Caffe::set_mode(caffe::Caffe::GPU);
    // net_.reset(new caffe::Net<float>(model_file, caffe::TEST));
    // net_->CopyTrainedLayersFrom(trained_file);
    // caffe::Blob<float>* input_layer = net_->input_blobs()[0];
    // num_channels_ = input_layer->channels();
    // input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
    // width_ = input_layer->width();
    // height_ = input_layer->height();
    device_setup();
}

//xzl 190729
SuperPoint::~SuperPoint()
{
    device_close();
}

//xzl 190729
void* SuperPoint::memory_map(unsigned int map_size, off_t base_addr, int memfd) //map_size = n MByte
{
    void *mapped_base;
    mapped_base = mmap(0, map_size*1024*1024, PROT_READ | PROT_WRITE, MAP_SHARED
, memfd, base_addr);
    if (mapped_base == (void *) -1) {
        printf("Can't map memory to user space.\n");
        exit(0);
    }
#ifdef DEBUG
    printf("Memory mapped at address %p.\n", mapped_base);
#endif
    return mapped_base;
}

//xzl 190729
void SuperPoint::device_setup()
{
    // Attach to DPU driver and prepare for running
    dpuOpen();

    // Load DPU Kernel for DenseBox neural network
    SPtask.kernel = dpuLoadKernel("superpoint");

	  //cout<<"SPtask.task:"<<SPtask.task<<endl;
    SPtask.task = dpuCreateTask(SPtask.kernel, 0);
    //cout<<"SPtask.task:"<<SPtask.task<<endl;
    
    off_t   reg_base = REG_BASE_ADDRESS;
    off_t   ddr1_base = DDR1_BASE_ADDRESS;
    off_t   ddr2_base = DDR2_BASE_ADDRESS;
    off_t   inst_base = INST_BASE_ADDRESS;  
    
    //printf("open\n");
    SPtask.memfd = open("/dev/mem", O_RDWR | O_SYNC);
    //printf("reg\n");
    SPtask.mapped_reg_base = memory_map(1, reg_base, SPtask.memfd);
    //printf("ddr1\n");
    SPtask.mapped_ddr1_base = memory_map(1024, ddr1_base, SPtask.memfd);
    //printf("ddr2\n");
    SPtask.mapped_ddr2_base = memory_map(1024, ddr2_base, SPtask.memfd);
    //printf("inst\n");
    SPtask.mapped_inst_base = memory_map(1024, inst_base, SPtask.memfd);
    //printf("finish\n");
    
    SPtask.mapped_softmax_reg_base = SPtask.mapped_reg_base;
    SPtask.mapped_normalize_reg_base = SPtask.mapped_reg_base + 0x1000;
    
    //reset DMA
    writel(SPtask.mapped_normalize_reg_base+0x00,0x00000004);
    writel(SPtask.mapped_normalize_reg_base+0x00,0x00000000);
}

//xzl 190729
void SuperPoint::device_close()
{
    dpuDestroyTask(SPtask.task);

    // Destroy DPU Kernel & free resources
    dpuDestroyKernel(SPtask.kernel);

    // Dettach from DPU driver & release resources
    dpuClose();
}

//xzl 190729
void SuperPoint::run_DPU(cv::Mat img, int8_t* &result_semi_int, int8_t* &result_desc_int, int &num_semi, int &num_desc)
{
    assert(SPtask.task);
    
    int num = dpuGetInputTensorSize(SPtask.task, INPUT_NODE);
    //cout << "input num:" << num << endl;
    int8_t* input_img = new int8_t[num]();
    uint8_t* data = (uint8_t*)img.data;
    for(int i=0; i<num; i++) {
        input_img[i] = (int8_t)(data[i]/2);
    }
    
    dpuSetInputTensorInHWCInt8(SPtask.task, INPUT_NODE, (int8_t*)input_img, num);
    
    //cout << "Run DPU ..." << endl;
    //cout<<"SPtask.task:"<<SPtask.task<<endl;
    dpuRunTask(SPtask.task);
    //cout << "Finish DPU ..." << endl;
    
    num_semi = dpuGetOutputTensorSize(SPtask.task, OUTPUT_NODE_semi);
    num_desc = dpuGetOutputTensorSize(SPtask.task, OUTPUT_NODE_desc);
    
    DPUTensor* semi_tensor = dpuGetOutputTensorInHWCInt8(SPtask.task, OUTPUT_NODE_semi);
    result_semi_int = dpuGetTensorAddress(semi_tensor);
    DPUTensor* desc_tensor = dpuGetOutputTensorInHWCInt8(SPtask.task, OUTPUT_NODE_desc);
    result_desc_int = dpuGetTensorAddress(desc_tensor);
    
    delete[] input_img;
}

//xzl 190729
void SuperPoint::run_Softmax_fpga(int8_t* result_semi_int, int num_semi, point* coarse_semi[])
{
    memcpy(SPtask.mapped_ddr1_base,result_semi_int,num_semi);
    
    //start
    writel(SPtask.mapped_softmax_reg_base,0x00000000);
    //printf("reset\n");
    writel(SPtask.mapped_softmax_reg_base,0x000000aa);
    //printf("written gpio\n");
    
    //wait
    unsigned int a;
    int wait_num=0;
    do
    {
        usleep(200);
        a = readl(SPtask.mapped_softmax_reg_base);
        //cout<<"wait"<<endl;
        wait_num++;
    }while(!(a&0x00000001) && wait_num<100);
    if(wait_num>=10)return;
    
    //read result
    uint16_t* result_softmax = (uint16_t*)SPtask.mapped_ddr2_base;
    
    for(int i=0; i<Height/Cell; i++) {
        for(int j=0; j<Width/Cell; j++) {
            int channel_id = result_softmax[(i*Width/Cell+j)*2+1];
            coarse_semi[i][j].H = i*Cell + channel_id/Cell;
            coarse_semi[i][j].W = j*Cell + channel_id%Cell;
            coarse_semi[i][j].semi = result_softmax[(i*Width/Cell+j)*2];
            coarse_semi[i][j].num = i*Width/Cell+j;
      
            if(coarse_semi[i][j].num>Height/Cell*Width/Cell || coarse_semi[i][j].num<0)
            {
                cout<<"coarse_semi[i][j].num:"<< coarse_semi[i][j].num<<endl;
                return;
            }
        }
    }
}

//xzl 190729
void SuperPoint::run_NMS(point* coarse_semi[], vector<point> &tmp_point, int threshold)
{
    for(int i=0; i<Height/Cell; i++) {
        for(int j=0; j<Width/Cell; j++) {
            if(coarse_semi[i][j].semi != 65535) {
                if(( coarse_semi[i][j].H <= HALF_PATCH_SIZE_SP+1 ) || ( coarse_semi[i][j].H >= Height - HALF_PATCH_SIZE_SP - 1) || ( coarse_semi[i][j].W <= HALF_PATCH_SIZE_SP+1 ) || ( coarse_semi[i][j].W >= Width - HALF_PATCH_SIZE_SP - 1)){
                    coarse_semi[i][j].semi = 65535;
                }
                else
                {
                    float tmp_semi = coarse_semi[i][j].semi;
                    for(int kh=max(0,i-1); kh<min(Height/Cell,i+1+1); kh++)
                        for(int kw=max(0,j-1); kw<min(Width/Cell,j+1+1); kw++)
                            if(i!=kh||j!=kw) {
                                if(abs(coarse_semi[i][j].H-coarse_semi[kh][kw].H)<=threshold && abs(coarse_semi[i][j].W-coarse_semi[kh][kw].W)<=threshold) {
                                    if(tmp_semi<=coarse_semi[kh][kw].semi)
                                        coarse_semi[kh][kw].semi = 65535;
                                    else
                                        coarse_semi[i][j].semi = 65535;
                                }
                            }
                }
                if(coarse_semi[i][j].semi != 65535)
                    tmp_point.push_back(coarse_semi[i][j]);
            }
        }
    }
}

//xzl 190729
void SuperPoint::run_Normalize_fpga(int8_t* result_desc_int, int num_desc, std::vector<point> tmp_point, std::vector<std::vector<float> >& desc)
{
    memcpy(SPtask.mapped_ddr1_base, result_desc_int, num_desc);
    //cout<<"unnorm data copy"<<endl;
    //--------------------sg write------------------------
    unsigned int* sg = (unsigned int*)SPtask.mapped_inst_base;
    //read-sg
    for(int i=0; i<tmp_point.size()*2; i++) {
        for(int j=0; j<16; j++)
            sg[i*16+j] = 0;
        sg[i*16] = INST_BASE_ADDRESS+(i+1)*64;
        sg[i*16+2] = DDR1_BASE_ADDRESS + tmp_point[i/2].num*D;
        sg[i*16+6] = 0x0C000000+D;
        if(sg[i*16+2]>DDR1_BASE_ADDRESS + Height/Cell*Width/Cell*D)
        {
            cout<<"tmp_point[i/2].num:"<< tmp_point[i/2].num<<endl;
            return;
        }
    }
    //write-sg
    for(int i=tmp_point.size()*2; i<tmp_point.size()*3; i++) {
        for(int j=0; j<16; j++)
            sg[i*16+j] = 0;
        sg[i*16] = INST_BASE_ADDRESS+(i+1)*64;
        sg[i*16+2] = DDR2_BASE_ADDRESS + (i-tmp_point.size()*2)*D;
        sg[i*16+6] = 0x0C000000+D;
    }
    
    //read state
    unsigned int a,b;
    a = readl(SPtask.mapped_normalize_reg_base+0x04);
    //printf("MM2S DMA Status register:%x\n", a);
    a = readl(SPtask.mapped_normalize_reg_base+0x34);
    //printf("S2MM DMA Status register:%x\n", a);
    
    //start norm
    //cout<<"start norm"<<endl;
    writel(SPtask.mapped_normalize_reg_base+0x30,0x00027004);
    //printf("write reg\n");
    writel(SPtask.mapped_normalize_reg_base+0x30,0x00027000);
    //printf("rst\n");
    writel(SPtask.mapped_normalize_reg_base+0x38,INST_BASE_ADDRESS+tmp_point.size()*2*16*4);//
    //printf("1\n");
    writel(SPtask.mapped_normalize_reg_base+0x30,0x00027001);
    //printf("2\n");
    writel(SPtask.mapped_normalize_reg_base+0x40,INST_BASE_ADDRESS+(tmp_point.size()*3-1)*16*4);//
    //printf("3\n");
    
    writel(SPtask.mapped_normalize_reg_base+0x00,0x00027000);
    //printf("rst\n");
    writel(SPtask.mapped_normalize_reg_base+0x08,INST_BASE_ADDRESS+0x00000000);//
    //printf("4\n");
    writel(SPtask.mapped_normalize_reg_base+0x00,0x00027001);
    //printf("5\n"); 
    writel(SPtask.mapped_normalize_reg_base+0x10,INST_BASE_ADDRESS+(tmp_point.size()*2-1)*16*4);//
    //printf("write reg finish \n");
    
    //wait
    int wait_num=0;
    do
    {
        usleep(50);
        a = readl(SPtask.mapped_normalize_reg_base+0x34);
        //cout<<"wait"<<endl;
        wait_num++;
    }while(!(a&0x00000002) && wait_num<100);
    
    if(wait_num>=10)
    {
        a = readl(SPtask.mapped_normalize_reg_base+0x04);
        printf("MM2S DMA Status register:%x\n", a);
        a = readl(SPtask.mapped_normalize_reg_base+0x34);
        printf("S2MM DMA Status register:%x\n", a);
        // a = readl(SPtask.mapped_normalize_reg_base+0x08);
        // printf("MM2S DMA CURDESC:%x\n", a);
        // b = readl(SPtask.mapped_inst_base+a-INST_BASE_ADDRESS+0);
        // printf("NXTDESC:%x\n", b);
        // b = readl(SPtask.mapped_inst_base+a-INST_BASE_ADDRESS+4);
        // printf("NXTDESC_MSB:%x\n", b);
        // b = readl(SPtask.mapped_inst_base+a-INST_BASE_ADDRESS+8);
        // printf("BUFFER_ADDRESS:%x\n", b);
        // a = readl(SPtask.mapped_normalize_reg_base+0x38);
        // printf("S2MM DMA CURDESC:%x\n", a);
        
        return;
    }
    
    int8_t* result_norm = (int8_t*)SPtask.mapped_ddr2_base;
    
    //output desc
    desc.clear();
    
    for(int i=0;i<tmp_point.size();i++)
    {
        std::vector<float> dspt;
        dspt.resize(D);
        for (int j = 0; j < D; j++)
        {
	        dspt[j]  = result_norm[j+i*D];            
        }
        desc.push_back(dspt);
    }
}

/* void SuperPoint::WrapInputLayer(std::vector<cv::Mat>* input_channels)
{

    caffe::Blob<float>* input_layer = net_->input_blobs()[0];

    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i)
    {
        cv::Mat channel(height_, width_, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width_ * height_;
    }
} */

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

void SuperPoint::Top_K(std::vector<point>& input_arr, int32_t n, int32_t k) 
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

void SuperPoint::Bottom_K(std::vector<point>& input_arr, int32_t n, int32_t k) 
{
    // O(k)
    // we suppose the k element of the max heap if the default top k element
    max_heap_t max_heap(input_arr, k);
    max_heap.build_heap_from_bottom_to_top();
    
    for (int32_t i = k; i < n; ++i) {
        // compare each element with the max element of the max heap
        // if the element < the max element of the max heap
        // we think may be the element is one of what we wanna to find in the top k
        if (input_arr[i].semi < max_heap.arr[0].semi){
            // swap
            max_heap.arr[0] = input_arr[i];
            
            // heap adjust
            max_heap.heap_adjust_from_top_to_bottom(0, k - 1);
        }
    }
    
    input_arr.assign(max_heap.arr.begin(),max_heap.arr.end());
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
    
    dspts.clear();
    kpts.clear();
    
    //xzl 190729
    //-------------------------------DPU----------------------------
    int num_semi;
    int8_t* result_semi_int;
    int num_desc;
    int8_t* result_desc_int;
    
    run_DPU(image, result_semi_int, result_desc_int, num_semi, num_desc);
    
    //------------------------softmax----------------------------------
    point* coarse_semi[Height/Cell];
    for(int i=0; i<Height/Cell; i++)
    {
        coarse_semi[i]=new point[Width/Cell]();
    }
    
    run_Softmax_fpga(result_semi_int, num_semi, coarse_semi);
    
    //---------------------------------NMS---------------------------------
    vector<point> tmp_point;
    
    run_NMS(coarse_semi, tmp_point, NMS_Threshold);
    //cout<<"tmp_point.size:"<<tmp_point.size()<<endl;
    
    for(int i=0; i<Height/Cell; i++)
    {
        delete[] coarse_semi[i];
    }
    
    //--------------------------------rank------------------------------
    if(tmp_point.size()>KEEP_K_POINTS)
    {
        Bottom_K(tmp_point,tmp_point.size(),KEEP_K_POINTS);
    }
    //cout<<"tmp_point.size:"<<tmp_point.size()<<endl;
    
    kpts.resize(tmp_point.size());
    for(int i=0;i<tmp_point.size();i++)
    {
        kpts[i].pt.x = tmp_point[i].W;
        kpts[i].pt.y = tmp_point[i].H;

        kpts[i].angle = SP_Angle(image_angle, kpts[i], umax); //20190511 zoe
        kpts[i].octave = 0;
    }
    
    //-------------------------------normalize----------------------
    run_Normalize_fpga(result_desc_int, num_desc, tmp_point, dspts);
    
    return 0;
}
