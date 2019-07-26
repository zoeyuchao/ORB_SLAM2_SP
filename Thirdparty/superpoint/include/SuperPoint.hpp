#ifndef _SUPERPOINT_H_
#define _SUPERPOINT_H_

#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdlib.h>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <sstream>

class point
{
    public:
        int W;   
        int H;  
        float semi;   
        point(int a, int b, float c) {H=a;W=b;semi=c;}
        point() {}
};

class SuperPoint
{
  public:
    SuperPoint(const std::string& model_file, const std::string& trained_file, int keep_k_points);
    int ExactSP(const cv::Mat& image, std::vector<cv::KeyPoint>& kpts, std::vector<std::vector<float> >& dspts);

  private:
    caffe::shared_ptr<caffe::Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;  
    int height_;
    int width_; 

    int Cell = 8;
    int D = 256;
    int Feature_Length = 65;
    int NMS_Threshold = 4;
    int KEEP_K_POINTS = 200;
    int HALF_PATCH_SIZE_SP = 15;

    void WrapInputLayer(std::vector<cv::Mat>* input_channels);
    void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);
    float SP_Angle(const cv::Mat& image, cv::KeyPoint& kpt, const std::vector<int>& u_max);
    void TopK(std::vector<point>& input_arr, int32_t n, int32_t k);
};



class max_heap_t
{
    public:
    
    std::vector<point> arr;
    int32_t  n;

    max_heap_t (std::vector<point> input_arr, int32_t arr_size){
        arr.assign(input_arr.begin(),input_arr.begin()+arr_size);
        n = arr_size;
    }

    ~max_heap_t () {
    }

    /* time complexity => O(nlogn) */
    void    build_heap_from_top_to_bottom() {
      
        for (int32_t i = 1; i < n; i++) {
           heap_ajust_from_bottom_to_top(i);
        }
    }

    /* O(logn) */
    void    heap_ajust_from_bottom_to_top(int32_t bottom_index) {
        point tmp = arr[bottom_index];
        while (bottom_index > 0) {
            int32_t parent_index = (bottom_index - 1) / 2;
            if (arr[parent_index].semi < tmp.semi ) {
                arr[bottom_index] = arr[parent_index];
                bottom_index = parent_index;
            }
            else {
                break;
            }
        }
        arr[bottom_index] = tmp;
    }

     /* O(n) */
    void    build_heap_from_bottom_to_top() {
        int32_t max_index = n - 1;
        for (int32_t i = (max_index - 1) / 2; i >= 0; i--) {
            heap_adjust_from_top_to_bottom(i, max_index);
        }
    }

    /* O(logn) */
    void    heap_adjust_from_top_to_bottom(int32_t top_index, int32_t bottom_index) {
        point tmp = arr[top_index];
        while (top_index <= (bottom_index - 1) / 2) {
            point max_one = tmp;
            int32_t child_idx = 0;
            int32_t left_child_idx = top_index * 2 + 1;
            int32_t right_child_idx = top_index * 2 + 2;
            
            if (left_child_idx <= bottom_index && max_one.semi < arr[left_child_idx].semi ) {
                max_one = arr[left_child_idx];
                child_idx = left_child_idx;
            }
            if (right_child_idx <= bottom_index && max_one.semi < arr[right_child_idx].semi ) {
                max_one = arr[right_child_idx];
                child_idx = right_child_idx;
            }
          
            if (max_one.semi != tmp.semi) {
                arr[top_index] = max_one;
                top_index = child_idx;
            }
            else {
                break;
            }
        }
        arr[top_index] = tmp;
    }

    void    sort() {
        // build  heap first
        build_heap_from_bottom_to_top();

        // sort
        point tmp;
        for (int32_t i = n - 1; i > 0;) {
            // move heap top to end
            tmp = arr[0];
            arr[0] = arr[i];
            arr[i] = tmp;

            // adjust the heap
            heap_adjust_from_top_to_bottom(0, --i);
        }
    }

};

class min_heap_t 
{
    public:
    
    std::vector<point> arr;
    int32_t  n;

    min_heap_t (std::vector<point> input_arr, int32_t arr_size){
        arr.assign(input_arr.begin(),input_arr.begin()+arr_size);
        n = arr_size;
    }

    ~min_heap_t () {
    }

    /* time complexity => O(nlogn) */
    void    build_heap_from_top_to_bottom() {
      
        for (int32_t i = 1; i < n; i++) {
           heap_ajust_from_bottom_to_top(i);
        }
    }

    /* O(logn) */
    void    heap_ajust_from_bottom_to_top(int32_t bottom_index) {
        point tmp = arr[bottom_index];
        while (bottom_index > 0) {
            int32_t parent_index = (bottom_index - 1) / 2;
            if (arr[parent_index].semi > tmp.semi ) {
                arr[bottom_index] = arr[parent_index];
                bottom_index = parent_index;
            }
            else {
                break;
            }
        }
        arr[bottom_index] = tmp;
    }

     /* O(n) */
    void    build_heap_from_bottom_to_top() {
        int32_t max_index = n - 1;
        for (int32_t i = (max_index - 1) / 2; i >= 0; i--) {
            heap_adjust_from_top_to_bottom(i, max_index);
        }
    }

    /* O(logn) */
    void    heap_adjust_from_top_to_bottom(int32_t top_index, int32_t bottom_index) {
        point tmp = arr[top_index];
        while (top_index <= (bottom_index - 1) / 2) {
            point max_one = tmp;
            int32_t child_idx = 0;
            int32_t left_child_idx = top_index * 2 + 1;
            int32_t right_child_idx = top_index * 2 + 2;
            
            if (left_child_idx <= bottom_index && max_one.semi > arr[left_child_idx].semi ) {
                max_one = arr[left_child_idx];
                child_idx = left_child_idx;
            }
            if (right_child_idx <= bottom_index && max_one.semi > arr[right_child_idx].semi ) {
                max_one = arr[right_child_idx];
                child_idx = right_child_idx;
            }
          
            if (max_one.semi != tmp.semi) {
                arr[top_index] = max_one;
                top_index = child_idx;
            }
            else {
                break;
            }
        }
        arr[top_index] = tmp;
    }

    void    sort() {
        // build  heap first
        build_heap_from_bottom_to_top();

        // sort
        point tmp;
        for (int32_t i = n - 1; i > 0;) {
            // move heap top to end
            tmp = arr[0];
            arr[0] = arr[i];
            arr[i] = tmp;

            // adjust the heap
            heap_adjust_from_top_to_bottom(0, --i);
        }
    }

};

#endif
