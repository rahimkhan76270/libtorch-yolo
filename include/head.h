#ifndef HEAD_H
#define HEAD_H

#include <torch/torch.h>
#include <vector>
#include <string>
#include "block.h"

class Detect : public torch::nn::Module
{
public:
    Detect(int64_t nc_=80,std::vector<int64_t> ch={});
    torch::Tensor forward(torch::Tensor x);
    torch::Tensor forward_end2end(torch::Tensor x);
    torch::Tensor _inference(torch::Tensor x);
    torch::Tensor postprocess(torch::Tensor preds, int max_det, int nc = 80);
    torch::Tensor decode_bboxes(torch::Tensor distance, 
                        torch::Tensor anchor_points, 
                        bool xywh = true);
    int64_t nc;
    int64_t nl;
    int64_t reg_max=16;
    int64_t no;
    torch::Tensor stride;
    int64_t c2;
    int64_t c3;
    torch::IntArrayRef shape;
    std::string format;
    bool end2end=false;
    bool dynamic = false ;
    bool export_ = false ;
    bool training = false;
    int64_t max_det = 300 ; 
    torch::Tensor anchors = torch::empty(0) ;
    torch::Tensor strides = torch::empty(0) ;
    bool legacy = false ;
private:
    torch::nn::ModuleList cv2{nullptr};
    torch::nn::ModuleList cv3{nullptr};
    torch::nn::ModuleList cv4{nullptr};
    torch::nn::ModuleHolder<DFL> dfl{nullptr};
    torch::nn::Identity identity{nullptr};
    torch::nn::ModuleList one2one_cv2{nullptr};
    torch::nn::ModuleList one2one_cv3{nullptr};
};

#endif