#ifndef YOLOV11_H
#define YOLOV11_H
#include <torch/torch.h>
#include <vector>
#include <string>
#include "config.h"
#include "conv.h"
#include "head.h"
#include "block.h"

class YOLOv11 : public torch::nn::Module
{
public:
    YOLOv11(std::string scale = "n",int channels=3);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Sequential main_container{nullptr};
};

#endif