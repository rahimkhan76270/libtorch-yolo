#include "yolov11.h"
#include <iostream>

int main()
{
    auto yolo =std::make_shared<YOLOv11>("n");
    auto sample=torch::rand({1,3,64,64});
    auto y=yolo->forward(sample);
    std::cout<<y<<std::endl;
    return 0;
}