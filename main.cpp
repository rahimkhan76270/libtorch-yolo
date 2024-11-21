#include "yolov11.h"
#include <iostream>

int main()
{
    YOLOv11_cfg* v11_cfg=new YOLOv11_cfg();
    for(auto pair:v11_cfg->scale_config.scale_dict)
    {
        std::cout<<pair.first<<" "<<pair.second<<std::endl;
    }
    return 0;
}