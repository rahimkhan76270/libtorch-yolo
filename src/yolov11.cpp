#include "config.h"
#include "yolov11.h"
#include <vector>
#include <map>
#include <string>

YOLOv11_cfg::YOLOv11_cfg()
{
    nc=80;
    //scales start
    scales scale;
    std::map<std::string,std::vector<float>> scale_dict;
    scale_dict["n"]={0.50, 0.25, 1024};
    scale_dict["s"]={0.50, 0.50, 1024};
    scale_dict["m"]={0.50, 1.00, 512};
    scale_dict["l"]={1.00, 1.00, 512};
    scale_dict["x"]={1.00, 1.50, 512};
    scale.scale_dict=scale_dict;
    scale_config=scale;
    //scales end
    
}