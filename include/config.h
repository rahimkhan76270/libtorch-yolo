#ifndef CONFIG_H
#define CONFIG_H
#include <torch/torch.h>
#include <variant>
#include <vector>
#include <map>
#include <string>

using ParamType = std::variant<int, float, bool,std::vector<int>,std::string>;

class backbone_layer_config{
    public:
        int from =-1;
        int repeats=1;
        torch::nn::Module module;
        std::vector<ParamType> layer_config;
        // backbone_layer_config(int from,
        //                     int repeats,
        //                     torch::nn::Module module,
        //                     std::vector<ParamType> layer_config);
};
class scales
{
    public:
        std::map<std::string,std::vector<float>> scale_dict;
        // scales(std::map<std::string,std::vector<float>> scales);
};

class backbone{
    public:
        std::vector<backbone_layer_config> config;
        // backbone(std::vector<backbone_layer_config> config);
};

class head_layer_config{
    public:
        ParamType param1;
        int param2;
        torch::nn::Module layer;
        std::vector<ParamType> layer_config;
        // head_layer_config(ParamType param1,int param2,torch::nn::Module layer,std::vector<ParamType> layer_config);
};

class head{
    public:
        std::vector<head_layer_config> head_vector;
        // head(std::vector<head_layer_config> head_vector);
};
class YOLO_cfg{
    public:
        int nc=80;
        scales scale_config;
        backbone backbone_config;
        head head_config;
};
#endif