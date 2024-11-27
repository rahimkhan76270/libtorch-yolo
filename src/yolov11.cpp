#include "config.h"
#include "yolov11.h"
#include <vector>
#include <map>
#include <string>

int make_divisible(float x,int divisor)
{
    return std::ceil(x/divisor)*divisor;
}

YOLOv11::YOLOv11(std::string scale,int ch)
{
    int nc=80;
    std::map<std::string,std::vector<float>> scale_dict;
    scale_dict["n"]={0.50, 0.25, 1024};
    scale_dict["s"]={0.50, 0.50, 1024};
    scale_dict["m"]={0.50, 1.00, 512};
    scale_dict["l"]={1.00, 1.00, 512};
    scale_dict["x"]={1.00, 1.50, 512};
    std::vector<float> scale_vec=scale_dict[scale];
    float depth=scale_vec[0];
    float width =scale_vec[1];
    int max_channels =static_cast<int>(scale_vec[2]);
    std::vector<int> ch_={ch};
    int c1=ch;
    //backbone start
    int c2=make_divisible(std::min(64,max_channels),8);
    ch_.push_back(c2);
    torch::nn::ModuleHolder<Conv> conv1(Conv(c1,c2,3,2));
    c1=c2;
    c2=make_divisible(std::min(128,max_channels),8);
    ch_.push_back(c2);
    torch::nn::ModuleHolder<Conv> conv2(Conv(c1,c2,3,2));
    c1=c2;
    c2=make_divisible(std::min(256,max_channels),8);
    ch_.push_back(c2);
    torch::nn::ModuleHolder<C3k2> c3k2_1(C3k2(c1,c2,2,false,0.25));
    c1=c2;
    c2=make_divisible(std::min(256,max_channels),8);
    ch_.push_back(c2);
    torch::nn::ModuleHolder<Conv> conv3(Conv(c1,c2,3,2));
    c1=c2;
    c2=make_divisible(std::min(512,max_channels),8);
    ch_.push_back(c2);
    torch::nn::ModuleHolder<C3k2> c3k2_2(C3k2(c1,c2,2,false,0.25));
    c1=c2;
    c2=make_divisible(std::min(512,max_channels),8);
    ch_.push_back(c2);
    torch::nn::ModuleHolder<Conv> conv4(Conv(c1,c2,3,2));
    c1=c2;
    c2=make_divisible(std::min(512,max_channels),8);
    ch_.push_back(c2);
    torch::nn::ModuleHolder<C3k2> c3k2_3(C3k2(c1,c2,2,true));
    c1=c2;
    c2=make_divisible(std::min(1024,max_channels),8);
    ch_.push_back(c2);
    torch::nn::ModuleHolder<Conv> conv5(Conv(c1,c2,3,2));
    c1=c2;
    c2=make_divisible(std::min(1024,max_channels),8);
    ch_.push_back(c2);
    torch::nn::ModuleHolder<C3k2> c3k2_4(C3k2(c1,c2,2,true));
    c1=c2;
    c2=make_divisible(std::min(1024,max_channels),8);
    ch_.push_back(c2);
    torch::nn::ModuleHolder<SPPF> sppf(SPPF(c1,c2,5));
    c1=c2;
    c2=make_divisible(std::min(1024,max_channels),8);
    ch_.push_back(c2);
    torch::nn::ModuleHolder<C2PSA> c2psa(C2PSA(c1,c2,2));
    // backbone end

    // head start
    torch::nn::ModuleHolder<Upsample> upsample1(Upsample(2));
    torch::nn::ModuleHolder<Concat> concat1(Concat(1));
    //for concat
    c1=c2;
    c2=ch_.back()+ch_[6];
    ch_.push_back(c2);
    //c3k2
    c1=c2;
    c2=make_divisible(std::min(512,max_channels),8);
    ch_.push_back(c2);
    torch::nn::ModuleHolder<C3k2> c3k2_5(C3k2(c1,c2,2,false));
    torch::nn::ModuleHolder<Upsample> upsample2(Upsample(2));
    c1=c2;
    c2=ch_.back()+ch_[4];
    ch_.push_back(c2);
    torch::nn::ModuleHolder<Concat> concat2{nullptr};
    c1=c2;
    c2=make_divisible(std::min(256,max_channels),8);
    ch_.push_back(c2);
    torch::nn::ModuleHolder<C3k2> c3k2_6(C3k2(c1,c2,2,false));
    c1=c2;
    c2=make_divisible(std::min(256,max_channels),8);
    ch_.push_back(c2);
    torch::nn::ModuleHolder<Conv> conv6(Conv(c1,c2,3,2));
    c1=c2;
    c2=ch_.back()+ch_[13];
    ch_.push_back(c2);
    torch::nn::ModuleHolder<Concat> concat3(Concat(1));
    c1=c2;
    c2=make_divisible(std::min(512,max_channels),8);
    ch_.push_back(c2);
    torch::nn::ModuleHolder<C3k2> c3k2_7(C3k2(c1,c2,2,false));

    c1=c2;
    c2=make_divisible(std::min(512,max_channels),8);
    ch_.push_back(c2);
    torch::nn::ModuleHolder<Conv> conv7(Conv(c1,c2,3,2));
    c1=c2;
    c2=ch_.back()+ch_[10];
    ch_.push_back(c2);
    torch::nn::ModuleHolder<Concat> concat4(Concat(1));
    c1=c2;
    c2=make_divisible(std::min(1024,max_channels),8);
    ch_.push_back(c2);
    torch::nn::ModuleHolder<C3k2> c3k2_8(C3k2(c1,c2,2,true));

    torch::nn::ModuleHolder<Detect> detect(Detect(nc,{16,19,22}));
    // head end

    main_container=register_module("main_container",torch::nn::Sequential());

    main_container->push_back(conv1);
    main_container->push_back(conv2);
    main_container->push_back(c3k2_1);
    main_container->push_back(conv3);
    main_container->push_back(c3k2_2);
    main_container->push_back(conv4);
    main_container->push_back(c3k2_3);
    main_container->push_back(conv5);
    main_container->push_back(c3k2_4);
    main_container->push_back(sppf);
    main_container->push_back(c2psa);

    main_container->push_back(upsample1);
    main_container->push_back(concat1);
    main_container->push_back(c3k2_5);

    main_container->push_back(upsample2);
    main_container->push_back(concat2);
    main_container->push_back(c3k2_6);

    main_container->push_back(conv6);
    main_container->push_back(concat3);
    main_container->push_back(c3k2_7);

    main_container->push_back(conv7);
    main_container->push_back(concat4);
    main_container->push_back(c3k2_8);

    main_container->push_back(detect);
}

torch::Tensor YOLOv11::forward(torch::Tensor x)
{
    return main_container->forward(x);
}