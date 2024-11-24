#include "conv.h"
#include <vector>
#include <optional>

int64_t autopad(int64_t k, std::optional<int64_t> p, int64_t d) {
    if (d > 1) {
        k = d * (k - 1) + 1;
    }
    if (!p.has_value()) {
        p = k / 2; 
    }
    return p.value();
}

std::vector<int64_t> autopad(std::vector<int64_t> k, std::optional<std::vector<int64_t>> p, int64_t d) {
    std::vector<int64_t> adjusted_k = k;
    if (d > 1) {
        for (size_t i = 0; i < adjusted_k.size(); ++i) {
            adjusted_k[i] = d * (adjusted_k[i] - 1) + 1;
        }
    }
    if (!p.has_value()) {
        std::vector<int64_t> auto_padded_k;
        for (size_t i = 0; i < adjusted_k.size(); ++i) {
            auto_padded_k.push_back(adjusted_k[i] / 2);
        }
        return auto_padded_k;
    }
    return p.value();
}

Conv::Conv(int64_t c1,int64_t c2,int64_t k,int64_t s,std::optional<int64_t> p,int64_t g,int64_t d,bool act)
{
    conv=register_module("conv",torch::nn::Conv2d(torch::nn::Conv2dOptions(c1,c2,k).stride(s).padding(autopad(k,p,d)).groups(g).dilation(d).bias(false)));
    bn  =register_module("bn",torch::nn::BatchNorm2d(c2));
    identity=register_module("identity",torch::nn::Identity());
    default_act=register_module("default_act",torch::nn::SiLU());
    activation=act;
}

Conv::Conv(int64_t c1,int64_t c2,std::vector<int64_t> k,int64_t s,std::optional<std::vector<int64_t>> p,int64_t g,int64_t d,bool act)
{
    conv=register_module("conv",torch::nn::Conv2d(torch::nn::Conv2dOptions(c1,c2,k).stride(s).padding(autopad(k,p,d)).groups(g).dilation(d).bias(false)));
    bn  =register_module("bn",torch::nn::BatchNorm2d(c2));
    identity=register_module("identity",torch::nn::Identity());
    default_act=register_module("default_act",torch::nn::SiLU());
    activation=act;
}

torch::Tensor Conv::forward(torch::Tensor x)
{
    if(this->activation)
    {
        return default_act->forward(bn->forward(conv->forward(x)));
    }
    else
    {
        return identity->forward(bn->forward(conv->forward(x)));
    }
}

torch::Tensor Conv::forward_fuse(torch::Tensor x)
{
    if(this->activation)
    {
        return default_act->forward(conv->forward(x));
    }
    else
    {
        return identity->forward(conv->forward(x));
    }
}

Conv2::Conv2(int64_t c1, int64_t c2, int64_t k, int64_t s, std::optional<int64_t> p, int64_t g, int64_t d, bool act )
{
    conv=register_module("conv",torch::nn::Conv2d(torch::nn::Conv2dOptions(c1,c2,k).stride(s).padding(autopad(k,p,d)).groups(g).dilation(d).bias(false)));
    cv2=register_module("cv2",torch::nn::Conv2d(torch::nn::Conv2dOptions(c1,c2,1).stride(s).padding(autopad(1,p,d)).groups(g).dilation(d).bias(false)));
    bn  =register_module("bn",torch::nn::BatchNorm2d(c2));
    identity=register_module("identity",torch::nn::Identity());
    default_act=register_module("default_act",torch::nn::SiLU());
    activation=act;
}

torch::Tensor Conv2::forward(torch::Tensor x)
{
    if(this->activation)
    {
        return default_act->forward(bn->forward(conv->forward(x)+cv2->forward(x)));
    }
    else
    {
        return identity->forward(bn->forward(conv->forward(x)+cv2->forward(x)));
    }
}

torch::Tensor Conv2::forward_fuse(torch::Tensor x)
{
    if(this->activation)
    {
        return default_act->forward(bn->forward(conv->forward(x)));
    }
    else
    {
        return identity->forward(bn->forward(conv->forward(x)));
    }
}

Concat::Concat(int64_t dimension) : dimension_(dimension) {}
torch::Tensor Concat::forward(const std::vector<torch::Tensor>& inputs) {
    return torch::cat(inputs, dimension_);
}

DWConv::DWConv(int64_t c1, int64_t c2, int64_t k, int64_t s, int64_t d, bool act)
    : Conv(c1, c2, k, s, std::nullopt, std::gcd(c1, c2), d, act) {}