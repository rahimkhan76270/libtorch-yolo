#include "block.h"
#include "conv.h"
#include "config.h"
#include <vector>
#include <stdexcept>
#include <math.h>

Bottleneck::Bottleneck(int64_t c1, int64_t c2, bool shortcut, int64_t g, SingleKernel k, float e) : c_(static_cast<int64_t>(c2 * e)), cv1(Conv(c1, c_, k.kernel[0], 1, std::nullopt, 1, 1, true)), cv2(Conv(c_, c2, k.kernel[1], 1, std::nullopt, g, 1, true))
{
    register_module("cv1", cv1);
    register_module("cv2", cv2);
    add = shortcut && (c1 == c2);
}
Bottleneck::Bottleneck(int64_t c1, int64_t c2, bool shortcut, int64_t g, MultiKernel k, float e) : c_(static_cast<int64_t>(c2 * e)),
                                                                                                   cv1(Conv(c1, c_, k.kernels[0], 1, std::nullopt, 1, 1, true)), cv2(Conv(c_, c2, k.kernels[1], 1, std::nullopt, g, 1, true))
{
    register_module("cv1", cv1);
    register_module("cv2", cv2);
    add = shortcut && (c1 == c2);
}

torch::Tensor Bottleneck::forward(torch::Tensor x)
{
    if (add)
    {
        return x + cv2->forward(cv1->forward(x));
    }
    else
    {
        return cv2->forward(cv1->forward(x));
    }
}

C2f::C2f(int64_t c1, int64_t c2, int64_t n, bool shortcut, int64_t g, float e) : c(static_cast<int64_t>(c2 * e)), cv1(Conv(c1, 2 * c, 1, 1, std::nullopt, 1, 1, true)),
                                                                                 cv2(Conv((2 + n) * c, c2, 1, 1, std::nullopt, 1, 1, true))
{
    m = register_module("m", torch::nn::ModuleList());
    register_module("cv1", cv1);
    register_module("cv2", cv2);

    std::vector<std::vector<int64_t>> kernel = {{3, 3}, {3, 3}};
    for (int i = 0; i < n; i++)
    {
        m->push_back(std::make_shared<Bottleneck>(c, c, shortcut, g, MultiKernel{kernel}, 1.0));
    }
}

torch::Tensor C2f::forward(torch::Tensor x)
{
    auto y = cv1->forward(x).chunk(2, 1);
    std::vector<torch::Tensor> outputs(y.begin(), y.end());
    for (const auto &module : *m)
    {
        outputs.push_back(module->as<Bottleneck>()->forward(outputs.back()));
    }
    return cv2->forward(torch::cat(outputs, 1));
}

torch::Tensor C2f::forward_split(torch::Tensor x)
{
    auto y_split = cv1->forward(x).split({c, c}, 1);
    std::vector<torch::Tensor> outputs = {y_split[0], y_split[1]};
    for (const auto &module : *m)
    {
        outputs.push_back(module->as<Bottleneck>()->forward(outputs.back()));
    }
    return cv2->forward(torch::cat(outputs, 1));
}
C3::C3(int64_t c1, int64_t c2, int64_t n, bool shortcut, int64_t g, float e)
    : c_(static_cast<int64_t>(c2 * e)),
      cv1(Conv(c1, c_, 1, 1, std::nullopt, 1, 1, true)),
      cv2(Conv(c1, c_, 1, 1, std::nullopt, 1, 1, true)),
      cv3(Conv(2 * c_, c2, 1, 1, std::nullopt, 1, 1, true)),
      m(register_module("m", torch::nn::Sequential()))
{
    register_module("cv1", cv1);
    register_module("cv2", cv2);
    register_module("cv3", cv3);
    for (int64_t i = 0; i < n; ++i)
    {
        m->push_back(Bottleneck(c_, c_, shortcut, g, MultiKernel{{{1, 1}, {3, 3}}}, 1.0));
    }
}

torch::Tensor C3::forward(torch::Tensor x)
{
    auto y1 = m->forward(cv1->forward(x));
    auto y2 = cv2->forward(x);
    return cv3->forward(torch::cat({y1, y2}, 1));
}

C3k::C3k(int64_t c1, int64_t c2, int64_t n, bool shortcut, int64_t g, float e, int64_t k)
    : C3(c1, c2, n, shortcut, g, e)
{
    m = register_module("m", torch::nn::Sequential());
    int64_t c_ = static_cast<int64_t>(c2 * e);
    std::vector<int64_t> kernel = {k, k};
    for (int64_t i = 0; i < n; ++i)
    {
        m->push_back(Bottleneck(c_, c_, shortcut, g, SingleKernel{kernel}, 1.0));
    }
}

C3k2::C3k2(int64_t c1, int64_t c2, int64_t n, bool c3k, float e, int64_t g, bool shortcut)
    : C2f(c1, c2, n, shortcut, g, e)
{
    m = register_module("m", torch::nn::ModuleList());
    for (int64_t i = 0; i < n; ++i)
    {
        if (c3k)
        {
            m->push_back(std::make_shared<C3k>(c, c, 2, shortcut, g, 0.5, 3));
        }
        else
        {
            m->push_back(std::make_shared<Bottleneck>(c, c, shortcut, g, SingleKernel{{3, 3}}, 0.5));
        }
    }
}

SPPF::SPPF(int64_t c1, int64_t c2, int64_t k) : c_(c1 / 2), cv1(Conv(c1, c_, 1, 1, std::nullopt, 1, 1, true)), cv2(Conv(c_ * 4, c2, 1, 1, std::nullopt, 1, 1, true))
{
    register_module("cv1", cv1);
    register_module("cv2", cv2);
    m = register_module("m", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(k).stride(1).padding(k / 2)));
}

torch::Tensor SPPF::forward(torch::Tensor x)
{
    std::vector<torch::Tensor> y = {cv1->forward(x)};
    for (int i = 0; i < 3; i++)
    {
        y.push_back(m->forward(y.back()));
    }
    return cv2->forward(torch::cat(y, 1));
}

Attention::Attention(int64_t dim_, int64_t num_heads_, float attn_ratio_) : num_heads(num_heads_), head_dim(dim_ / num_heads), key_dim(static_cast<int64_t>(head_dim * attn_ratio_)), scale(std::pow(key_dim, -0.5)),
                                                                            nh_kd(key_dim * num_heads), h(dim_ + nh_kd * 2), qkv(Conv(dim_, h, 1, 1, std::nullopt, 1, 1, false)), proj(Conv(dim_, dim_, 1, 1, std::nullopt, 1, 1, false)),
                                                                            pe(Conv(dim_, dim_, 3, 1, std::nullopt, dim_, 1, false))
{
    register_module("qkv", qkv);
    register_module("proj", proj);
    register_module("pe", pe);
}

torch::Tensor Attention::forward(torch::Tensor x)
{
    auto size = x.sizes();
    int64_t B = size[0], C = size[1], H = size[2], W = size[3];
    int64_t N = H * W;
    auto qkv_ = qkv(x);
    auto view = qkv_.view({B, num_heads, key_dim * 2 + head_dim, N}).split({key_dim, key_dim, head_dim}, 2);
    torch::Tensor q = view[0], k = view[1], v = view[2];
    auto attn = (torch::matmul(q.transpose(-2, -1), k))*scale;
    attn = attn.softmax(-1);
    x = torch::matmul(v, attn.transpose(-2, -1)).view({B, C, H, W}) + pe->forward(v.reshape({B, C, H, W}));
    x = proj(x);
    return x;
}

PSABlock::PSABlock(int64_t c, float attn_ratio, int64_t num_heads, bool shortcut) : add(shortcut), attn(Attention(c, num_heads, attn_ratio))
{
    register_module("attn", attn);
    ffn = register_module("ffn", torch::nn::Sequential());
    ffn->push_back(Conv(c, c * 2, 1, 1, std::nullopt, 1, 1, true));
    ffn->push_back(Conv(c * 2, c, 1, 1, std::nullopt, 1, 1, false));
}

torch::Tensor PSABlock::forward(torch::Tensor x)
{
    if (this->add)
    {
        x = x + attn(x);
    }
    else
    {
        x = attn(x);
    }
    if (this->add)
    {
        x = x + ffn->forward(x);
    }
    else
    {
        x = ffn->forward(x);
    }
    return x;
}

C2PSA::C2PSA(int64_t c1, int64_t c2, int64_t n, float e) : c(static_cast<int64_t>(c1 * e)), cv1(Conv(c1, 2 * c, 1, 1, std::nullopt, 1, 1, true)), cv2(Conv(c * 2, c1, 1, 1, std::nullopt, 1, 1, true))
{
    register_module("cv1", cv1);
    register_module("cv2", cv2);
    m = register_module("m", torch::nn::Sequential());
    for (int i = 0; i < n; i++)
    {
        m->push_back(PSABlock(c, 0.5, c / 64, true));
    }
}

torch::Tensor C2PSA::forward(torch::Tensor x)
{
    auto out = cv1(x).split({c, c}, 1);
    torch::Tensor a = out[0], b = out[1];
    b = m->forward(b);
    return cv2->forward(torch::cat({a, b}, 1));
}

DFL::DFL(int64_t c1_) : c1(c1_)
{
    conv = register_module(
        "conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(c1, 1, 1).bias(false)));
    for (auto &p : conv->parameters())
    {
        p.requires_grad_(false);
    }
    torch::Tensor x = torch::arange(c1, torch::kFloat);
    conv->weight.data().copy_(x.view({1, c1, 1, 1}));
}

torch::Tensor DFL::forward(torch::Tensor x)
{
    auto b = x.size(0);
    auto a = x.size(2);
    auto reshaped = x.view({b, 4, c1, a}).transpose(1, 2).softmax(1);
    auto out = conv->forward(reshaped);
    return out.view({b, 4, a});
}