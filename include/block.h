#ifndef BLOCK_H
#define BLOCK_H
#include <torch/torch.h>
#include <vector>
#include "conv.h"

class Bottleneck : public torch::nn::Module
{
public:
    Bottleneck(int64_t c1, int64_t c2, bool shortcut = true, int64_t g = 1, std::vector<int64_t> k = {3, 3}, float e = 0.5);
    Bottleneck(int64_t c1, int64_t c2, bool shortcut = true, int64_t g = 1, std::vector<std::vector<int64_t>> k = {{3, 3}, {3, 3}}, float e = 0.5);
    torch::Tensor forward(torch::Tensor x);

private:
    int64_t c_;
    torch::nn::ModuleHolder<Conv> cv1{nullptr};
    torch::nn::ModuleHolder<Conv> cv2{nullptr};
    bool add;
};

class C2f : public torch::nn::Module
{
public:
    int64_t c;
    C2f(int64_t c1, int64_t c2, int64_t n = 1, bool shortcut = false, int64_t g = 1, float e = 0.5);
    torch::Tensor forward(torch::Tensor x);
    torch::Tensor forward_split(torch::Tensor x);

private:
    torch::nn::ModuleHolder<Conv> cv1{nullptr};
    torch::nn::ModuleHolder<Conv> cv2{nullptr};
    torch::nn::ModuleList m{nullptr};
};

class C3 : public torch::nn::Module
{
public:
    C3(int64_t c1, int64_t c2, int64_t n = 1, bool shortcut = true, int64_t g = 1, float e = 0.5);
    torch::Tensor forward(torch::Tensor x);
    int64_t c_;
    torch::nn::ModuleHolder<Conv> cv1{nullptr};
    torch::nn::ModuleHolder<Conv> cv2{nullptr};
    torch::nn::ModuleHolder<Conv> cv3{nullptr};
    torch::nn::Sequential m{nullptr};
};

class C3k : public C3 {
public:
    C3k(int64_t c1, int64_t c2, int64_t n = 1, bool shortcut = true, int64_t g = 1, float e = 0.5, int64_t k = 3);
};


class C3k2 : public C2f
{
public:
    C3k2(int64_t c1, int64_t c2, int64_t n = 1, bool c3k = false, float e = 0.5, int64_t g = 1, bool shortcut = true);

private:
    torch::nn::ModuleList m{nullptr};
};

#endif