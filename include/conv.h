#ifndef MODULES_H
#define MODULES_H
#include <torch/torch.h>
#include <optional>
#include <vector>

std::vector<int64_t> autopad(std::vector<int64_t> k, std::optional<std::vector<int64_t>> p = std::nullopt, int64_t d = 1);
int64_t autopad(int64_t k, std::optional<int64_t> p, int64_t d);

class Conv : public torch::nn::Module
{
public:
    Conv(int64_t c1, int64_t c2, int64_t k = 1, int64_t s = 1, std::optional<int64_t> p = std::nullopt, int64_t g = 1, int64_t d = 1, bool act = true);
    Conv(int64_t c1, int64_t c2, std::vector<int64_t> k = {1,1}, int64_t s = 1, std::optional<std::vector<int64_t>> p = std::nullopt, int64_t g = 1, int64_t d = 1, bool act = true);
    torch::Tensor forward(torch::Tensor x);
    torch::Tensor forward_fuse(torch::Tensor x);

private:
    torch::nn::SiLU default_act{nullptr};
    torch::nn::Conv2d conv{nullptr};
    torch::nn::BatchNorm2d bn{nullptr};
    torch::nn::Identity identity{nullptr};
    bool activation;
};

class Conv2 : public torch::nn::Module
{
public:
    Conv2(int64_t c1, int64_t c2, int64_t k = 1, int64_t s = 1, std::optional<int64_t> p = std::nullopt, int64_t g = 1, int64_t d = 1, bool act = true);
    torch::Tensor forward(torch::Tensor x);
    torch::Tensor forward_fuse(torch::Tensor x);
    void fuse_convs();

private:
    torch::nn::SiLU default_act{nullptr};
    torch::nn::Conv2d conv{nullptr};
    torch::nn::Conv2d cv2{nullptr};
    torch::nn::BatchNorm2d bn{nullptr};
    torch::nn::Identity identity{nullptr};
    bool activation;
};

class Concat : public torch::nn::Module {
public:
    Concat(int64_t dimension = 1);
    torch::Tensor forward(const std::vector<torch::Tensor>& inputs);

private:
    int64_t dimension_;
};

#endif