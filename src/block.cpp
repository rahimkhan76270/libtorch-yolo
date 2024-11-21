#include "block.h"
#include "conv.h"
#include <vector>
#include <stdexcept>

Bottleneck::Bottleneck(int64_t c1, int64_t c2, bool shortcut, int64_t g, std::vector<int64_t> k, float e)
{
    this->c_ = (int64_t)c2 * e;
    cv1(Conv(c1, c_, k[0], 1, std::nullopt, 1, 1, true));
    cv2(Conv(c_, c2, k[1], 1, std::nullopt, g, 1, true));
    register_module("cv1", cv1);
    register_module("cv2", cv2);
    add = shortcut && (c1 == c2);
}
Bottleneck::Bottleneck(int64_t c1, int64_t c2, bool shortcut, int64_t g, std::vector<std::vector<int64_t>> k, float e) : c_((int64_t)(c2 * e)),
                                                                                                                         cv1(Conv(c1, c_, k[0], 1, std::nullopt, 1, 1, true)), cv2(Conv(c_, c2, k[1], 1, std::nullopt, g, 1, true))
{
    register_module("cv1", cv1);
    register_module("cv2", cv2);
    add = shortcut && (c1 == c2);
}

torch::Tensor Bottleneck::forward(torch::Tensor x)
{
    return (add == true) ? (x + cv2->forward(cv1->forward(x))) : cv2->forward(cv1->forward(x));
}

C2f::C2f(int64_t c1, int64_t c2, int64_t n, bool shortcut, int64_t g, float e) : c((int64_t)(c2 * e)), cv1(Conv(c1, 2 * c, 1, 1, std::nullopt, 1, 1, true)),
                                                                                 cv2(Conv((2 + n) * c, c2, 1, 1, std::nullopt, 1, 1, true)), m(register_module("m", torch::nn::ModuleList()))
{
    register_module("cv1", cv1);
    register_module("cv2", cv2);

    std::vector<std::vector<int64_t>> kernel = {{3, 3}, {3, 3}};
    for (int i = 0; i < n; i++)
    {
        m->push_back(std::make_shared<Bottleneck>(c, c, shortcut, g, kernel, 1.0));
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
        m->push_back(Bottleneck(c_, c_, shortcut, g, {{1, 1}, {3, 3}}, 1.0));
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
    std::vector<std::vector<int64_t>> kernel={{k,k}};
    for (int64_t i = 0; i < n; ++i)
    {
        m->push_back(Bottleneck(c_, c_, shortcut, g, kernel, 1.0));
    }
}

C3k2::C3k2(int64_t c1, int64_t c2, int64_t n, bool c3k, float e, int64_t g, bool shortcut)
    : C2f(c1, c2, n, shortcut, g, e), m(register_module("m", torch::nn::ModuleList()))
{
    for (int64_t i = 0; i < n; ++i)
    {
        if (c3k)
        {
            m->push_back(std::make_shared<C3k>(c, c, 2, shortcut, g));
        }
        else
        {
            m->push_back(std::make_shared<Bottleneck>(c, c, shortcut, g));
        }
    }
}