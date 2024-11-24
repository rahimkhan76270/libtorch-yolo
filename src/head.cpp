#include "head.h"
#include <vector>
#include <tuple>

bool areIntArrayRefsEqual(const torch::IntArrayRef &a, const torch::IntArrayRef &b)
{
    if (a.size() != b.size())
    {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (a[i] != b[i])
        {
            return false;
        }
    }
    return true;
}

torch::Tensor dist2bbox(torch::Tensor distance,
                        torch::Tensor anchor_points,
                        bool xywh = true,
                        int64_t dim = -1)
{

    auto chunks = distance.chunk(2, dim);
    auto lt = chunks[0];
    auto rb = chunks[1];
    auto x1y1 = anchor_points - lt;
    auto x2y2 = anchor_points + rb;

    if (xywh)
    {
        auto c_xy = (x1y1 + x2y2) / 2;
        auto wh = x2y2 - x1y1;
        return torch::cat({c_xy, wh}, dim);
    }
    return torch::cat({x1y1, x2y2}, dim);
}

std::pair<torch::Tensor, torch::Tensor> make_anchors(
    const std::vector<torch::Tensor> &feats,
    const std::vector<int> &strides,
    float grid_cell_offset = 0.5)
{
    std::vector<torch::Tensor> anchor_points, stride_tensors;
    TORCH_CHECK(!feats.empty(), "Input features cannot be empty.");
    auto dtype = feats[0].dtype();
    auto device = feats[0].device();
    for (size_t i = 0; i < strides.size(); ++i)
    {
        int h = feats[i].size(2);
        int w = feats[i].size(3);
        auto sx = torch::arange(0, w, torch::TensorOptions().dtype(dtype).device(device)) + grid_cell_offset;
        auto sy = torch::arange(0, h, torch::TensorOptions().dtype(dtype).device(device)) + grid_cell_offset;
        auto grid = torch::meshgrid({sy, sx}, /*indexing=*/"ij");
        auto sx_grid = grid[1];
        auto sy_grid =grid[0];

        auto points = torch::stack({sx_grid, sy_grid}, -1).view({-1, 2});
        anchor_points.push_back(points);

        // Create stride tensor with shape (h * w, 1)
        auto stride_tensor = torch::full({h * w, 1}, strides[i], torch::TensorOptions().dtype(dtype).device(device));
        stride_tensors.push_back(stride_tensor);
    }
    auto anchors = torch::cat(anchor_points, 0);
    auto strides_out = torch::cat(stride_tensors, 0);

    return {anchors, strides_out};
}


Detect::Detect(int64_t nc_, std::vector<int64_t> ch) : nc(nc_), nl(ch.size()), no(nc + reg_max * 4), stride(torch::zeros({nl})), dfl(DFL(reg_max))
{
    this->c2 = std::max(static_cast<int64_t>(16), std::max(ch[0] / 4, reg_max * 4));
    this->c3 = std::max(ch[0], std::min(nc, static_cast<int64_t>(100)));
    register_module("dfl", dfl);
    cv2 = register_module("cv2", torch::nn::ModuleList());
    cv3 = register_module("cv3", torch::nn::ModuleList());
    cv4 = register_module("cv4", torch::nn::ModuleList());
    identity = register_module("identity", torch::nn::Identity());
    for (auto x : ch)
    {
        torch::nn::Sequential sq;
        sq->push_back(Conv(x, c2, 3));
        sq->push_back(Conv(c2, c2, 3));
        sq->push_back(torch::nn::Conv2d(c2, reg_max, 1));
        cv2->push_back(sq);
    }

    for (auto x : ch)
    {
        if (legacy)
        {
            torch::nn::Sequential sq;
            sq->push_back(Conv(x, c3, 3));
            sq->push_back(Conv(c3, c3, 3));
            sq->push_back(torch::nn::Conv2d(c3, nc, 1));
            cv3->push_back(sq);
        }
        else
        {
            torch::nn::Sequential sq;
            torch::nn::Sequential sq1;
            torch::nn::Sequential sq2;
            sq1->push_back(DWConv(x, x, 3));
            sq1->push_back(Conv(x, c3, 1));
            sq2->push_back(DWConv(c3, c3, 3));
            sq2->push_back(Conv(c3, c3, 1));
            sq->push_back(sq1);
            sq->push_back(sq2);
            sq->push_back(torch::nn::Conv2d(c3, nc, 1));
        }
    }
    if (end2end)
    {
        for (const auto &module : *cv2)
        {
            one2one_cv2->push_back(register_module("one2one_cv2", module->clone()));
        }
        for (const auto &module : *cv3)
        {
            one2one_cv3->push_back(register_module("one2one_cv3", module->clone()));
        }
    }
}

torch::Tensor Detect::forward(torch::Tensor x)
{
    if (end2end)
    {
        return forward_end2end(x);
    }
    for (int i = 0; i < nl; i++)
    {
        x[i] = torch::cat({cv2[i]->as<torch::nn::Sequential>()->forward(x[i]), cv3[i]->as<torch::nn::Sequential>()->forward(x[i])}, 1);
    }
    if (training)
    {
        return x;
    }
    auto y = _inference(x);
    if (export_)
    {
        return y;
    }
    else
    {
        return torch::cat({y, x}, 1);
    }
}

torch::Tensor Detect::postprocess(torch::Tensor preds, int max_det, int nc)
{
    auto shape = preds.sizes();
    int64_t batch_size = shape[0];
    int64_t num_anchors = shape[1];
    auto splits = preds.split_with_sizes({4, nc}, -1);
    auto boxes = splits[0];
    auto scores = splits[1];
    auto scores_max = std::get<0>(scores.amax(-1).topk(std::min(max_det, (int)num_anchors), -1));
    auto index = std::get<1>(scores.amax(-1).topk(std::min(max_det, (int)num_anchors), -1))
                     .unsqueeze(-1);
    auto boxes_selected = boxes.gather(1, index.repeat({1, 1, 4}));
    auto scores_selected = scores.gather(1, index.repeat({1, 1, nc}));
    auto scores_flattened = scores_selected.flatten(1);
    auto topk_scores = std::get<0>(scores_flattened.topk(std::min(max_det, (int)(num_anchors * nc)), -1));
    auto topk_indices = std::get<1>(scores_flattened.topk(std::min(max_det, (int)(num_anchors * nc)), -1));
    auto batch_indices = torch::arange(batch_size).view({-1, 1}).to(torch::kLong);
    auto final_boxes = boxes_selected.index({batch_indices, topk_indices / nc});
    auto final_scores = topk_scores.unsqueeze(-1);
    auto final_classes = (topk_indices % nc).unsqueeze(-1).to(torch::kFloat);
    return torch::cat({final_boxes, final_scores, final_classes}, -1);
}

torch::Tensor Detect::decode_bboxes(torch::Tensor distance,
                                    torch::Tensor anchor_points,
                                    bool xywh = true)
{
    return dist2bbox(distance, anchor_points, xywh && !end2end, 1);
}

torch::Tensor Detect::_inference(torch::Tensor x)
{
    auto shape_ = x[0].sizes();
    std::vector<torch::Tensor> list_tensors;
    for (int i = 0; i < x.size(0); i++)
    {
        list_tensors.push_back(x[i].view({shape_[0], no, -1}));
    }
    auto x_cat = torch::cat(list_tensors, 2);

    if (format != "imx" && (dynamic || areIntArrayRefsEqual(shape_, shape)))
    {
    }
}