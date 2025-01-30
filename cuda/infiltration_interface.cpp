/*** 
 * @Author: JJiangcv jinghua.jiang21@gmail.com
 * @Date: 2021-09-02 15:53:53
 * @LastEditors: JJiangcv jinghua.jiang21@gmail.com
 * @LastEditTime: 2025-01-30 11:39:31
 * @FilePath: /cvjj7/hipims/cuda/infiltration_interface.cpp
 * @Description: 
 * @
 * @Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
 */
#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
void infiltrationCalculation_cuda(at::Tensor wetMask, at::Tensor h_update,
                                  at::Tensor landuse, at::Tensor h,
                                  at::Tensor hydraulic_conductivity,
                                  at::Tensor capillary_head,
                                  at::Tensor water_content_diff,
                                  at::Tensor cumulative_depth,
                                  at::Tensor sewer_sink,
                                  at::Tensor dt);
// C++ interface
#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor. ")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous. ")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

void infiltrationCalculation(at::Tensor wetMask, at::Tensor h_update,
                             at::Tensor landuse, at::Tensor h,
                             at::Tensor hydraulic_conductivity,
                             at::Tensor capillary_head,
                             at::Tensor water_content_diff,
                             at::Tensor cumulative_depth,
                             at::Tensor sewer_sink,
                             at::Tensor dt)

{
  CHECK_INPUT(wetMask);
  CHECK_INPUT(landuse);
  CHECK_INPUT(h);
  CHECK_INPUT(h_update);
  CHECK_INPUT(dt);
  CHECK_INPUT(hydraulic_conductivity);
  CHECK_INPUT(capillary_head);
  CHECK_INPUT(water_content_diff);
  CHECK_INPUT(cumulative_depth);
  CHECK_INPUT(sewer_sink);

  infiltrationCalculation_cuda(wetMask, h_update, landuse, h,
                               hydraulic_conductivity, capillary_head,
                               water_content_diff, cumulative_depth, sewer_sink, dt);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("addinfiltration", &infiltrationCalculation,
        "Infiltration Updating, CUDA version");
}