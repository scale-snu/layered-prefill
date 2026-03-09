/*
 * Adapted from https://github.com/vllm-project/vllm
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include <torch/extension.h>

void rms_norm(
  torch::Tensor& out,
  torch::Tensor& input,
  torch::Tensor& weight,
  float epsilon);

void add_rms_norm(
  torch::Tensor& out,
  torch::Tensor& residual,
  torch::Tensor& input,
  torch::Tensor& weight,
  float epsilon);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "rms_norm",
    &rms_norm,
    "Apply Root Mean Square (RMS) Normalization to the input tensor.");
  m.def(
    "add_rms_norm",
    &add_rms_norm,
    "Apply Root Mean Square (RMS) Normalization to the input tensor.");
}