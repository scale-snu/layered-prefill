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

void silu_and_mul(
  torch::Tensor& out,
  torch::Tensor& input);

void gelu_new(
  torch::Tensor& out,
  torch::Tensor& input);

void gelu_fast(
  torch::Tensor& out,
  torch::Tensor& input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "silu_and_mul",
    &silu_and_mul,
    "Activation function used in SwiGLU.");
  m.def(
    "gelu_new",
    &gelu_new,
    "GELU implementation used in GPT-2.");
  m.def(
    "gelu_fast",
    &gelu_fast,
    "Approximate GELU implementation.");
}