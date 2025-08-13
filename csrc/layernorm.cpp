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