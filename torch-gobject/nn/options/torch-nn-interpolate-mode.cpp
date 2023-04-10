
/*
 * torch-gobject/nn/options/torch-nn-interpolate-mode.cpp
 *
 * Interpolate mode implementation
 *
 * Copyright (C) 2021 Sam Spilsbury.
 *
 * torch-gobject is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or
 * (at your option) any later version.
 *
 * torch-gobject is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with torch-gobject; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include <torch/enum.h>
#include <torch-gobject/nn/options/torch-nn-interpolate-mode-internal.h>
#include <torch-gobject/nn/options/torch-nn-interpolate-mode.h>

torch::nn::functional::InterpolateFuncOptions::mode_t
torch_nn_interpolate_mode_to_real_interpolate_mode (TorchNNInterpolateMode mode)
{
  switch (mode)
    {
      case TORCH_NN_INTERPOLATE_MODE_NEAREST:
        return torch::enumtype::kNearest();
      case TORCH_NN_INTERPOLATE_MODE_LINEAR:
        return torch::enumtype::kLinear();
      case TORCH_NN_INTERPOLATE_MODE_BILINEAR:
        return torch::enumtype::kBilinear();
      case TORCH_NN_INTERPOLATE_MODE_BICUBIC:
        return torch::enumtype::kBicubic();
      case TORCH_NN_INTERPOLATE_MODE_TRILINEAR:
        return torch::enumtype::kTrilinear();
      case TORCH_NN_INTERPOLATE_MODE_AREA:
        return torch::enumtype::kArea();
      case TORCH_NN_INTERPOLATE_MODE_NEAREST_EXACT:
        return torch::enumtype::kNearestExact();
      default:
        throw std::logic_error("Invalid interpolate mode");
    }
}

TorchNNInterpolateMode
torch_nn_interpolate_mode_from_real_interpolate_mode (torch::nn::functional::InterpolateFuncOptions::mode_t const &mode)
{
  if (c10::get_if<torch::enumtype::kNearest> (&mode)) {
    return TORCH_NN_INTERPOLATE_MODE_NEAREST;
  }

  if (c10::get_if<torch::enumtype::kLinear> (&mode)) {
    return TORCH_NN_INTERPOLATE_MODE_LINEAR;
  }

  if (c10::get_if<torch::enumtype::kBilinear> (&mode)) {
    return TORCH_NN_INTERPOLATE_MODE_BILINEAR;
  }

  if (c10::get_if<torch::enumtype::kBicubic> (&mode)) {
    return TORCH_NN_INTERPOLATE_MODE_BICUBIC;
  }

  if (c10::get_if<torch::enumtype::kTrilinear> (&mode)) {
    return TORCH_NN_INTERPOLATE_MODE_TRILINEAR;
  }

  if (c10::get_if<torch::enumtype::kArea> (&mode)) {
    return TORCH_NN_INTERPOLATE_MODE_AREA;
  }

  if (c10::get_if<torch::enumtype::kNearestExact> (&mode)) {
    return TORCH_NN_INTERPOLATE_MODE_NEAREST_EXACT;
  }

  throw std::logic_error ("Invalid interpolate mode");
}