
/*
 * torch-gobject/nn/options/torch-nn-upsample-mode.cpp
 *
 * Upsample mode implementation
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
#include <torch-gobject/nn/options/torch-nn-upsample-mode-internal.h>
#include <torch-gobject/nn/options/torch-nn-upsample-mode.h>

c10::variant<
  torch::enumtype::kNearest,
  torch::enumtype::kLinear,
  torch::enumtype::kBilinear,
  torch::enumtype::kBicubic,
  torch::enumtype::kTrilinear
>
torch_nn_upsample_mode_to_real_upsample_mode (TorchNNUpsampleMode mode)
{
  switch (mode)
    {
      case TORCH_NN_UPSAMPLE_MODE_NEAREST:
        return torch::enumtype::kNearest();
      case TORCH_NN_UPSAMPLE_MODE_LINEAR:
        return torch::enumtype::kLinear();
      case TORCH_NN_UPSAMPLE_MODE_BILINEAR:
        return torch::enumtype::kBilinear();
      case TORCH_NN_UPSAMPLE_MODE_BICUBIC:
        return torch::enumtype::kBicubic();
      case TORCH_NN_UPSAMPLE_MODE_TRILINEAR:
        return torch::enumtype::kTrilinear();
      default:
        throw std::logic_error("Invalid upsample mode");
    }
}