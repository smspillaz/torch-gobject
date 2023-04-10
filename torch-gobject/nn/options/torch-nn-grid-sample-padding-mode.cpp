
/*
 * torch-gobject/nn/options/torch-nn-grid-sample-padding-mode.cpp
 *
 * GridSample padding mode implementation
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
#include <torch-gobject/nn/options/torch-nn-grid-sample-padding-mode-internal.h>
#include <torch-gobject/nn/options/torch-nn-grid-sample-padding-mode.h>

torch::nn::functional::GridSampleFuncOptions::padding_mode_t
torch_nn_grid_sample_padding_mode_to_real_grid_sample_padding_mode (TorchNNGridSamplePaddingMode mode)
{
  switch (mode)
    {
      case TORCH_NN_GRID_SAMPLE_PADDING_MODE_ZEROS:
        return torch::enumtype::kZeros();
      case TORCH_NN_GRID_SAMPLE_PADDING_MODE_BORDER:
        return torch::enumtype::kBorder();
      case TORCH_NN_GRID_SAMPLE_PADDING_MODE_REFLECTION:
        return torch::enumtype::kReflection();
      default:
        throw std::logic_error("Invalid grid sample padding mode");
    }
}

TorchNNGridSamplePaddingMode
torch_nn_grid_sample_mode_padding_new_from_real_grid_sample_padding_mode (torch::nn::functional::GridSampleFuncOptions::padding_mode_t const &mode)
{
  if (c10::get_if<torch::enumtype::kZeros> (&mode)) {
    return TORCH_NN_GRID_SAMPLE_PADDING_MODE_ZEROS;
  }

  if (c10::get_if<torch::enumtype::kBorder> (&mode)) {
    return TORCH_NN_GRID_SAMPLE_PADDING_MODE_BORDER;
  }

  if (c10::get_if<torch::enumtype::kReflection> (&mode)) {
    return TORCH_NN_GRID_SAMPLE_PADDING_MODE_REFLECTION;
  }

  throw std::logic_error ("Invalid grid sample padding mode");
}