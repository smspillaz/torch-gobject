
/*
 * torch-gobject/nn/options/torch-nn-grid-sample-mode.cpp
 *
 * GridSample mode implementation
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
#include <torch-gobject/nn/options/torch-nn-grid-sample-mode-internal.h>
#include <torch-gobject/nn/options/torch-nn-grid-sample-mode.h>

torch::nn::functional::GridSampleFuncOptions::mode_t
torch_nn_grid_sample_mode_to_real_grid_sample_mode (TorchNNGridSampleMode mode)
{
  switch (mode)
    {
      case TORCH_NN_GRID_SAMPLE_MODE_BILINEAR:
        return torch::enumtype::kBilinear();
      case TORCH_NN_GRID_SAMPLE_MODE_NEAREST:
        return torch::enumtype::kNearest();
      default:
        throw std::logic_error("Invalid grid sample mode");
    }
}

TorchNNGridSampleMode
torch_nn_grid_sample_mode_new_from_real_grid_sample_mode (torch::nn::functional::GridSampleFuncOptions::mode_t mode)
{
  if (c10::get_if<torch::enumtype::kBilinear> (&mode)) {
    return TORCH_NN_GRID_SAMPLE_MODE_BILINEAR;
  }

  if (c10::get_if<torch::enumtype::kNearest> (&mode)) {
    return TORCH_NN_GRID_SAMPLE_MODE_NEAREST;
  }

  throw std::logic_error ("Invalid grid sample mode");
}