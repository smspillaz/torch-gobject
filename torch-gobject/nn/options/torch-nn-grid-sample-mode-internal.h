
/*
 * torch-gobject/nn/options/torch-nn-grid-sample-mode-internal.h
 *
 * GridSample mode internal functionality
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

#pragma once

#include <torch/nn/options/vision.h>
#include <torch-gobject/torch-util.h>
#include <torch-gobject/nn/options/torch-nn-grid-sample-mode.h>

torch::nn::functional::GridSampleFuncOptions::mode_t
torch_nn_grid_sample_mode_to_real_grid_sample_mode (TorchNNGridSampleMode mode);

TorchNNGridSampleMode
torch_nn_grid_sample_mode_from_real_grid_sample_mode (torch::nn::functional::GridSampleFuncOptions::mode_t const &mode);

namespace torch
{
  namespace gobject
  {
    template<>
    struct ConversionTrait<TorchNNGridSampleMode>
    {
      typedef torch::nn::functional::GridSampleFuncOptions::mode_t real_type;
      static constexpr auto from = torch_nn_grid_sample_mode_to_real_grid_sample_mode;
      static constexpr auto to = torch_nn_grid_sample_mode_from_real_grid_sample_mode;
    };

    template<>
    struct ReverseConversionTrait<torch::nn::functional::GridSampleFuncOptions::mode_t>
    {
      typedef TorchNNGridSampleMode gobject_type;
    };
  }
}