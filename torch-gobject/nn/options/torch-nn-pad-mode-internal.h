
/*
 * torch-gobject/nn/options/torch-nn-pad-mode-internal.h
 *
 * Pad mode for nn_functional_pad, internal functionality
 *
 * Copyright (C) 2022 Sam Spilsbury.
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

#include <torch/nn/options/padding.h>
#include <torch-gobject/torch-util.h>
#include <torch-gobject/nn/options/torch-nn-pad-mode.h>

torch::nn::functional::PadFuncOptions::mode_t
torch_nn_pad_mode_to_real_pad_mode (TorchNNPadMode mode);

TorchNNPadMode
torch_nn_pad_mode_from_real_pad_mode (torch::nn::functional::PadFuncOptions::mode_t const &mode);

namespace torch
{
  namespace gobject
  {
    template<>
    struct ConversionTrait<TorchNNPadMode>
    {
      typedef torch::nn::functional::PadFuncOptions::mode_t real_type;
      static constexpr auto from = torch_nn_pad_mode_to_real_pad_mode;
      static constexpr auto to = torch_nn_pad_mode_from_real_pad_mode;
    };

    template<>
    struct ReverseConversionTrait<torch::nn::functional::PadFuncOptions::mode_t>
    {
      typedef TorchNNPadMode gobject_type;
    };
  }
}