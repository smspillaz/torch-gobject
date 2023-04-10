
/*
 * torch-gobject/nn/options/torch-nn-rnn-nonlinearity-type.h
 *
 * RNN nonlinearity type internal functionality
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

#include <torch/nn/options/rnn.h>
#include <torch-gobject/torch-util.h>
#include <torch-gobject/nn/options/torch-nn-rnn-nonlinearity-type.h>

torch::nn::RNNCellOptions::nonlinearity_t
torch_nn_rnn_nonlinearity_type_to_real_rnn_nonlinearity_type (TorchNNRNNNonlinearityType mode);

TorchNNRNNNonlinearityType
torch_nn_rnn_nonlinearity_type_from_real_rnn_nonlinearity_type (torch::nn::RNNCellOptions::nonlinearity_t const &mode);

namespace torch
{
  namespace gobject
  {
    template<>
    struct ConversionTrait<TorchNNRNNNonlinearityType>
    {
      typedef torch::nn::RNNCellOptions::nonlinearity_t real_type;
      static constexpr auto from = torch_nn_rnn_nonlinearity_type_to_real_rnn_nonlinearity_type;
      static constexpr auto to = torch_nn_rnn_nonlinearity_type_from_real_rnn_nonlinearity_type;
    };

    template<>
    struct ReverseConversionTrait<torch::nn::RNNCellOptions::nonlinearity_t>
    {
      typedef TorchNNRNNNonlinearityType gobject_type;
    };
  }
}