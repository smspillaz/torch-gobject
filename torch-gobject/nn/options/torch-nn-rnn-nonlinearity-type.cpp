/*
 * torch-gobject/nn/options/torch-nn-rnn-nonlinearity-type.cpp
 *
 * RNN Nonlinearity types.
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

#include <torch-gobject/nn/options/torch-nn-rnn-nonlinearity-type.h>
#include <torch-gobject/nn/options/torch-nn-rnn-nonlinearity-type-internal.h>

torch::nn::RNNCellOptions::nonlinearity_t
torch_nn_rnn_nonlinearity_type_to_real_rnn_nonlinearity_type (TorchNNRNNNonlinearityType mode)
{
  switch (mode)
    {
      case TORCH_NN_RNN_NONLINEARITY_TYPE_RELU:
        return torch::enumtype::kReLU();
      case TORCH_NN_RNN_NONLINEARITY_TYPE_TANH:
        return torch::enumtype::kTanh();
      default:
        throw std::logic_error("Invalid RNN nonlinearity type");
    }
}

TorchNNRNNNonlinearityType
torch_nn_rnn_nonlinearity_type_from_real_rnn_nonlinearity_type (torch::nn::RNNCellOptions::nonlinearity_t const &mode)
{
  if (c10::get_if<torch::enumtype::kReLU> (&mode)) {
    return TORCH_NN_RNN_NONLINEARITY_TYPE_RELU;
  }

  if (c10::get_if<torch::enumtype::kTanh> (&mode)) {
    return TORCH_NN_RNN_NONLINEARITY_TYPE_TANH;
  }

  throw std::logic_error ("Invalid RNN nonlinearity type");
}