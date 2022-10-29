/*
 * torch-gobject/nn/options/torch-nn-conv-padding-type.c
 *
 * Convolution operator padding type
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

#include <torch-gobject/nn/options/torch-nn-conv-padding-mode.h>
#include <torch-gobject/nn/options/torch-nn-conv-padding-mode-internal.h>

torch::nn::detail::conv_padding_mode_t
torch_nn_conv_padding_mode_to_real_conv_padding_mode (TorchNNConvPaddingMode mode)
{
    switch (mode)
      {
        case TORCH_NN_CONV_PADDING_MODE_ZEROS:
          return torch::enumtype::kZeros();
        case TORCH_NN_CONV_PADDING_MODE_REFLECT:
          return torch::enumtype::kReflect();
        case TORCH_NN_CONV_PADDING_MODE_REPLICATE:
          return torch::enumtype::kReplicate();
        case TORCH_NN_CONV_PADDING_MODE_CIRCULAR:
          return torch::enumtype::kCircular();
        default:
          throw std::logic_error ("Invalid conv padding type");
      }
}