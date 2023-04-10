/*
 * torch-gobject/nn/options/torch-nn-conv-padding-options-internal.h
 *
 * Convolution operator padding options, internal conversion operators.
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

#include <tuple>
#include <vector>

#include <torch/nn/options/conv.h>

#include <torch-gobject/torch-util.h>
#include <torch-gobject/nn/options/torch-nn-conv-padding-options.h>

namespace {
  template <int D>
  typename torch::nn::ConvOptions <D>::padding_t torch_nn_conv_padding_options_to_real_padding_t (TorchNNConvPaddingOptions *opts)
  {
    TorchNNConvPaddingType padding_type = torch_nn_conv_padding_options_get_padding_type (opts);

    switch (padding_type)
      {
        case TORCH_NN_CONV_PADDING_TYPE_SPECIFIED:
          {
            GArray *config = torch_nn_conv_padding_options_get_padding_config (opts);
            g_assert (config != nullptr);

            return torch_array_ref_from_garray <int64_t> (config);
          }
          break;
        case TORCH_NN_CONV_PADDING_TYPE_VALID:
          return torch::enumtype::kValid();
        case TORCH_NN_CONV_PADDING_TYPE_SAME:
          return torch::enumtype::kSame();
        default:
          throw std::logic_error("Invalid padding type specified");
      }
  }

  template <int D>
  TorchNNConvPaddingOptions * torch_nn_conv_padding_options_new_from_real_conv_padding_options (typename torch::nn::ConvOptions <D>::padding_t const &padding_opts)
  {
    if (c10::get_if <torch::ExpandingArray <D>> (&padding_opts))
      {
        auto array = c10::get <torch::ExpandingArray <D>> (padding_opts);
        return torch_nn_conv_padding_options_new (
          TORCH_NN_CONV_PADDING_TYPE_SPECIFIED,
          array->data(),
          array.size()
        );
      }

    if (c10::get_if <torch::enumtype::kValid> (&padding_opts))
      {
        return torch_nn_conv_padding_options_new (
          TORCH_NN_CONV_PADDING_TYPE_VALID,
          nullptr,
          0
        );
      }

    if (c10::get_if <torch::enumtype::kSame> (&padding_opts))
      {
        return torch_nn_conv_padding_options_new (
          TORCH_NN_CONV_PADDING_TYPE_SAME,
          nullptr,
          0
        );
      }
  }
}
