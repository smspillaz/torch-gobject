/*
 * torch-gobject/torch-nn-transformer-decoder-layer-internal.h
 *
 * Base class for the TransformerDecoderLayer.
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

#include <torch-gobject/nn/torch-nn-transformer-decoder-layer.h>
#include <torch-gobject/torch-util.h>
#include <torch/torch.h>

torch::nn::TransformerDecoderLayer & torch_nn_transformer_decoder_layer_to_real_transformer_decoder_layer (TorchNNTransformerDecoderLayer *mod);

TorchNNTransformerDecoderLayer * torch_nn_transformer_decoder_layer_new_from_real_transformer_decoder_layer (torch::nn::TransformerDecoderLayer const &allocator);

namespace torch
{
  namespace gobject
  {
    template<>
    struct ConversionTrait<TorchNNTransformerDecoderLayer *>
    {
      typedef torch::nn::TransformerDecoderLayer & real_type;
      static constexpr auto from = torch_nn_transformer_decoder_layer_to_real_transformer_decoder_layer;
      static constexpr auto to = torch_nn_transformer_decoder_layer_new_from_real_transformer_decoder_layer;
    };

    template<>
    struct ReverseConversionTrait<torch::nn::TransformerDecoderLayer>
    {
      typedef TorchNNTransformerDecoderLayer * gobject_type;
    };
  }
}