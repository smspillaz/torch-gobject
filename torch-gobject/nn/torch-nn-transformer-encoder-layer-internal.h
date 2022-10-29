/*
 * torch-gobject/torch-nn-transformer-encoder-layer-internal.h
 *
 * Base class for the TransformerEncoderLayer.
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

#include <torch-gobject/nn/torch-nn-transformer-encoder-layer.h>

#include <torch/torch.h>

torch::nn::TransformerEncoderLayer & torch_nn_transformer_encoder_layer_to_real_transformer_encoder_layer (TorchNNTransformerEncoderLayer *mod);

TorchNNTransformerEncoderLayer * torch_nn_transformer_encoder_layer_new_from_real_transformer_encoder_layer (torch::nn::TransformerEncoderLayer const &allocator);
