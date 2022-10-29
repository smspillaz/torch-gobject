/*
 * torch-gobject/nn/options/torch-nn-transformer-decoder-layer.h
 *
 * Base class for the TransformerDecoderLayer
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

#include <glib-object.h>
#include <torch-gobject/nn/torch-nn-module-base.h>

G_BEGIN_DECLS

#define TORCH_TYPE_NN_TRANSFORMER_DECODER_LAYER torch_nn_transformer_decoder_layer_get_type ()
G_DECLARE_FINAL_TYPE (TorchNNTransformerDecoderLayer, torch_nn_transformer_decoder_layer, TORCH, NN_TRANSFORMER_DECODER_LAYER, TorchNNModuleBase)

TorchNNTransformerDecoderLayer * torch_nn_transformer_decoder_layer_new (void);

G_END_DECLS