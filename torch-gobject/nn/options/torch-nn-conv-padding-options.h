/*
 * torch-gobject/nn/options/torch-nn-conv-padding-options.h
 *
 * Convolution operator padding options
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

#include <glib-object.h>
#include <torch-gobject/nn/options/torch-nn-conv-padding-type.h>

G_BEGIN_DECLS

typedef struct _TorchNNConvPaddingOptions TorchNNConvPaddingOptions;

GType torch_nn_conv_padding_options_get_type (void);
#define TORCH_TYPE_NN_CONV_PADDING_OPTIONS (torch_nn_conv_padding_options_get_type())

TorchNNConvPaddingOptions * torch_nn_conv_padding_options_new (TorchNNConvPaddingType padding_type, int64_t *padding_config, size_t padding_config_length);

TorchNNConvPaddingOptions * torch_nn_conv_padding_options_copy (TorchNNConvPaddingOptions *opts);

TorchNNConvPaddingType torch_nn_conv_padding_options_get_padding_type (TorchNNConvPaddingOptions *opts);

GArray * torch_nn_conv_padding_options_get_padding_config (TorchNNConvPaddingOptions *opts);

void torch_nn_conv_padding_options_free (TorchNNConvPaddingOptions *opts);

G_END_DECLS