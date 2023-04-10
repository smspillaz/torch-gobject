
/*
 * torch-gobject/nn/options/torch-nn-conv-padding-mode-internal.h
 *
 * Conv padding mode internal functionality
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

#include <torch/nn/options/conv.h>

torch::nn::detail::conv_padding_mode_t
torch_nn_conv_padding_mode_to_real_conv_padding_mode (TorchNNConvPaddingMode mode);

TorchNNConvPaddingMode
torch_nn_conv_padding_mode_from_real_conv_padding_mode (torch::nn::detail::conv_padding_mode_t const &mode);
