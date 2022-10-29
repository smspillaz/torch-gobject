/*
 * torch-gobject/nn/options/torch-nn-pad-mode.h
 *
 * Padding mode for nn_functional_pad
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

/**
 * TorchNNPadMode
 * @TORCH_NN_PAD_MODE_CONSTANT: Padding type was specified by the user
 * @TORCH_NN_PAD_MODE_REFLECT: Reflect along the edges.
 * @TORCH_NN_PAD_MODE_REPLICATE: Repeat the input from the first element of the padding axis.
 * @TORCH_NN_PAD_MODE_CIRCULAR: Circular padding.
 */
typedef enum {
  TORCH_NN_PAD_MODE_CONSTANT,
  TORCH_NN_PAD_MODE_REFLECT,
  TORCH_NN_PAD_MODE_REPLICATE,
  TORCH_NN_PAD_MODE_CIRCULAR
} TorchNNPadMode;