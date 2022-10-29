/*
 * torch-gobject/nn/options/torch-nn-upsample-mode.h
 *
 * Upsample mode.
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

/**
 * TorchNNUpsampleMode
 * @TORCH_NN_UPSAMPLE_MODE_NEAREST: Do nearest-neighbour upsampling
 * @TORCH_NN_UPSAMPLE_MODE_LINEAR: Do linear interpolation when upsampling
 * @TORCH_NN_UPSAMPLE_MODE_BILINEAR: Do bilinear interpolation when upsampling
 * @TORCH_NN_UPSAMPLE_MODE_BICUBIC: Do bicubic interpolation when upsampling
 * @TORCH_NN_UPSAMPLE_MODE_TRILINEAR: Do trilinear interpolation when u[sampling
 */
typedef enum {
  TORCH_NN_UPSAMPLE_MODE_NEAREST,
  TORCH_NN_UPSAMPLE_MODE_LINEAR,
  TORCH_NN_UPSAMPLE_MODE_BILINEAR,
  TORCH_NN_UPSAMPLE_MODE_BICUBIC,
  TORCH_NN_UPSAMPLE_MODE_TRILINEAR
} TorchNNUpsampleMode;