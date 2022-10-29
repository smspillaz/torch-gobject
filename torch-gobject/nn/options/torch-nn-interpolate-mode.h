/*
 * torch-gobject/nn/options/torch-nn-interpolate-mode.h
 *
 * Interpolate mode.
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
 * TorchNNInterpolateMode
 * @TORCH_NN_INTERPOLATE_MODE_NEAREST: Do nearest-neighbour sampling
 * @TORCH_NN_INTERPOLATE_MODE_LINEAR: Do linear interpolatation
 * @TORCH_NN_INTERPOLATE_MODE_BILINEAR: Do linear interpolatation
 * @TORCH_NN_INTERPOLATE_MODE_BICUBIC: Do bicubic interpolation
 * @TORCH_NN_INTERPOLATE_MODE_TRILINEAR: Do trilinear interpolation
 * @TORCH_NN_INTERPOLATE_MODE_AREA: Do area interpolation
 * @TORCH_NN_INTERPOLATE_MODE_NEAREST_EXACT: Do nearest-exact interpolation
 */
typedef enum {
  TORCH_NN_INTERPOLATE_MODE_NEAREST,
  TORCH_NN_INTERPOLATE_MODE_LINEAR,
  TORCH_NN_INTERPOLATE_MODE_BILINEAR,
  TORCH_NN_INTERPOLATE_MODE_BICUBIC,
  TORCH_NN_INTERPOLATE_MODE_TRILINEAR,
  TORCH_NN_INTERPOLATE_MODE_AREA,
  TORCH_NN_INTERPOLATE_MODE_NEAREST_EXACT
} TorchNNInterpolateMode;