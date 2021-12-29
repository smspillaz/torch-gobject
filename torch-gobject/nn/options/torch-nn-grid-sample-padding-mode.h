/*
 * torch-gobject/nn/options/torch-nn-grid-sample-padding-mode.h
 *
 * GridSample padding mode.
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
 * TorchNNGridSamplePaddingMode
 * @TORCH_NN_GRID_SAMPLE_PADDING_MODE_ZEROS: Pad edges with zeros
 * @TORCH_NN_GRID_SAMPLE_PADDING_MODE_BORDER: Pad edges with border pixels
 * @TORCH_NN_GRID_SAMPLE_PADDING_MODE_REFLECTION: Pad edges by reflecting image at borders
 */
typedef enum {
  TORCH_NN_GRID_SAMPLE_PADDING_MODE_ZEROS,
  TORCH_NN_GRID_SAMPLE_PADDING_MODE_BORDER,
  TORCH_NN_GRID_SAMPLE_PADDING_MODE_REFLECTION
} TorchNNGridSamplePaddingMode;