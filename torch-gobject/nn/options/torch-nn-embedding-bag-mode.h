/*
 * torch-gobject/nn/options/torch-nn-embedding-bag-mode.h
 *
 * EmbeddingBag mode.
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
 * TorchNNEmbeddingBagMode
 * @TORCH_NN_EMBEDDING_BAG_MODE_SUM: Equivalent to using an embedding and taking sum over weights
 * @TORCH_NN_EMBEDDING_BAG_MODE_MEAN: Equivalent to using an embedding and taking mean over weights
 * @TORCH_NN_EMBEDDING_BAG_MODE_MAX: Equivalent to using an embedding and taking max over weights
 */
typedef enum {
  TORCH_NN_EMBEDDING_BAG_MODE_SUM,
  TORCH_NN_EMBEDDING_BAG_MODE_MEAN,
  TORCH_NN_EMBEDDING_BAG_MODE_MAX
} TorchNNEmbeddingBagMode;