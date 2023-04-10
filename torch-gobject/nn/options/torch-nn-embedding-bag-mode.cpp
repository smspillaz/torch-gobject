/*
 * torch-gobject/nn/options/torch-nn-embedding-bag-mode.cpp
 *
 * EmbeddingBag mode implementation
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

#include <torch/enum.h>
#include <torch-gobject/nn/options/torch-nn-embedding-bag-mode.h>
#include <torch-gobject/nn/options/torch-nn-embedding-bag-mode-internal.h>

torch::nn::EmbeddingBagMode
torch_nn_embedding_bag_mode_to_real_embedding_bag_mode (TorchNNEmbeddingBagMode mode)
{
  switch (mode)
    {
      case TORCH_NN_EMBEDDING_BAG_MODE_SUM:
        return torch::enumtype::kSum();
      case TORCH_NN_EMBEDDING_BAG_MODE_MEAN:
        return torch::enumtype::kMean();
      case TORCH_NN_EMBEDDING_BAG_MODE_MAX:
        return torch::enumtype::kMax();
      default:
        throw std::logic_error("Invalid embedding bag mode");
    }
}

TorchNNEmbeddingBagMode
torch_nn_embedding_bag_mode_from_real_embedding_bag_mode (torch::nn::EmbeddingBagMode const &mode)
{
  if (c10::get_if<torch::enumtype::kSum> (&mode)) {
    return TORCH_NN_EMBEDDING_BAG_MODE_SUM;
  }

  if (c10::get_if<torch::enumtype::kMean> (&mode)) {
    return TORCH_NN_EMBEDDING_BAG_MODE_MEAN;
  }

  if (c10::get_if<torch::enumtype::kMax> (&mode)) {
    return TORCH_NN_EMBEDDING_BAG_MODE_MAX;
  }

  throw std::logic_error ("Invalid embedding bag mode");
}