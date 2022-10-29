
/*
 * torch-gobject/nn/options/torch-nn-loss-reduction-mode-internal.h
 *
 * GridSample mode internal functionality
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

#include <torch/enum.h>
#include <torch-gobject/nn/options/torch-nn-loss-reduction-mode.h>

template <typename VariantType>
VariantType torch_nn_loss_reduction_mode_to_real_loss_reduction_mode (TorchNNLossReductionMode mode)
{
    switch (mode)
    {
      case TORCH_NN_LOSS_REDUCTION_MODE_MEAN:
        return torch::enumtype::kMean();
      case TORCH_NN_LOSS_REDUCTION_MODE_SUM:
        return torch::enumtype::kSum();
      case TORCH_NN_LOSS_REDUCTION_MODE_NONE:
        return torch::enumtype::kNone();
      default:
        throw std::logic_error("Invalid loss reduction mode");
    }
}
