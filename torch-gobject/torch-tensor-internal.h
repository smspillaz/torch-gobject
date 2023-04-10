/*
 * torch-gobject/torch-tensor-internal.h
 *
 * Tensor abstraction for data to be passed to a tensor, internal funcitons
 *
 * Copyright (C) 2020 Sam Spilsbury.
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

#include <torch-gobject/torch-tensor.h>
#include <torch-gobject/torch-util.h>

#include <torch/torch.h>

torch::Tensor & torch_tensor_get_real_tensor (TorchTensor *tensor);

TorchTensor * torch_tensor_new_from_real_tensor (torch::Tensor const &tensor);

gboolean torch_tensor_init_internal (TorchTensor  *tensor,
                                     GError      **error);

namespace torch
{
  namespace gobject
  {
    template<>
    struct ConversionTrait<TorchTensor *>
    {
      typedef torch::Tensor & real_type;
      static constexpr auto from = torch_tensor_get_real_tensor;
      static constexpr auto to = torch_tensor_new_from_real_tensor;
    };

    template<>
    struct ReverseConversionTrait<torch::Tensor>
    {
      typedef TorchTensor * gobject_type;
    };
  }
}
