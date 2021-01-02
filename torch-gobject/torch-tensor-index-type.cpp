/*
 * torch-gobject/torch-index-type.cpp
 *
 * TensorIndex type specifiers.
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

#include <torch/torch.h>

#include <torch-gobject/torch-tensor-index-type.h>
#include <torch-gobject/torch-tensor-index-type-internal.h>

torch::indexing::TensorIndexType
torch_index_type_get_real_index_type (TorchTensorIndexType index_type)
{
  switch (index_type)
    {
      case TORCH_TENSOR_INDEX_TYPE_NONE:
        return torch::indexing::TensorIndexType::None;
      case TORCH_TENSOR_INDEX_TYPE_ELLIPSIS:
        return torch::indexing::TensorIndexType::Ellipsis;
      case TORCH_TENSOR_INDEX_TYPE_INTEGER:
        return torch::indexing::TensorIndexType::Integer;
      case TORCH_TENSOR_INDEX_TYPE_BOOLEAN:
        return torch::indexing::TensorIndexType::Boolean;
      case TORCH_TENSOR_INDEX_TYPE_SLICE:
        return torch::indexing::TensorIndexType::Slice;
      case TORCH_TENSOR_INDEX_TYPE_TENSOR:
        return torch::indexing::TensorIndexType::Tensor;
      default:
        throw std::runtime_error ("Unsupported index type");
    }
}

TorchTensorIndexType
torch_index_type_from_real_index_type (torch::indexing::TensorIndexType index_type)
{
  switch (index_type)
    {
      case torch::indexing::TensorIndexType::None:
        return TORCH_TENSOR_INDEX_TYPE_NONE;
      case torch::indexing::TensorIndexType::Ellipsis:
        return TORCH_TENSOR_INDEX_TYPE_ELLIPSIS;
      case torch::indexing::TensorIndexType::Integer:
        return TORCH_TENSOR_INDEX_TYPE_INTEGER;
      case torch::indexing::TensorIndexType::Boolean:
        return TORCH_TENSOR_INDEX_TYPE_BOOLEAN;
      case torch::indexing::TensorIndexType::Slice:
        return TORCH_TENSOR_INDEX_TYPE_SLICE;
      case torch::indexing::TensorIndexType::Tensor:
        return TORCH_TENSOR_INDEX_TYPE_TENSOR;
      default:
        throw std::runtime_error ("Unsupported index type");
    }
}
