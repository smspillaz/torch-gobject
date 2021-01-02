/*
 * torch-gobject/torch-tensor-index.cpp
 *
 * Object representing an index of a tensor.
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

#include <torch-gobject/torch-tensor.h>
#include <torch-gobject/torch-tensor-index.h>
#include <torch-gobject/torch-tensor-index-internal.h>
#include <torch-gobject/torch-tensor-index-type.h>
#include <torch-gobject/torch-tensor-internal.h>
#include <torch-gobject/torch-slice.h>
#include <torch-gobject/torch-slice-internal.h>

struct _TorchIndex {
  TorchTensorIndexType index_type;
  union {
      TorchSlice  *v_slice;
      TorchTensor *v_tensor;
      int64_t      v_int64;
  } value;
};

GType torch_index_get_type ()
{
  static GType index_type = 0;

  if (g_once_init_enter (&index_type))
    {
      index_type = g_boxed_type_register_static ("TorchIndex",
                                                 (GBoxedCopyFunc) torch_index_copy,
                                                 (GBoxedFreeFunc) torch_index_free);
    }

  return index_type;
}

torch::indexing::TensorIndex
torch_index_get_real_index (TorchIndex *index)
{
  switch (index->index_type)
    {
      case TORCH_TENSOR_INDEX_TYPE_NONE:
        return torch::indexing::TensorIndex (nullptr);
      case TORCH_TENSOR_INDEX_TYPE_ELLIPSIS:
        return torch::indexing::TensorIndex (torch::indexing::Ellipsis);
      case TORCH_TENSOR_INDEX_TYPE_INTEGER:
        return torch::indexing::TensorIndex (index->value.v_int64);
      case TORCH_TENSOR_INDEX_TYPE_BOOLEAN:
        return torch::indexing::TensorIndex ((bool) (index->value.v_int64));
      case TORCH_TENSOR_INDEX_TYPE_SLICE:
        return torch::indexing::TensorIndex (torch_slice_get_real_slice (index->value.v_slice));
      case TORCH_TENSOR_INDEX_TYPE_TENSOR:
        return torch::indexing::TensorIndex (torch_tensor_get_real_tensor (index->value.v_tensor));
      default:
        throw std::logic_error ("Unable to handle tensor index type");
    }
}

static TorchIndex *
torch_index_new (TorchTensorIndexType index_type)
{
  TorchIndex *index = static_cast <TorchIndex *> (g_new0 (TorchIndex, 1));

  index->index_type = index_type;
  return index;
}

/**
 * torch_index_new_none:
 *
 * Make a new #TensorIndex of type %TORCH_TENSOR_INDEX_TYPE_NONE
 *
 * Returns: (transfer full): A new #TensorIndex
 */ 
TorchIndex *
torch_index_new_none (void)
{
  TorchIndex *index = torch_index_new (TORCH_TENSOR_INDEX_TYPE_NONE);
  return index;
}

/**
 * torch_index_new_int:
 * @integer: A gint64 with the integer for this index
 *
 * Make a new #TensorIndex of type %TORCH_TENSOR_INDEX_TYPE_INTEGER with @integer
 *
 * Returns: (transfer full): A new #TensorIndex
 */ 
TorchIndex *
torch_index_new_int (int64_t integer)
{
  TorchIndex *index = torch_index_new (TORCH_TENSOR_INDEX_TYPE_INTEGER);
  index->value.v_int64 = integer;
  return index;
}

/**
 * torch_index_new_boolean:
 * @bool_value: A gboolean with either %TRUE or %FALSE for this index
 *
 * Make a new #TensorIndex of type %TORCH_TENSOR_INDEX_TYPE_BOOLEAN with @bool_value
 *
 * Returns: (transfer full): A new #TensorIndex
 */ 
TorchIndex *
torch_index_new_boolean (gboolean bool_value)
{
  TorchIndex *index = torch_index_new (TORCH_TENSOR_INDEX_TYPE_BOOLEAN);
  index->value.v_int64 = bool_value;
  return index;
}

/**
 * torch_index_new_slice:
 * @slice: (transfer full): The #TorchSlice pertaining to this range. Consumes the reference.
 *
 * Make a new #TensorIndex of type %TORCH_TENSOR_INDEX_TYPE_SLICE with @slice
 *
 * Returns: (transfer full): A new #TensorIndex
 */ 
TorchIndex *
torch_index_new_slice (TorchSlice *slice)
{
  TorchIndex *index = torch_index_new (TORCH_TENSOR_INDEX_TYPE_SLICE);
  index->value.v_slice = slice;
  return index;
}

/**
 * torch_index_new_range:
 * @start: The start value of this range
 * @stop: The stop value of this range
 * @step: The step value of this range
 *
 * Make a new #TensorIndex of type %TORCH_TENSOR_INDEX_TYPE_SLICE with
 * slice arguments @start, @stop and @step
 *
 * Returns: (transfer full): A new #TensorIndex
 */ 
TorchIndex *
torch_index_new_range (int64_t start, int64_t stop, int64_t step)
{
  return torch_index_new_slice (torch_slice_new (start, stop, step));
}

/**
 * torch_index_new_ellipsis:
 *
 * Make a new #TensorIndex of type %TORCH_TENSOR_INDEX_TYPE_ELLIPSIS
 *
 * Returns: (transfer full): A new #TensorIndex
 */ 
TorchIndex *
torch_index_new_ellipsis (void)
{
  TorchIndex *index = torch_index_new (TORCH_TENSOR_INDEX_TYPE_ELLIPSIS);
  return index;
}

/**
 * torch_index_new_tensor:
 * @tensor: (transfer none): The #TorchTensor containing the index values for this axis
 *
 * Make a new #TensorIndex of type %TORCH_TENSOR_INDEX_TYPE_TENSOR
 *
 * Returns: (transfer full): A new #TensorIndex
 */ 
TorchIndex *
torch_index_new_tensor (TorchTensor *tensor)
{
  TorchIndex *index = torch_index_new (TORCH_TENSOR_INDEX_TYPE_TENSOR);
  index->value.v_tensor = static_cast <TorchTensor *> (g_object_ref (tensor));
  return index;
}

/**
 * torch_index_copy:
 * @index: A #TorchIndex
 *
 * Make a copy of this #TorchIndex
 *
 * Returns: (transfer full): A new #TorchIndex with the same value as @index
 */
TorchIndex *
torch_index_copy (TorchIndex *index)
{
  TorchIndex *new_index = torch_index_new (index->index_type);
  new_index->value = index->value;

  switch (index->index_type)
    {
      case TORCH_TENSOR_INDEX_TYPE_SLICE:
        new_index->value.v_slice = torch_slice_copy (new_index->value.v_slice);
        break;
      case TORCH_TENSOR_INDEX_TYPE_TENSOR:
        new_index->value.v_tensor = static_cast <TorchTensor *> (g_object_ref (new_index->value.v_tensor));
        break;
      default:
        break;
    }

  return new_index;
}

/**
 * torch_index_free:
 * @index: (transfer full): A #TorchIndex
 *
 * Free this torch index.
 */
void torch_index_free (TorchIndex *index)
{
  /* Because we consume the slice, we need to free it here */
  switch (index->index_type)
    {
      case TORCH_TENSOR_INDEX_TYPE_SLICE:
        g_clear_pointer (&index->value.v_slice, torch_slice_free);
        break;
      case TORCH_TENSOR_INDEX_TYPE_TENSOR:
        g_clear_pointer (&index->value.v_tensor, g_object_unref);
        break;
      default:
        break;
    }

  g_free (index);
}
