/*
 * torch-gobject/torch-tensor-index-array.c
 *
 * Object representing an array of indices on different tensor axes.
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

#include <glib.h>
#include <torch-gobject/torch-tensor-index.h>

/**
 * torch_tensor_index_array_new_zero_terminated_steal: (skip)
 * @indices: (array zero-terminated=1): A zero-terminated array of indices.
 *                                      which are transferred to the returned GPtrArray
 *
 * Make a new #GPtrArray of #TorchIndex where the pointers are transferred from
 * @indices to the return value (such that all the pointers in @indices) will
 * be %NULL once the transfer is complete.
 *
 * Returns: (transfer full) (element-type TorchIndex): A new #GPtrArray of #TorchIndex
 *                                                     which owns the elements
 */
GPtrArray *
torch_tensor_index_array_new_zero_terminated_steal (TorchIndex **indices)
{
  g_autoptr (GPtrArray) array = g_ptr_array_new_with_free_func ((GDestroyNotify) torch_index_free);

  for (size_t i = 0; indices[i] != NULL; ++i)
    g_ptr_array_add (array, indices[i]);

  return g_steal_pointer (&array);
}

/**
 * torch_tensor_index_array_new_va: (skip)
 * @index: The first index
 *
 * A va_list version of %torch_tensor_index_array_new_zero_terminated
 * which allows a more ergonomic construction of a #GArray of #TorchIndex .
 *
 * Returns: (transfer full) (element-type TorchIndex): A new #GPtrArray of #TorchIndex
 *                                                     which owns the elements
 */
GPtrArray *
torch_tensor_index_array_new_va (TorchIndex *index, va_list ap)
{
  g_autoptr (GPtrArray) array = g_ptr_array_new_with_free_func ((GDestroyNotify) torch_index_free);
  TorchIndex *next = index;

  while (next != NULL)
    {
      g_ptr_array_add (array, next);
      next = va_arg (ap, TorchIndex *);
    }

  return g_steal_pointer (&array);
}

/**
 * torch_tensor_index_array_new: (skip)
 * @index: The first index
 *
 * A varargs version of %torch_tensor_index_array_new_zero_terminated
 * which allows a more ergonomic construction of a #GArray of #TorchIndex .
 * The argument list must be terminated by %NULL and the ownership of the
 * arguments are transferred to the #GPtrArray
 *
 * Returns: (transfer full) (element-type TorchIndex): A new #GPtrArray of #TorchIndex
 *                                                     which owns the elements
 */
GPtrArray *
torch_tensor_index_array_new (TorchIndex *index, ...)
{
  g_autoptr (GPtrArray) array = NULL;
  va_list ap;

  va_start (ap, index);
  array = torch_tensor_index_array_new_va (index, ap);
  va_end (ap);

  return g_steal_pointer (&array);
}
