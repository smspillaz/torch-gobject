/*
 * torch-gobject/torch-util-internal.h
 *
 * Internal type conversion functions
 *
 * Copyright (C) 2020 Sam Spilsbury.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#pragma once

#include <algorithm>
#include <vector>

#include <glib.h>
#include <glib-object.h>

#include <c10/util/ArrayRef.h>
#include <c10/core/Scalar.h>
#include <ATen/TypeDefault.h>

namespace
{
  template <typename T>
  c10::ArrayRef <T>
  torch_array_ref_from_fixed_array (const T *array_data, size_t size)
  {
    return c10::ArrayRef <T> (array_data, size);
  }

  template <typename T>
  c10::ArrayRef <T>
  torch_array_ref_from_garray (GArray *array)
  {
    return c10::ArrayRef <T> (reinterpret_cast <T *> (array->data), array->len);
  }

  template <typename T>
  GArray *
  torch_new_garray_from_array_ref (c10::ArrayRef <T> const &array_ref)
  {
    g_autoptr (GArray) array = g_array_sized_new (FALSE, FALSE, sizeof (T), array_ref.size ());
    const T *array_data = reinterpret_cast <T *> (array->data);

    std::copy (array_ref.begin(), array_ref.end(), array_data);

    return static_cast <GArray *> (g_steal_pointer (&array));
  }
}


GValue * torch_gvalue_from_scalar (const c10::Scalar &scalar);

c10::Scalar torch_scalar_from_gvalue (const GValue *value);

GType torch_gtype_from_scalar_type (const c10::ScalarType &scalar_type);

c10::ScalarType torch_scalar_type_from_gtype (GType type);

GPtrArray * torch_tensor_ptr_array_from_tensor_list (at::TensorList const &list);

std::vector <at::Tensor> torch_tensor_list_from_tensor_ptr_array (GPtrArray *array);


