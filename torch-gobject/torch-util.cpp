/*
 * torch-gobject/torch-util-internal.cpp
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

#include <vector>

#include <torch-gobject/torch-tensor.h>
#include <torch-gobject/torch-tensor-internal.h>
#include <torch-gobject/torch-util.h>

GValue *
torch_gvalue_from_scalar (c10::Scalar const &scalar)
{
  g_autofree GValue *value = static_cast <GValue *> (g_malloc (sizeof (GValue)));
  *value = G_VALUE_INIT;

  switch (scalar.type ())
    {
      case c10::ScalarType::Double:
        g_value_init (value, G_TYPE_DOUBLE);
        g_value_set_double (value, scalar.to <double> ());
        return static_cast <GValue *> g_steal_pointer (&value);
      case c10::ScalarType::Long:
        g_value_init (value, G_TYPE_LONG);
        g_value_set_long (value, scalar.to <long> ());
        return static_cast <GValue *> g_steal_pointer (&value);
      case c10::ScalarType::Bool:
        g_value_init (value, G_TYPE_BOOLEAN);
        g_value_set_boolean (value, scalar.to <bool> ());
        return static_cast <GValue *> g_steal_pointer (&value);
      default:
        throw std::runtime_error ("Unsupported scalar type");
    }
}

c10::Scalar torch_scalar_from_gvalue (GValue const *value)
{
  c10::Scalar scalar;

  if (G_VALUE_HOLDS_DOUBLE (value))
    return c10::Scalar (g_value_get_double (value));

  if (G_VALUE_HOLDS_FLOAT (value))
    return c10::Scalar (g_value_get_float (value));

  if (G_VALUE_HOLDS_INT (value))
    return c10::Scalar (g_value_get_int (value));

  if (G_VALUE_HOLDS_INT64 (value))
    return c10::Scalar (g_value_get_int64 (value));

  if (G_VALUE_HOLDS_LONG (value))
    return c10::Scalar (g_value_get_long (value));

  if (G_VALUE_HOLDS_BOOLEAN (value))
    return c10::Scalar (static_cast <bool> (g_value_get_boolean (value)));

  throw std::runtime_error ("Unsupported value type");
}

GType torch_gtype_from_scalar_type (c10::ScalarType const &scalar_type)
{
  switch (scalar_type)
    {
      case c10::ScalarType::Double:
        return G_TYPE_DOUBLE;
      case c10::ScalarType::Long:
        return G_TYPE_LONG;
      case c10::ScalarType::Bool:
        return G_TYPE_BOOLEAN;
      default:
        throw std::runtime_error ("Unsupported scalar type");
    }
}

c10::ScalarType torch_scalar_type_from_gtype (GType type)
{
  switch (type)
    {
      case G_TYPE_FLOAT:
      case G_TYPE_DOUBLE:
        return c10::ScalarType::Double;
      case G_TYPE_INT:
      case G_TYPE_LONG:
      case G_TYPE_INT64:
      case G_TYPE_NONE:
        return c10::ScalarType::Long;
      case G_TYPE_BOOLEAN:
        return c10::ScalarType::Bool;
      default:
        throw std::runtime_error ("Unsupported GType");
    }
}

GPtrArray * torch_tensor_ptr_array_from_tensor_list (at::TensorList const &list)
{
  g_autoptr (GPtrArray) ptr_array = g_ptr_array_new_with_free_func (g_object_unref);

  for (auto const &tensor: list)
    g_ptr_array_add (ptr_array, torch_tensor_new_from_real_tensor (tensor));

  return static_cast <GPtrArray *> (g_steal_pointer (&ptr_array));
}

std::vector <at::Tensor> torch_tensor_list_from_tensor_ptr_array (GPtrArray *array)
{
  std::vector <at::Tensor> list;

  for (size_t i = 0; i < array->len; ++i)
    list.push_back(torch_tensor_get_real_tensor (static_cast <TorchTensor *> (g_ptr_array_index (array, i))));

  return list;
}
