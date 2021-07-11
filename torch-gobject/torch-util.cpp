/*
 * torch-gobject/torch-util-internal.cpp
 *
 * Internal type conversion functions
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

#include <vector>

#include <torch-gobject/torch-dimname.h>
#include <torch-gobject/torch-dimname-internal.h>
#include <torch-gobject/torch-tensor.h>
#include <torch-gobject/torch-tensor-internal.h>
#include <torch-gobject/torch-util.h>

void torch_throw_error (GError *error)
{
  if (error != NULL)
    throw std::runtime_error (error->message);
}

GValue *
torch_gvalue_from_scalar (c10::Scalar const &scalar)
{
  g_autofree GValue *value = static_cast <GValue *> (g_malloc (sizeof (GValue)));
  *value = G_VALUE_INIT;

  switch (scalar.type ())
    {
      case c10::ScalarType::Float:
        g_value_init (value, G_TYPE_DOUBLE);
        g_value_set_float (value, scalar.to <float> ());
        return static_cast <GValue *> g_steal_pointer (&value);
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
      case c10::ScalarType::Float:
        return G_TYPE_FLOAT;
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
        return c10::ScalarType::Float;
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

namespace {
  template <typename InternalType, typename ConversionFunc>
  GPtrArray * object_ptr_array_from_object_array_ref (c10::ArrayRef<InternalType> const &array, ConversionFunc &&conv)
  {
    g_autoptr (GPtrArray) ptr_array = g_ptr_array_new_with_free_func (g_object_unref);

    for (auto const &object: array)
      g_ptr_array_add (ptr_array, conv (object));

    return static_cast <GPtrArray *> (g_steal_pointer (&ptr_array));
  }

  template <typename InternalType, typename LibraryType, typename ConversionFunc>
  std::vector<InternalType> object_vector_from_object_ptr_array (GPtrArray *ptr_array, ConversionFunc &&conv)
  {
    std::vector <InternalType> array;

    for (size_t i = 0; i < ptr_array->len; ++i)
      array.push_back(conv (static_cast <LibraryType> (g_ptr_array_index (ptr_array, i))));

    return array;
  }
}

GPtrArray * torch_tensor_ptr_array_from_tensor_list (at::TensorList const &list)
{
  return object_ptr_array_from_object_array_ref (list, torch_tensor_new_from_real_tensor);
}

std::vector <at::Tensor> torch_tensor_list_from_tensor_ptr_array (GPtrArray *array)
{
  return object_vector_from_object_ptr_array <at::Tensor, TorchTensor *> (array, torch_tensor_get_real_tensor);
}

GPtrArray * torch_dimname_ptr_array_from_dimname_list (at::DimnameList const &list)
{
  return object_ptr_array_from_object_array_ref (list, torch_dimname_new_from_real_dimname);
}

std::vector <at::Dimname> torch_dimname_list_from_dimname_ptr_array (GPtrArray *array)
{
  return object_vector_from_object_ptr_array <at::Dimname, TorchDimname *> (array, torch_dimname_get_real_dimname);
}
