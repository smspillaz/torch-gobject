/*
 * torch-gobject/torch-util-internal.h
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

#pragma once

#include <algorithm>
#include <vector>

#include <glib.h>
#include <glib-object.h>

#include <c10/util/ArrayRef.h>
#include <c10/core/Scalar.h>
#include <ATen/Dimname.h>
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

  template <typename ErrorEnum>
  unsigned int set_error_from_exception (std::exception const  &exception,
                                         GQuark                 domain,
                                         ErrorEnum              code,
                                         GError               **error)
  {
    g_set_error (error,
                 domain,
                 code,
                 "%s",
                 exception.what ());
    return 0;
  }

  template <typename Func, typename... Args>
  typename std::result_of <Func(Args..., GError **)>::type
  call_and_warn_about_gerror(const char *operation, Func &&f, Args&& ...args)
  {
    GError *error = nullptr;

    auto result = f(args..., &error);

    if (error != nullptr)
      {
        g_warning ("Could not %s: %s", operation, error->message);
        decltype(result) rv = 0;
        return rv;
      }

    return result;
  }

  template <typename Func, typename ErrorEnum, typename... Args>
  typename std::result_of <Func(Args...)>::type
  call_set_error_on_exception (GError                                         **error,
                               GQuark                                           domain,
                               ErrorEnum                                        code,
                               typename std::result_of <Func(Args...)>::type    error_return,
                               Func                                            &&func,
                               Args&&                                         ...args)
  {
    try
      {
        return func (args...);
      }
    catch (const std::exception &e)
      {
        set_error_from_exception (e, domain, code, error);
        return error_return;
      }
  }
}

void torch_throw_error (GError *error);

GValue * torch_gvalue_from_scalar (const c10::Scalar &scalar);

c10::Scalar torch_scalar_from_gvalue (const GValue *value);

GType torch_gtype_from_scalar_type (const c10::ScalarType &scalar_type);

c10::ScalarType torch_scalar_type_from_gtype (GType type);

GPtrArray * torch_tensor_ptr_array_from_tensor_list (at::TensorList const &list);

std::vector <at::Tensor> torch_tensor_list_from_tensor_ptr_array (GPtrArray *array);

GPtrArray * torch_dimname_ptr_array_from_dimname_list (at::DimnameList const &list);

std::vector <at::Dimname> torch_dimname_list_from_dimname_ptr_array (GPtrArray *array);


