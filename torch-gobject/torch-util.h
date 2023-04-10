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
#include <ATen/core/ivalue.h>
#include <ATen/Dimname.h>
#include <ATen/TypeDefault.h>

#include <torch-gobject/torch-optional-value.h>

namespace torch
{
  namespace gobject
  {
    template<typename T>
    struct ConversionTrait
    {
      typedef T real_type;
      static T (*from) (T src);
      static T (*to) (T src);
    };

    template<typename T>
    struct ReverseConversionTrait
    {
      typedef T gobject_type;
    };

    template<typename T>
    struct ReverseConversionTraitConverter
    {
      static constexpr auto from = ConversionTrait<typename ReverseConversionTrait<T>::gobject_type>::to;
      static constexpr auto to = ConversionTrait<typename ReverseConversionTrait<T>::gobject_type>::from;
    };
  }
}

namespace
{
  template <typename T>
  T *
  torch_copy_assign (T *ptr)
  {
    if (ptr == nullptr)
      return nullptr;

    T *copy = g_new0 (T, 1);
    *copy = *ptr;

    return copy;
  }

  template <typename T>
  c10::optional <T>
  torch_pointer_to_optional (T *ptr)
  {
    if (ptr == nullptr)
      {
        return c10::nullopt;
      }

    return c10::optional <T> (*ptr);
  }

  template <typename T>
  std::vector <T>
  torch_c_array_to_vector (T *c_array, size_t length)
  {
    std::vector <T> vec (length);
    std::copy (c_array, (c_array + length), vec.begin ());

    return vec;
  }

  template <typename T>
  std::vector <T>
  torch_g_array_to_vector (GArray *array, size_t length)
  {
    return torch_c_array_to_vector <T> (reinterpret_cast <T *> (array->data), array->len);
  }

  template <typename T, typename C>
  std::vector <C>
  torch_c_array_to_vector_convert (T *c_array, size_t length)
  {
    std::vector <C> vec (length);
    std::transform (c_array, (c_array + length), vec.begin (), [](T x) { return static_cast <C> (x); });

    return vec;
  }

  template <typename T, typename C>
  std::vector <C>
  torch_g_array_to_vector_convert (GArray *array, size_t length)
  {
    return torch_c_array_to_vector_convert <T, C> (reinterpret_cast <T *> (array->data), array->len);
  }

  namespace internal
  {
    template <typename T>
    struct ValueConversion
    {
      typedef T Type;
    };

    template <>
    struct ValueConversion<gboolean>
    {
      typedef bool Type;
    };
  }

  template <typename T>
  c10::optional <typename internal::ValueConversion<T>::Type>
  torch_optional_value_to_c10_optional (TorchOptionalValue *value, T (*getter)(TorchOptionalValue *))
  {
    if (value == nullptr)
      {
        return c10::nullopt;
      }

    return c10::optional <typename internal::ValueConversion<T>::Type> ((*getter) (value));
  }

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

  template <typename T>
  size_t
  sentinel_terminated_array_length (T *c_array, int sentinel)
  {
    size_t i = 0;

    while ((*c_array++) != sentinel)
      ++i;

    return i;
  }

  template <typename T>
  GArray *
  torch_new_g_array_from_c_array (T *c_array, int fixed_length)
  {
    g_assert (fixed_length >= 0);

    g_autoptr (GArray) array = g_array_sized_new (FALSE, FALSE, sizeof (T), fixed_length);
    T *array_data = reinterpret_cast <T *> (array->data);

    if (fixed_length > 0)
      {
        std::copy (c_array, (c_array + fixed_length), array_data);
      }

    array->len = fixed_length;
    return static_cast <GArray *> (g_steal_pointer (&array));
  }

  template <typename T>
  GPtrArray *
  torch_new_g_ptr_array_from_c_array_null_terminated (T **c_array, GBoxedCopyFunc copy, GDestroyNotify destructor)
  {
    g_autoptr (GPtrArray) array = g_ptr_array_new_null_terminated (
      sentinel_terminated_array_length(c_array),
      destructor
    );
    T *array_data = reinterpret_cast <T *> (array->pdata);

    while (*c_array != nullptr)
      {
        g_ptr_array_add (copy (*c_array));
      }

    return static_cast <GPtrArray *> (g_steal_pointer (&array));
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

GPtrArray * torch_tensor_ptr_array_from_optional_tensor_list (c10::List <c10::optional <at::Tensor> > const &list);

c10::List <c10::optional <at::Tensor> > torch_optional_tensor_list_from_tensor_ptr_array (GPtrArray *array);

GPtrArray * torch_dimname_ptr_array_from_dimname_list (at::DimnameList const &list);

std::vector <at::Dimname> torch_dimname_list_from_dimname_ptr_array (GPtrArray *array);


