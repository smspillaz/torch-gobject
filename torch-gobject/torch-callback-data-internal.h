/*
 * torch-gobject/torch-callback-data-internal.h
 *
 * Lightweight container object for callback data. 
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

#include <utility>
#include <torch-gobject/torch-callback-data.h>

struct _TorchCallbackData {
  gpointer       callback;
  gpointer       user_data;
  GDestroyNotify user_data_destroy;
  guint          ref_count;
};

template <typename FuncType>
class TorchCallbackDataCallableWrapper
{
  public:
    TorchCallbackDataCallableWrapper (TorchCallbackData *callback_data) : 
      callback_data (torch_callback_data_ref (callback_data))
    {
    }

    TorchCallbackDataCallableWrapper (TorchCallbackDataCallableWrapper const &other) :
      callback_data (torch_callback_data_ref (other.callback_data))
    {
    }

    TorchCallbackDataCallableWrapper (TorchCallbackDataCallableWrapper &&other) :
      callback_data (std::exchange(other.callback_data, nullptr))
    {
    }

    ~TorchCallbackDataCallableWrapper ()
    {
      torch_callback_data_unref (callback_data);
    }

    TorchCallbackDataCallableWrapper & operator=(TorchCallbackDataCallableWrapper const &other)
    {
      *this = TorchCallbackDataCallableWrapper (other);
    }

    TorchCallbackDataCallableWrapper & operator=(TorchCallbackDataCallableWrapper &&other)
    {
      std::swap(*this, other);
      return *this;
    }

    friend void swap (TorchCallbackDataCallableWrapper &lhs, TorchCallbackDataCallableWrapper &rhs)
    {
      using std::swap;

      swap(lhs.callback_data, rhs.callback_data);
    }

    template <typename... Args>
    typename std::result_of<FuncType(Args&& ..., gpointer)>::type operator()(Args&& ...args) const
    {
      return reinterpret_cast <FuncType> (callback_data->callback) (args..., callback_data->user_data);
    }
  private:
    TorchCallbackData *callback_data;
};
