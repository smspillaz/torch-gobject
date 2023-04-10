/*
 * torch-gobject/torch-storage-internal.h
 *
 * Storage abstraction, internal functions
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

#include <torch-gobject/torch-storage.h>
#include <torch-gobject/torch-util.h>

#include <torch/torch.h>

c10::Storage & torch_storage_get_real_storage (TorchStorage *storage);

TorchStorage * torch_storage_new_from_real_storage (c10::Storage const &storage);

gboolean torch_storage_init_internal (TorchStorage *storage, GError **error);

namespace torch
{
  namespace gobject
  {
    template<>
    struct ConversionTrait<TorchStorage *>
    {
      typedef c10::Storage & real_type;
      static constexpr auto from = torch_storage_get_real_storage;
      static constexpr auto to = torch_storage_new_from_real_storage;
    };

    template<>
    struct ReverseConversionTrait<c10::Storage>
    {
      typedef TorchStorage * gobject_type;
    };
  }
}
