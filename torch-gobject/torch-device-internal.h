/*
 * torch-gobject/torch-device-internal.h
 *
 * Device abstraction for creating devices, internal functions
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

#include <torch-gobject/torch-device.h>
#include <torch-gobject/torch-util.h>

#include <torch/torch.h>

c10::Device & torch_device_get_real_device (TorchDevice *device);

TorchDevice * torch_device_new_from_real_device (c10::Device const &device);

gboolean torch_device_init_internal (TorchDevice  *device,
                                     GError      **error);

namespace torch
{
  namespace gobject
  {
    template<>
    struct ConversionTrait<TorchDevice *>
    {
      typedef c10::Device & real_type;
      static constexpr auto from = torch_device_get_real_device;
      static constexpr auto to = torch_device_new_from_real_device;
    };

    template<>
    struct ReverseConversionTrait<c10::Device>
    {
      typedef TorchDevice * gobject_type;
    };
  }
}