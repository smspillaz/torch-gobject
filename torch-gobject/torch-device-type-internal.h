/*
 * torch-gobject/torch-device-type-internal.h
 *
 * Device type specifiers, internal functions.
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

#include <torch-gobject/torch-device-type.h>
#include <torch-gobject/torch-util.h>

#include <torch/torch.h>

c10::DeviceType torch_device_type_get_real_device_type (TorchDeviceType device_type);

TorchDeviceType torch_device_type_from_real_device_type (c10::DeviceType device_type);

namespace torch
{
  namespace gobject
  {
    template<>
    struct ConversionTrait<TorchDeviceType>
    {
      typedef c10::DeviceType real_type;
      static constexpr auto from = torch_device_type_get_real_device_type;
      static constexpr auto to = torch_device_type_from_real_device_type;
    };

    template<>
    struct ReverseConversionTrait<c10::DeviceType>
    {
      typedef TorchDeviceType gobject_type;
    };
  }
}
