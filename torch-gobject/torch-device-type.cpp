/*
 * torch-gobject/torch-device-type.cpp
 *
 * Device type specifiers.
 *
 * Copyright (C) 2020 Sam Spilsbury.
 *
 * torch-gobject is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * torch-gobject is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with torch-gobject; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include <torch-gobject/torch-device-type.h>
#include <torch-gobject/torch-device-type-internal.h>

c10::DeviceType
torch_device_type_get_real_device_type (TorchDeviceType device_type)
{
  switch (device_type)
    {
      case TORCH_DEVICE_TYPE_CPU:
        return c10::DeviceType::CPU;
      case TORCH_DEVICE_TYPE_VULKAN:
        return c10::DeviceType::Vulkan;
      case TORCH_DEVICE_TYPE_CUDA:
        return c10::DeviceType::CUDA;
      default:
        throw std::runtime_error ("Unsupported device type");
    }
}

TorchDeviceType
torch_device_type_from_real_device_type (c10::DeviceType device_type)
{
  switch (device_type)
    {
      case c10::DeviceType::CPU:
        return TORCH_DEVICE_TYPE_CPU;
      case c10::DeviceType::Vulkan:
        return TORCH_DEVICE_TYPE_VULKAN;
      case c10::DeviceType::CUDA:
        return TORCH_DEVICE_TYPE_CUDA;
      default:
        throw std::runtime_error ("Unsupported device type");
    }
}
