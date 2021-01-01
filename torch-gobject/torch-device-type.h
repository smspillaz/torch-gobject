/*
 * torch-gobject/torch-device-type.h
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

#pragma once

#include <glib.h>

G_BEGIN_DECLS

/**
 * TorchDeviceType
 * @TORCH_DEVICE_TYPE_CPU: CPU-device bound tensor, operations in software
 * @TORCH_DEVICE_TYPE_VULKAN: GPU-device tensor, operations in Vulkan
 * @TORCH_DEVICE_TYPE_CUDA: GPU-device tensor, operations in CUDA
 *
 * Error enumeration for Scorch related errors.
 */
typedef enum {
  TORCH_DEVICE_TYPE_CPU,
  TORCH_DEVICE_TYPE_VULKAN,
  TORCH_DEVICE_TYPE_CUDA
} TorchDeviceType;

G_END_DECLS
