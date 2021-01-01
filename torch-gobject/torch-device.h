/*
 * torch-gobject/torch-device.h
 *
 * Device abstraction for creating devices.
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

#pragma once

#include <glib-object.h>
#include <torch-gobject/torch-device-type.h>

G_BEGIN_DECLS

#define TORCH_TYPE_DEVICE torch_device_get_type ()
G_DECLARE_FINAL_TYPE (TorchDevice, torch_device, TORCH, DEVICE, GObject)

TorchDevice * torch_device_new (GError **error);

TorchDevice * torch_device_new_from_string (const char  *string,
                                            GError     **error);

TorchDevice * torch_device_new_from_type_index (TorchDeviceType   type,
                                                short             index,
                                                GError          **error);

gboolean torch_device_get_index (TorchDevice  *device,
                                 short        *out_index,
                                 GError      **error);

gboolean torch_device_get_device_type (TorchDevice      *device,
                                       TorchDeviceType  *out_device_type,
                                       GError          **error);

char * torch_device_get_string (TorchDevice  *device,
                                GError      **error);

G_END_DECLS
