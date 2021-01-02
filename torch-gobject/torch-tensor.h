/*
 * torch-gobject/torch-tensor.h
 *
 * Tensor abstraction for data to be passed to a tensor.
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

#include <glib.h>
#include <glib-object.h>

#include <torch-gobject/torch-device.h>
#include <torch-gobject/torch-tensor-index.h>

G_BEGIN_DECLS

#define TORCH_TYPE_TENSOR torch_tensor_get_type ()
G_DECLARE_FINAL_TYPE (TorchTensor, torch_tensor, TORCH, TENSOR, GObject)

TorchTensor * torch_tensor_new (void);

TorchTensor * torch_tensor_new_from_data (GVariant *data);

TorchTensor * torch_tensor_index_array (TorchTensor  *tensor,
                                        GPtrArray    *indices,
                                        GError      **error);

TorchTensor * torch_tensor_index_list (TorchTensor  *tensor,
                                       GList        *indices,
                                       GError      **error);

TorchTensor * torch_tensor_index_array_steal (TorchTensor  *tensor,
                                              GPtrArray    *indices,
                                              GError      **error);

TorchTensor * torch_tensor_index (TorchTensor  *tensor,
                                  GError      **error,
                                  TorchIndex   *index,
                                  ...);

TorchTensor * torch_tensor_copy_to_device (TorchTensor  *tensor,
                                           TorchDevice  *device,
                                           GError      **error);

TorchTensor * torch_tensor_copy_to_cpu (TorchTensor  *tensor,
                                        GError      **error);

GType torch_tensor_get_dtype (TorchTensor  *tensor,
                              GError      **error);

GVariant * torch_tensor_get_tensor_data (TorchTensor  *tensor,
                                         GError      **error);

gboolean torch_tensor_set_data (TorchTensor  *tensor,
                                GVariant     *data,
                                GError      **error);

gboolean torch_tensor_set_dims (TorchTensor  *tensor,
                                GList        *dims,
                                GError      **error);

GList * torch_tensor_get_dims (TorchTensor  *tensor,
                               GError      **error);

G_END_DECLS
