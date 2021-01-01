/*
 * torch-gobject/torch-storage.h
 *
 * Storage abstraction for data to be passed to a tensor.
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

#include <glib-object.h>
#include <torch-gobject/torch-allocator.h>

G_BEGIN_DECLS

#define TORCH_TYPE_STORAGE torch_storage_get_type ()
G_DECLARE_FINAL_TYPE (TorchStorage, torch_storage, TORCH, STORAGE, GObject)

TorchStorage * torch_storage_new_with_allocator (size_t          size_bytes,
                                                 TorchAllocator *allocator,
                                                 gboolean        resizable);

TorchStorage * torch_storage_new_with_reallocatable_data (size_t          size_bytes,
                                                          gpointer        data,
                                                          GDestroyNotify  destroy_func,
                                                          TorchAllocator *allocator,
                                                          gboolean        resizable);

TorchStorage * torch_storage_new_with_fixed_data (GBytes *data);

gboolean torch_storage_get_resizable (TorchStorage  *storage,
                                      gboolean      *out_resizable,
                                      GError       **error);

gboolean torch_storage_get_n_bytes (TorchStorage  *storage,
                                    size_t        *out_n_bytes,
                                    GError       **error);

const gpointer torch_storage_get_data (TorchStorage  *storage,
                                       GError       **error);

GBytes * torch_storage_get_bytes (TorchStorage  *storage,
                                  GError       **error);

TorchAllocator * torch_storage_get_allocator (TorchStorage *storage);

G_END_DECLS
