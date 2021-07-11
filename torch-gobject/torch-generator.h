/*
 * torch-gobject/torch-generator.h
 *
 * Generator abstraction for creating RNGs.
 *
 * Copyright (C) 2021 Sam Spilsbury.
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

#include <stdint.h>

#include <glib-object.h>

#include <torch-gobject/torch-device.h>

G_BEGIN_DECLS

#define TORCH_TYPE_GENERATOR torch_generator_get_type ()
G_DECLARE_FINAL_TYPE (TorchGenerator, torch_generator, TORCH, GENERATOR, GObject)

uint64_t torch_generator_get_current_seed (TorchGenerator *generator);

gboolean torch_generator_set_current_seed (TorchGenerator  *generator,
                                           uint64_t         seed,
                                           GError         **error);

TorchDevice * torch_generator_get_device (TorchGenerator  *generator,
                                          GError         **error);

gboolean torch_generator_set_device (TorchGenerator  *generator,
                                     TorchDevice     *device,
                                     GError         **error);

gboolean torch_generator_seed (TorchGenerator  *generator,
                               uint64_t        *out_seed,
                               GError         **error);

gboolean torch_generator_lock (TorchGenerator  *generator,
                               GError         **error);

gboolean torch_generator_unlock (TorchGenerator  *generator,
                                 GError         **error);

G_END_DECLS
