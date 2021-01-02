/*
 * torch-gobject/torch-slice.h
 *
 * Object representing a slice.
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

G_BEGIN_DECLS

typedef struct _TorchSlice TorchSlice;

#define TORCH_TYPE_SLICE torch_slice_get_type ()
GType torch_slice_get_type ();

TorchSlice * torch_slice_new (int64_t start, int64_t stop, int64_t step);

TorchSlice * torch_slice_copy (TorchSlice *slice);

void torch_slice_free (TorchSlice *slice);

G_END_DECLS
