/*
 * torch-gobject/torch-tensor-index.h
 *
 * Object representing an index of a tensor.
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
#include <torch-gobject/torch-slice.h>
#include <torch-gobject/torch-tensor-index-type.h>

G_BEGIN_DECLS

typedef struct _TorchIndex TorchIndex;
typedef struct _TorchTensor TorchTensor;

#define TORCH_TYPE_INDEX torch_index_get_type ()
GType torch_index_get_type ();

TorchIndex * torch_index_new_none (void);

TorchIndex * torch_index_new_int (int64_t integer);

TorchIndex * torch_index_new_boolean (gboolean bool_value);

TorchIndex * torch_index_new_range (int64_t start, int64_t stop, int64_t step);

TorchIndex * torch_index_new_slice (TorchSlice *slice);

TorchIndex * torch_index_new_ellipsis (void);

TorchIndex * torch_index_new_tensor (TorchTensor *tensor);

TorchIndex * torch_index_copy (TorchIndex *index);

void torch_index_free (TorchIndex *index);

G_END_DECLS
