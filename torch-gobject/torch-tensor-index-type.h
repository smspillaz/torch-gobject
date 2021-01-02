/*
 * torch-gobject/torch-tensor-index-type.h
 *
 * TensorIndex type specifiers.
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

G_BEGIN_DECLS

/**
 * TorchTensorIndexType
 * @TORCH_TENSOR_INDEX_TYPE_NONE: 'None' - copies entire axis
 * @TORCH_TENSOR_INDEX_TYPE_ELLIPSIS: '...'
 * @TORCH_TENSOR_INDEX_TYPE_INTEGER: int64 valued inde 
 * @TORCH_TENSOR_INDEX_TYPE_BOOLEAN: boolean valued index
 * @TORCH_TENSOR_INDEX_TYPE_SLICE: slice index (start, stop, step)
 * @TORCH_TENSOR_INDEX_TYPE_TENSOR: tensor-valued index (pick certain elements on axis)
 *
 * Error enumeration for Scorch related errors.
 */
typedef enum {
  TORCH_TENSOR_INDEX_TYPE_NONE,
  TORCH_TENSOR_INDEX_TYPE_ELLIPSIS,
  TORCH_TENSOR_INDEX_TYPE_INTEGER,
  TORCH_TENSOR_INDEX_TYPE_BOOLEAN,
  TORCH_TENSOR_INDEX_TYPE_SLICE,
  TORCH_TENSOR_INDEX_TYPE_TENSOR
} TorchTensorIndexType;

G_END_DECLS
