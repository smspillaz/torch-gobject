/*
 * torch-gobject/torch-memory-format.h
 *
 * Memory format specifiers.
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
 * TorchMemoryFormat
 * @TORCH_MEMORY_FORMAT_PRESERVE: If any of the input tensors is in channels_last format,
 *                                operator output should be in channels_last format
 * @TORCH_MEMORY_FORMAT_CONTIGUOUS: Regardless of input tensors format, the output should be contiguous Tensor.
 * @TORCH_MEMORY_FORMAT_CHANNELS_LAST: Regardless of input tensors format, the output should be
 *                                     in channels_last format.
 * @TORCH_MEMORY_FORMAT_CHANNELS_LAST_3D: Regardless of input tensors format, the output should be
 *                                        in channels_last_3d format. 
 *
 * Error enumeration for Scorch related errors.
 */
typedef enum {
  TORCH_MEMORY_FORMAT_PRESERVE,
  TORCH_MEMORY_FORMAT_CONTIGUOUS,
  TORCH_MEMORY_FORMAT_CHANNELS_LAST,
  TORCH_MEMORY_FORMAT_CHANNELS_LAST_3D,
} TorchMemoryFormat;

G_END_DECLS
