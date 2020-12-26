/*
 * torch-gobject/torch-memory-format.cpp
 *
 * Memory format specifiers.
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

#include <torch-gobject/torch-memory-format.h>

#include <c10/core/MemoryFormat.h>

c10::MemoryFormat
torch_memory_format_get_real_memory_format (TorchMemoryFormat memory_format)
{
  switch (memory_format)
    {
      case TORCH_MEMORY_FORMAT_PRESERVE:
        return c10::MemoryFormat::Preserve;
      case TORCH_MEMORY_FORMAT_CONTIGUOUS:
        return c10::MemoryFormat::Contiguous;
      case TORCH_MEMORY_FORMAT_CHANNELS_LAST:
        return c10::MemoryFormat::ChannelsLast;
      case TORCH_MEMORY_FORMAT_CHANNELS_LAST_3D:
        return c10::MemoryFormat::ChannelsLast3d;
      default:
        throw std::runtime_error ("Unsupported memory format");
    }
}

TorchMemoryFormat
torch_memory_format_from_real_memory_format (c10::MemoryFormat memory_format)
{
  switch (memory_format)
    {
      case c10::MemoryFormat::Preserve:
        return TORCH_MEMORY_FORMAT_PRESERVE;
      case c10::MemoryFormat::Contiguous:
        return TORCH_MEMORY_FORMAT_CONTIGUOUS;
      case c10::MemoryFormat::ChannelsLast:
        return TORCH_MEMORY_FORMAT_CHANNELS_LAST;
      case c10::MemoryFormat::ChannelsLast3d:
        return TORCH_MEMORY_FORMAT_CHANNELS_LAST_3D;
      default:
        throw std::runtime_error ("Unsupported memory format");
    }
}
