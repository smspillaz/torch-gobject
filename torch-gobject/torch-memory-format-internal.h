/*
 * torch-gobject/torch-memory-format-internal.h
 *
 * Memory format specifiers, internal functions.
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

#include <torch-gobject/torch-memory-format.h>
#include <torch-gobject/torch-util.h>

#include <torch/torch.h>

c10::MemoryFormat torch_memory_format_get_real_memory_format (TorchMemoryFormat memory_format);

TorchMemoryFormat torch_memory_format_from_real_memory_format (c10::MemoryFormat memory_format);

namespace torch
{
  namespace gobject
  {
    template<>
    struct ConversionTrait<TorchMemoryFormat>
    {
      typedef c10::MemoryFormat real_type;
      static constexpr auto from = torch_memory_format_get_real_memory_format;
      static constexpr auto to = torch_memory_format_from_real_memory_format;
    };

    template<>
    struct ReverseConversionTrait<c10::MemoryFormat>
    {
      typedef TorchMemoryFormat gobject_type;
    };
  }
}
