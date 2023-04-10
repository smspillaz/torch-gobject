/*
 * torch-gobject/torch-allocator-internal.h
 *
 * Allocator wrapper, internal functions
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

#include <torch-gobject/torch-allocator.h>
#include <torch-gobject/torch-util.h>

#include <torch/torch.h>

c10::Allocator & torch_allocator_get_real_allocator (TorchAllocator *allocator);

TorchAllocator * torch_allocator_new_from_real_allocator (c10::Allocator const &allocator);

namespace torch
{
  namespace gobject
  {
    template<>
    struct ConversionTrait<TorchAllocator *>
    {
      typedef c10::Allocator & real_type;
      static constexpr auto from = torch_allocator_get_real_allocator;
      static constexpr auto to = torch_allocator_new_from_real_allocator;
    };

    template<>
    struct ReverseConversionTrait<c10::Allocator>
    {
      typedef TorchAllocator * gobject_type;
    };
  }
}
