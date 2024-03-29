/*
 * torch-gobject/torch-slice-internal.h
 *
 * Object representing a slice, internal funcitons
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

#include <torch-gobject/torch-slice.h>
#include <torch-gobject/torch-util.h>

#include <torch/torch.h>

torch::indexing::Slice torch_slice_get_real_slice (TorchSlice *slice);

TorchSlice * torch_slice_new_from_real_slice (torch::indexing::Slice const &slice);

namespace torch
{
  namespace gobject
  {
    template<>
    struct ConversionTrait<TorchSlice *>
    {
      typedef torch::indexing::Slice real_type;
      static constexpr auto from = torch_slice_get_real_slice;
      static constexpr auto to = torch_slice_new_from_real_slice;
    };

    template<>
    struct ReverseConversionTrait<torch::indexing::Slice>
    {
      typedef TorchSlice * gobject_type;
    };
  }
}
