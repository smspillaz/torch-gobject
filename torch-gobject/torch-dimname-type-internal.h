/*
 * torch-gobject/torch-dimname-type-internal.h
 *
 * Dimname type specifiers, internal functions.
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

#include <torch-gobject/torch-dimname-type.h>
#include <torch-gobject/torch-util.h>

#include <ATen/core/Dimname.h>

#include <torch/torch.h>

at::NameType torch_dimname_type_get_real_type (TorchDimnameType type);

TorchDimnameType torch_dimname_type_from_real_type (at::NameType type);

namespace torch
{
  namespace gobject
  {
    template<>
    struct ConversionTrait<TorchDimnameType>
    {
      typedef at::NameType real_type;
      static constexpr auto func = torch_dimname_type_get_real_type;
      static constexpr auto to = torch_dimname_type_from_real_type;
    };

    template<>
    struct ReverseConversionTrait<at::NameType>
    {
      typedef TorchDimnameType gobject_type;
    };
  }
}
