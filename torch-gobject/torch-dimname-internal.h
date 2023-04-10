/*
 * torch-gobject/torch-dimname-internal.h
 *
 * Object representing a dimension name, internal functions.
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
#include <torch-gobject/torch-dimname.h>
#include <torch-gobject/torch-util.h>

#include <ATen/core/Dimname.h>

at::Dimname & torch_dimname_get_real_dimname (TorchDimname *dimname);

TorchDimname * torch_dimname_new_from_real_dimname (const at::Dimname &dimname_real);

gboolean torch_dimname_init_internal (TorchDimname  *dimname,
                                      GError       **error);

namespace torch
{
  namespace gobject
  {
    template<>
    struct ConversionTrait<TorchDimname>
    {
      typedef at::Dimname real_type;
      static constexpr auto from = torch_dimname_get_real_dimname;
      static constexpr auto to = torch_dimname_new_from_real_dimname;
    };

    template<>
    struct ReverseConversionTrait<at::Dimname>
    {
      typedef TorchDimname * gobject_type;
    };
  }
}
