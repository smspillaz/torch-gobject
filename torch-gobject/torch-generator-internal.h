/*
 * torch-gobject/torch-generator-internal.h
 *
 * Generator abstraction for creating RNGs, internal functions
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

#include <torch-gobject/torch-generator.h>
#include <torch-gobject/torch-util.h>

#include <torch/torch.h>

at::Generator & torch_generator_get_real_generator (TorchGenerator *generator);

TorchGenerator * torch_generator_new_from_real_generator (at::Generator const &generator);

gboolean torch_generator_init_internal (TorchGenerator  *generator,
                                        GError         **error);

namespace torch
{
  namespace gobject
  {
    template<>
    struct ConversionTrait<TorchGenerator *>
    {
      typedef at::Generator & real_type;
      static constexpr auto from = torch_generator_get_real_generator;
      static constexpr auto to = torch_generator_new_from_real_generator;
    };

    template<>
    struct ReverseConversionTrait<at::Generator>
    {
      typedef TorchGenerator * gobject_type;
    };
  }
}
