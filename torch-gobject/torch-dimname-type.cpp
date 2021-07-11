/*
 * torch-gobject/torch-dimname-type.cpp
 *
 * Dimname type specifiers.
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

#include <torch-gobject/torch-enums.h>
#include <torch-gobject/torch-dimname-internal.h>
#include <torch-gobject/torch-dimname-type.h>



at::NameType
torch_dimname_type_get_real_type (TorchDimnameType dimname_type)
{
  switch (dimname_type)
    {
      case TORCH_DIMNAME_TYPE_BASIC:
        return at::NameType::BASIC;
      case TORCH_DIMNAME_TYPE_WILDCARD:
        return at::NameType::BASIC;
      default:
        throw std::runtime_error ("Unsupported dimname type");
    }
}

TorchDimnameType
torch_dimname_type_from_real_type (at::NameType dimname_type)
{
  switch (dimname_type)
    {
      case at::NameType::BASIC:
        return TORCH_DIMNAME_TYPE_BASIC;
      case at::NameType::WILDCARD:
        return TORCH_DIMNAME_TYPE_WILDCARD;
      default:
        throw std::runtime_error ("Unsupported dimname type");
    }
}
