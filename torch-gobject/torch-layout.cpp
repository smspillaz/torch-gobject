/*
 * torch-gobject/torch-layout.cpp
 *
 * Layout type specifiers.
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

#include <torch-gobject/torch-layout.h>
#include <torch-gobject/torch-layout-internal.h>

c10::Layout
torch_layout_get_real_layout (TorchLayout layout)
{
  switch (layout)
    {
      case TORCH_LAYOUT_STRIDED:
        return c10::Layout::Strided;
      case TORCH_LAYOUT_SPARSE:
        return c10::Layout::Sparse;
      case TORCH_LAYOUT_MKLDNN:
        return c10::Layout::Mkldnn;
      default:
        throw std::runtime_error ("Unsupported layout type");
    }
}

TorchLayout
torch_layout_from_real_layout (c10::Layout layout)
{
  switch (layout)
    {
      case c10::Layout::Strided:
        return TORCH_LAYOUT_STRIDED;
      case c10::Layout::Sparse:
        return TORCH_LAYOUT_SPARSE;
      case c10::Layout::Mkldnn:
        return TORCH_LAYOUT_MKLDNN;
      default:
        throw std::runtime_error ("Unsupported layout type");
    }
}
