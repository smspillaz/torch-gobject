/*
 * torch-gobject/nn/options/torch-nn-namedshape-element.cpp
 *
 * An element of a named shape record.
 *
 * Copyright (C) 2022 Sam Spilsbury.
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

#include <torch-gobject/nn/options/torch-nn-namedshape-element.h>
#include <torch-gobject/nn/options/torch-nn-namedshape-element-internal.h>

torch::nn::UnflattenOptions::namedshape_t
torch_nn_namedshape_array_to_real_namedshape (GArray *array, size_t n)
{
  TorchNNNamedshapeElement *array_data = reinterpret_cast <TorchNNNamedshapeElement *> (array->data);
  torch::nn::UnflattenOptions::namedshape_t real_namedshape;

  for (size_t i = 0; i < n; ++i) {
    real_namedshape.emplace_back (std::pair <std::string, int> (array_data[i].name, array_data[i].dim));
  }

  return real_namedshape;
}