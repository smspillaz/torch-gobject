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
torch_nn_namedshape_array_to_real_namedshape (GArray *array)
{
  TorchNNNamedshapeElement *array_data = reinterpret_cast <TorchNNNamedshapeElement *> (array->data);
  torch::nn::UnflattenOptions::namedshape_t real_namedshape;

  for (size_t i = 0; i < array->len; ++i) {
    real_namedshape.emplace_back (std::pair <std::string, int> (array_data[i].name, array_data[i].dim));
  }

  return real_namedshape;
}

/* Not exactly safe - the real namedshape needs to outlive this one */
GArray *
torch_nn_namedshape_array_new_from_real_namedshape (torch::nn::UnflattenOptions::namedshape_t const &real_namedshape)
{
  GArray *array = g_array_sized_new (false, true, sizeof (TorchNNNamedshapeElement), real_namedshape.size ());
  TorchNNNamedshapeElement *array_data = reinterpret_cast <TorchNNNamedshapeElement *> (array->data);

  for (auto const &pair : real_namedshape)
    {
      array_data->name = pair.first.c_str ();
      array_data->dim = pair.second;

      ++array_data;
    }

  array->len = real_namedshape.size ();
  return array;
}
