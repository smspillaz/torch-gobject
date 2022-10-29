/*
 * torch-gobject/torch-nn-any-module-internal.h
 *
 * NN Module, internal functions
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

#pragma once

#include <torch-gobject/nn/torch-nn-any-module.h>
#include <torch/torch.h>

TorchNNAnyModule * torch_nn_any_module_new_from_real_any_module (torch::nn::AnyModule const &real_module);

template <typename T>
TorchNNAnyModule * torch_nn_any_module_convert_from_real_module (T const &real_module)
{
  return torch_nn_any_module_new_from_real_any_module (torch::nn::AnyModule (real_module));
}

torch::nn::AnyModule const & torch_nn_any_module_to_real_any_module (TorchNNAnyModule *any_module);
