/*
 * torch-gobject/torch-tensor-internal.h
 *
 * Tensor abstraction for data to be passed to a tensor, internal funcitons
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

#pragma once

#include <torch-gobject/torch-tensor.h>

#include <torch/torch.h>

at::Tensor & torch_tensor_get_real_tensor (TorchTensor *tensor);

TorchTensor * torch_tensor_new_from_real_tensor (at::Tensor const &tensor);
