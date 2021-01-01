/*
 * torch-gobject/torch-tensor-options-internal.h
 *
 * Helper class to store options for tensor creation, internal functions.
 *
 * Copyright (C) 2020 Sam Spilsbury.
 *
 * torch-gobject is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * torch-gobject is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with torch-gobject; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#pragma once

#include <torch-gobject/torch-tensor-options.h>

#include <torch/torch.h>

c10::TensorOptions & torch_tensor_options_get_real_tensor_options (TorchTensorOptions *tensor_options);

TorchTensorOptions * torch_tensor_options_new_from_real_tensor_options (c10::TensorOptions const &tensor_options);
