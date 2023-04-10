/*
 * torch-gobject/torch-tensor-index-internal.h
 *
 * Object representing a slice.
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

#include <torch-gobject/torch-tensor-index.h>

#include <torch/torch.h>

torch::indexing::TensorIndex torch_index_get_real_index (TorchIndex *index);

TorchIndex * torch_index_new_from_real_index (torch::indexing::TensorIndex const &real_index);
