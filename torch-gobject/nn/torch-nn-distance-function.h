/*
 * torch-gobject/nn/torch-nn-distance-function.h
 *
 * Definition for TorchNNDistanceFunction callback
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

#include <glib.h>
#include <torch-gobject/torch-tensor.h>

/**
 * TorchNNDistanceFunction:
 * @first: (transfer none): The first #TorchTensor in the comparison operation
 * @second: (transfer none): The second #TorchTensor in the comparison operation
 *
 * Returns: (transfer full): A #TorchTensor containing the result of the distance function comparison,
 *                           which should be one dimension less than the operands.
 */
typedef TorchTensor * (*TorchNNDistanceFunction) (TorchTensor const *first, TorchTensor const *second, gpointer data);
