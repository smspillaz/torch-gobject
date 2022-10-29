/*
 * torch-gobject/torch-nn-any-module-castable-internal.h
 *
 * An NNModule that can be casted to an NNAnyModule, internal functionality.
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

#include <glib-object.h>
#include <torch-gobject/nn/torch-nn-any-module-internal.h>
#include <torch-gobject/nn/torch-nn-any-module-castable.h>

torch::nn::AnyModule torch_nn_any_module_castable_to_real_any_module (TorchNNAnyModuleCastable *castable);