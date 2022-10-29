/*
 * torch-gobject/nn/options/torch-nn-any-module.h
 *
 * Base class for all Torch NN modules
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
#include <torch-gobject/nn/torch-nn-module-base.h>

G_BEGIN_DECLS

#define TORCH_TYPE_NN_ANY_MODULE torch_nn_any_module_get_type ()
G_DECLARE_FINAL_TYPE (TorchNNAnyModule, torch_nn_any_module, TORCH, NN_ANY_MODULE, TorchNNModuleBase)

G_END_DECLS