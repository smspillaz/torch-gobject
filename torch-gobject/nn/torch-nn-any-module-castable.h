/*
 * torch-gobject/torch-nn-any-module-castable.h
 *
 * An NNModule that can be casted to an NNAnyModule.
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
#include <torch-gobject/nn/torch-nn-any-module.h>

G_BEGIN_DECLS

#define TORCH_TYPE_NN_ANY_MODULE_CASTABLE torch_nn_any_module_castable_get_type()
G_DECLARE_INTERFACE (TorchNNAnyModuleCastable, torch_nn_any_module_castable, TORCH, NN_ANY_MODULE_CASTABLE, GObject)

struct _TorchNNAnyModuleCastableInterface
{
  GTypeInterface parent;

  TorchNNAnyModule * (*convert) (TorchNNAnyModuleCastable *castable, GError **error);
};

TorchNNAnyModule * torch_nn_any_module_castable_convert (TorchNNAnyModuleCastable  *castable,
                                                         GError                   **error);

G_END_DECLS