/*
 * torch-gobject/nn/options/torch-nn-module-base.h
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

G_BEGIN_DECLS

#define TORCH_TYPE_NN_MODULE_BASE torch_nn_module_base_get_type()
G_DECLARE_DERIVABLE_TYPE (TorchNNModuleBase, torch_nn_module_base, TORCH, NN_MODULE_BASE, GObject)

struct _TorchNNModuleBaseClass
{
  GObjectClass object_class;

  gpointer padding[16];
};

G_END_DECLS