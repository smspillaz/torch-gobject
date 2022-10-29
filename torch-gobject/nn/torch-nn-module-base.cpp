/*
 * torch-gobject/torch-nn-module-base.cpp
 *
 * NNModule abstraction for creating storage.
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

#include <torch-gobject/nn/torch-nn-module-base.h>

typedef struct _TorchNNModuleBasePrivate
{
} TorchNNModuleBasePrivate;

G_DEFINE_TYPE_WITH_PRIVATE (TorchNNModuleBase, torch_nn_module_base, G_TYPE_OBJECT)
#define TORCH_NN_MODULE_BASE_GET_PRIVATE(x) static_cast <TorchNNModuleBasePrivate *> (torch_nn_module_base_get_instance_private ((x)))

static void
torch_nn_module_base_init (TorchNNModuleBase *nn_module)
{
}

static void
torch_nn_module_base_finalize (GObject *object)
{
}

static void
torch_nn_module_base_class_init (TorchNNModuleBaseClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = torch_nn_module_base_finalize;
}
