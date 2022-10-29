/*
 * torch-gobject/torch-nn-any-module.cpp
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
#include <torch-gobject/nn/torch-nn-any-module.h>
#include <torch-gobject/nn/torch-nn-any-module-internal.h>

#include <torch/torch.h>

struct _TorchNNAnyModule
{
  TorchNNModuleBase parent_instance;
};

typedef struct _TorchNNAnyModulePrivate
{
  /* We only support keeping the type-erased module with runtime checks */
  torch::nn::AnyModule *internal;
} TorchNNAnyModulePrivate;

G_DEFINE_TYPE_WITH_PRIVATE (TorchNNAnyModule, torch_nn_any_module, TORCH_TYPE_NN_MODULE_BASE)
#define TORCH_NN_ANY_MODULE_GET_PRIVATE(x) static_cast <TorchNNAnyModulePrivate *> (torch_nn_any_module_get_instance_private ((x)))

torch::nn::AnyModule const &
torch_nn_any_module_to_real_any_module (TorchNNAnyModule *nn_module)
{
  TorchNNAnyModulePrivate *priv = TORCH_NN_ANY_MODULE_GET_PRIVATE (nn_module);
  return *priv->internal;
}

TorchNNAnyModule *
torch_nn_any_module_new_from_real_any_module (torch::nn::AnyModule const &real_module)
{
  TorchNNAnyModule *mod = static_cast <TorchNNAnyModule *> (g_object_new (TORCH_TYPE_NN_ANY_MODULE, nullptr));
  TorchNNAnyModulePrivate *priv = TORCH_NN_ANY_MODULE_GET_PRIVATE (mod);

  priv->internal = new torch::nn::AnyModule (real_module);

  return mod;
}

static void
torch_nn_any_module_init (TorchNNAnyModule *nn_module)
{
  TorchNNAnyModulePrivate *priv = TORCH_NN_ANY_MODULE_GET_PRIVATE (nn_module);
  priv->internal = nullptr;
}

static void
torch_nn_any_module_finalize (GObject *object)
{
  TorchNNAnyModule *nn_module = TORCH_NN_ANY_MODULE (object);
  TorchNNAnyModulePrivate *priv = TORCH_NN_ANY_MODULE_GET_PRIVATE (nn_module);

  if (priv->internal)
    {
      delete priv->internal;
      priv->internal = nullptr;
    }
}

static void
torch_nn_any_module_class_init (TorchNNAnyModuleClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = torch_nn_any_module_finalize;
}

TorchNNAnyModule *
torch_nn_any_module_new (void)
{
  return static_cast<TorchNNAnyModule *> (g_object_new (TORCH_TYPE_NN_ANY_MODULE, NULL));
}
