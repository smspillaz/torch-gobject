/*
 * torch-gobject/torch-nn-any-module-castable.cpp
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

#include <torch-gobject/nn/torch-nn-any-module-internal.h>
#include <torch-gobject/nn/torch-nn-any-module-castable.h>
#include <torch-gobject/torch-util.h>

G_DEFINE_INTERFACE (TorchNNAnyModuleCastable, torch_nn_any_module_castable, G_TYPE_OBJECT)

static void
torch_nn_any_module_castable_default_init (TorchNNAnyModuleCastableInterface *castable_iface)
{
}

/**
 * torch_nn_any_module_castable_convert:
 * @castable: (transfer none): A #TorchNNAnyModuleCastable instance.
 * @error: A #GError which will be set if there is an error with casting.
 *
 * Try to cast the instance of #TorchNNAnyModuleCastable to a #TorchNNAnyModule
 *
 * Returns: (transfer full): A #TorchNNAnyModule wrapping the underlying module
 *                           of this instance.
 */
TorchNNAnyModule *
torch_nn_any_module_castable_convert (TorchNNAnyModuleCastable  *castable,
                                      GError                   **error)
{
  TorchNNAnyModuleCastableInterface *iface = TORCH_NN_ANY_MODULE_CASTABLE_GET_IFACE (castable);

  return iface->convert (castable, error);
}

torch::nn::AnyModule
torch_nn_any_module_castable_to_real_any_module (TorchNNAnyModuleCastable *castable)
{
  g_autoptr(GError) error = nullptr;
  g_autoptr(TorchNNAnyModule) any_module = torch_nn_any_module_castable_convert (castable, &error);

  if (error != nullptr)
    torch_throw_error (error);

  return torch_nn_any_module_to_real_any_module (any_module);
}