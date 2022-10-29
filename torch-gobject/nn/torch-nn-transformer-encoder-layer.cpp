/*
 * torch-gobject/torch-nn-transformer-encoder-layer.cpp
 *
 * Base class for the TransformerEncoderLayer.
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

#include <torch-gobject/nn/torch-nn-any-module.h>
#include <torch-gobject/nn/torch-nn-any-module-internal.h>
#include <torch-gobject/nn/torch-nn-any-module-castable.h>
#include <torch-gobject/nn/torch-nn-module-base.h>
#include <torch-gobject/nn/torch-nn-transformer-encoder-layer.h>
#include <torch-gobject/nn/torch-nn-transformer-encoder-layer-internal.h>

#include <torch/torch.h>

struct _TorchNNTransformerEncoderLayer
{
  TorchNNModuleBase parent_instance;
};

typedef struct _TorchNNTransformerEncoderLayerPrivate
{
  torch::nn::TransformerEncoderLayer *internal;
} TorchNNTransformerEncoderLayerPrivate;

static void torch_nn_transformer_encoder_layer_nn_any_module_castable_interface_init (TorchNNAnyModuleCastableInterface *iface);

G_DEFINE_TYPE_WITH_CODE (TorchNNTransformerEncoderLayer, torch_nn_transformer_encoder_layer, TORCH_TYPE_NN_MODULE_BASE,
                         G_ADD_PRIVATE (TorchNNTransformerEncoderLayer)
                         G_IMPLEMENT_INTERFACE (TORCH_TYPE_NN_ANY_MODULE_CASTABLE,
                                                torch_nn_transformer_encoder_layer_nn_any_module_castable_interface_init))

#define TORCH_NN_TRANSFORMER_ENCODER_LAYER_GET_PRIVATE(x) static_cast <TorchNNTransformerEncoderLayerPrivate *> (torch_nn_transformer_encoder_layer_get_instance_private ((x)))

torch::nn::TransformerEncoderLayer &
torch_nn_transformer_encoder_layer_to_real_transformer_encoder_layer (TorchNNTransformerEncoderLayer *nn_transformer_encoder_layer)
{
  TorchNNTransformerEncoderLayerPrivate *priv = TORCH_NN_TRANSFORMER_ENCODER_LAYER_GET_PRIVATE (nn_transformer_encoder_layer);

  return *priv->internal;
}

TorchNNTransformerEncoderLayer *
torch_nn_transformer_encoder_layer_new_from_real_transformer_encoder_layer (torch::nn::TransformerEncoderLayer const &real_transformer_encoder_layer)
{
  TorchNNTransformerEncoderLayer *mod = torch_nn_transformer_encoder_layer_new ();
  TorchNNTransformerEncoderLayerPrivate *priv = TORCH_NN_TRANSFORMER_ENCODER_LAYER_GET_PRIVATE (mod);

  priv->internal = new torch::nn::TransformerEncoderLayer (real_transformer_encoder_layer);

  return mod;
}

TorchNNAnyModule *
torch_nn_transformer_encoder_layer_convert (TorchNNAnyModuleCastable  *castable,
                                            GError                   **error)
{
  TorchNNTransformerEncoderLayer *encoder_layer = TORCH_NN_TRANSFORMER_ENCODER_LAYER (castable);

  return torch_nn_any_module_convert_from_real_module (torch_nn_transformer_encoder_layer_to_real_transformer_encoder_layer (encoder_layer));
}

static void
torch_nn_transformer_encoder_layer_init (TorchNNTransformerEncoderLayer *nn_transformer_encoder_layer)
{
  TorchNNTransformerEncoderLayerPrivate *priv = TORCH_NN_TRANSFORMER_ENCODER_LAYER_GET_PRIVATE (nn_transformer_encoder_layer);
  priv->internal = nullptr;
}

static void
torch_nn_transformer_encoder_layer_finalize (GObject *object)
{
  TorchNNTransformerEncoderLayer *nn_transformer_encoder_layer = TORCH_NN_TRANSFORMER_ENCODER_LAYER (object);
  TorchNNTransformerEncoderLayerPrivate *priv = TORCH_NN_TRANSFORMER_ENCODER_LAYER_GET_PRIVATE (nn_transformer_encoder_layer);

  if (priv->internal)
    {
      delete priv->internal;
      priv->internal = nullptr;
    }
}

static void
torch_nn_transformer_encoder_layer_nn_any_module_castable_interface_init (TorchNNAnyModuleCastableInterface *iface)
{
  iface->convert = torch_nn_transformer_encoder_layer_convert;
}

static void
torch_nn_transformer_encoder_layer_class_init (TorchNNTransformerEncoderLayerClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = torch_nn_transformer_encoder_layer_finalize;
}

TorchNNTransformerEncoderLayer *
torch_nn_transformer_encoder_layer_new (void)
{
  return static_cast<TorchNNTransformerEncoderLayer *> (g_object_new (TORCH_TYPE_NN_TRANSFORMER_ENCODER_LAYER, NULL));
}
