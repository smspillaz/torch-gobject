/*
 * torch-gobject/torch-nn-transformer-decoder-layer.cpp
 *
 * Base class for the TransformerDecoderLayer.
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
#include <torch-gobject/nn/torch-nn-transformer-decoder-layer.h>
#include <torch-gobject/nn/torch-nn-transformer-decoder-layer-internal.h>

#include <torch/torch.h>

struct _TorchNNTransformerDecoderLayer
{
  TorchNNModuleBase parent_instance;
};

typedef struct _TorchNNTransformerDecoderLayerPrivate
{
  torch::nn::TransformerDecoderLayer *internal;
} TorchNNTransformerDecoderLayerPrivate;

static void torch_nn_transformer_decoder_layer_nn_any_module_castable_interface_init (TorchNNAnyModuleCastableInterface *iface);

G_DEFINE_TYPE_WITH_CODE (TorchNNTransformerDecoderLayer, torch_nn_transformer_decoder_layer, TORCH_TYPE_NN_MODULE_BASE,
                         G_ADD_PRIVATE (TorchNNTransformerDecoderLayer)
                         G_IMPLEMENT_INTERFACE (TORCH_TYPE_NN_ANY_MODULE_CASTABLE,
                                                torch_nn_transformer_decoder_layer_nn_any_module_castable_interface_init))

#define TORCH_NN_TRANSFORMER_DECODER_LAYER_GET_PRIVATE(x) static_cast <TorchNNTransformerDecoderLayerPrivate *> (torch_nn_transformer_decoder_layer_get_instance_private ((x)))

torch::nn::TransformerDecoderLayer &
torch_nn_transformer_decoder_layer_to_real_transformer_decoder_layer (TorchNNTransformerDecoderLayer *nn_transformer_decoder_layer)
{
  TorchNNTransformerDecoderLayerPrivate *priv = TORCH_NN_TRANSFORMER_DECODER_LAYER_GET_PRIVATE (nn_transformer_decoder_layer);

  return *priv->internal;
}

TorchNNTransformerDecoderLayer *
torch_nn_transformer_decoder_layer_new_from_real_transformer_decoder_layer (torch::nn::TransformerDecoderLayer const &real_transformer_decoder_layer)
{
  TorchNNTransformerDecoderLayer *mod = torch_nn_transformer_decoder_layer_new ();
  TorchNNTransformerDecoderLayerPrivate *priv = TORCH_NN_TRANSFORMER_DECODER_LAYER_GET_PRIVATE (mod);

  priv->internal = new torch::nn::TransformerDecoderLayer (real_transformer_decoder_layer);

  return mod;
}

TorchNNAnyModule *
torch_nn_transformer_decoder_layer_convert (TorchNNAnyModuleCastable  *castable,
                                            GError                   **error)
{
  TorchNNTransformerDecoderLayer *decoder_layer = TORCH_NN_TRANSFORMER_DECODER_LAYER (castable);

  return torch_nn_any_module_convert_from_real_module (torch_nn_transformer_decoder_layer_to_real_transformer_decoder_layer (decoder_layer));
}

static void
torch_nn_transformer_decoder_layer_init (TorchNNTransformerDecoderLayer *nn_transformer_decoder_layer)
{
  TorchNNTransformerDecoderLayerPrivate *priv = TORCH_NN_TRANSFORMER_DECODER_LAYER_GET_PRIVATE (nn_transformer_decoder_layer);
  priv->internal = nullptr;
}

static void
torch_nn_transformer_decoder_layer_finalize (GObject *object)
{
  TorchNNTransformerDecoderLayer *nn_transformer_decoder_layer = TORCH_NN_TRANSFORMER_DECODER_LAYER (object);
  TorchNNTransformerDecoderLayerPrivate *priv = TORCH_NN_TRANSFORMER_DECODER_LAYER_GET_PRIVATE (nn_transformer_decoder_layer);

  if (priv->internal)
    {
      delete priv->internal;
      priv->internal = nullptr;
    }
}

static void
torch_nn_transformer_decoder_layer_nn_any_module_castable_interface_init (TorchNNAnyModuleCastableInterface *iface)
{
  iface->convert = torch_nn_transformer_decoder_layer_convert;
}

static void
torch_nn_transformer_decoder_layer_class_init (TorchNNTransformerDecoderLayerClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = torch_nn_transformer_decoder_layer_finalize;
}

TorchNNTransformerDecoderLayer *
torch_nn_transformer_decoder_layer_new (void)
{
  return static_cast<TorchNNTransformerDecoderLayer *> (g_object_new (TORCH_TYPE_NN_TRANSFORMER_DECODER_LAYER, NULL));
}
