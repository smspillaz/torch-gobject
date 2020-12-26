/*
 * torch-gobject/torch-tensor-options.cpp
 *
 * Helper class to store options for tensor creation.
 *
 * Copyright (C) 2020 Sam Spilsbury.
 *
 * libanimation is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * libanimation is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with eos-companion-app-service.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

#include <torch-gobject/torch-device.h>
#include <torch-gobject/torch-device-internal.h>
#include <torch-gobject/torch-layout.h>
#include <torch-gobject/torch-layout-internal.h>
#include <torch-gobject/torch-memory-format.h>
#include <torch-gobject/torch-memory-format-internal.h>
#include <torch-gobject/torch-tensor-options.h>
#include <torch-gobject/torch-tensor-options-internal.h>
#include <torch-gobject/torch-util.h>

#include <c10/core/TensorOptions.h>

#include "torch-enums.h"

struct _TorchTensorOptions
{
  GObject parent_instance;
};

typedef struct _TorchTensorOptionsPrivate
{
  c10::TensorOptions     *internal;
} TorchTensorOptionsPrivate;

G_DEFINE_TYPE_WITH_PRIVATE (TorchTensorOptions, torch_tensor_options, G_TYPE_OBJECT)
#define TORCH_TENSOR_OPTIONS_GET_PRIVATE(a) static_cast <TorchTensorOptionsPrivate *> (torch_tensor_options_get_instance_private ((a)))

enum {
  PROP_0,
  PROP_DEVICE,
  PROP_DTYPE,
  PROP_LAYOUT,
  PROP_MEMORY_FORMAT,
  PROP_REQUIRES_GRAD,
  PROP_PINNED_MEMORY,
  NPROPS
};

static GParamSpec *torch_tensor_options_props [NPROPS] = { NULL, };

c10::TensorOptions &
torch_tensor_options_get_real_tensor_options (TorchTensorOptions *tensor_options)
{
  TorchTensorOptionsPrivate *priv = TORCH_TENSOR_OPTIONS_GET_PRIVATE (tensor_options);

  return *priv->internal;
}

static void
torch_tensor_options_set_device (TorchTensorOptions *options,
                                 TorchDevice        *device)
{
  TorchTensorOptionsPrivate *priv = TORCH_TENSOR_OPTIONS_GET_PRIVATE (options);

  if (device != NULL)
    *priv->internal = priv->internal->device (torch_device_get_real_device (device));
  else
    *priv->internal = priv->internal->device ("cpu");
}

static void
torch_tensor_options_init (TorchTensorOptions *tensor_options)
{
  TorchTensorOptionsPrivate *priv = TORCH_TENSOR_OPTIONS_GET_PRIVATE (tensor_options);
  priv->internal = new c10::TensorOptions ();
}

static void
torch_tensor_options_set_property (GObject      *object,
                                   unsigned int  prop_id,
                                   const GValue *value,
                                   GParamSpec   *pspec)
{
  TorchTensorOptions *tensor_options = TORCH_TENSOR_OPTIONS (object);
  TorchTensorOptionsPrivate *priv = TORCH_TENSOR_OPTIONS_GET_PRIVATE (tensor_options);

  /* Properties only get set on construction.
   *
   * The replacement of priv->internal looks a bit strange
   * but this is because libtorch only exposes a chaining-API
   * for constructing TensorOptions (eg, TensorOptions is not
   * normally mutable).
   */
  switch (prop_id)
    {
      case PROP_DEVICE:
        torch_tensor_options_set_device (tensor_options,
                                         TORCH_DEVICE (g_value_get_object (value)));
        break;
      case PROP_DTYPE:
        *priv->internal = priv->internal->dtype (torch_scalar_type_from_gtype (g_value_get_gtype (value)));
        break;
      case PROP_LAYOUT:
        *priv->internal = priv->internal->layout (torch_layout_get_real_layout (static_cast <TorchLayout> (g_value_get_enum (value))));
        break;
      case PROP_MEMORY_FORMAT:
        *priv->internal = priv->internal->memory_format (torch_memory_format_get_real_memory_format (static_cast <TorchMemoryFormat> (g_value_get_enum (value))));
        break;
      case PROP_REQUIRES_GRAD:
        *priv->internal = priv->internal->requires_grad (g_value_get_boolean (value));
        break;
      case PROP_PINNED_MEMORY:
        *priv->internal = priv->internal->pinned_memory (g_value_get_boolean (value));
        break;
      default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
        break;
    }
}

static void
torch_tensor_options_finalize (GObject *object)
{
  TorchTensorOptions *tensor_options = TORCH_TENSOR_OPTIONS (object);
  TorchTensorOptionsPrivate *priv = TORCH_TENSOR_OPTIONS_GET_PRIVATE (tensor_options);

  if (priv->internal)
    {
      delete priv->internal;
      priv->internal = nullptr;
    }
}

static void
torch_tensor_options_class_init (TorchTensorOptionsClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->set_property = torch_tensor_options_set_property;
  object_class->finalize = torch_tensor_options_finalize;


  torch_tensor_options_props[PROP_DEVICE] =
    g_param_spec_object ("device",
                         "Type",
                         "Device Type",
                         TORCH_TYPE_DEVICE,
                         static_cast <GParamFlags> (G_PARAM_WRITABLE | G_PARAM_CONSTRUCT_ONLY));

  torch_tensor_options_props[PROP_DTYPE] =
    g_param_spec_gtype ("dtype",
                        "DType",
                        "Data Type",
                        G_TYPE_NONE,
                        static_cast <GParamFlags> (G_PARAM_WRITABLE | G_PARAM_CONSTRUCT_ONLY));

  torch_tensor_options_props[PROP_LAYOUT] =
    g_param_spec_enum ("layout",
                       "Layout",
                       "Data Layout",
                       TORCH_TYPE_LAYOUT,
                       TORCH_LAYOUT_STRIDED,
                       static_cast <GParamFlags> (G_PARAM_WRITABLE | G_PARAM_CONSTRUCT_ONLY));

  torch_tensor_options_props[PROP_MEMORY_FORMAT] =
    g_param_spec_enum ("memory-format",
                       "Memory Format",
                       "Memory Format",
                       TORCH_TYPE_MEMORY_FORMAT,
                       TORCH_MEMORY_FORMAT_PRESERVE,
                       static_cast <GParamFlags> (G_PARAM_WRITABLE | G_PARAM_CONSTRUCT_ONLY));

  torch_tensor_options_props[PROP_REQUIRES_GRAD] =
    g_param_spec_boolean ("requires-grad",
                          "Requires Grad",
                          "Requires a gradient",
                          FALSE,
                          static_cast <GParamFlags> (G_PARAM_WRITABLE | G_PARAM_CONSTRUCT_ONLY));

  torch_tensor_options_props[PROP_PINNED_MEMORY] =
    g_param_spec_boolean ("pinned-memory",
                          "Pinned Memory",
                          "Pinned Memory",
                          FALSE,
                          static_cast <GParamFlags> (G_PARAM_WRITABLE | G_PARAM_CONSTRUCT_ONLY));

  g_object_class_install_properties (object_class,
                                     NPROPS,
                                     torch_tensor_options_props);
}

TorchTensorOptions *
torch_tensor_options_new (void)
{
  return static_cast<TorchTensorOptions *> (g_object_new (TORCH_TYPE_TENSOR_OPTIONS, NULL));
}

TorchTensorOptions *
torch_tensor_options_new_from_real_tensor_options (c10::TensorOptions const &real_tensor_options)
{
  g_autoptr (TorchTensorOptions) tensor_options = torch_tensor_options_new ();
  TorchTensorOptionsPrivate *priv = TORCH_TENSOR_OPTIONS_GET_PRIVATE (tensor_options);

  delete priv->internal;
  priv->internal = new c10::TensorOptions (real_tensor_options);

  return static_cast <TorchTensorOptions *> (g_steal_pointer (&tensor_options));
}
