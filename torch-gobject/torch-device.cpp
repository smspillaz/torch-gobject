/*
 * torch-gobject/torch-device.cpp
 *
 * Device abstraction for creating device.
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
#include <torch-gobject/torch-device-type.h>
#include <torch-gobject/torch-device-type-internal.h>

#include <c10/core/Device.h>

#include "torch-enums.h"

struct _TorchDevice
{
  GObject parent_instance;
};

typedef struct _TorchDevicePrivate
{
  c10::Device     *internal;
  gchar           *construction_string;
  TorchDeviceType  construction_device_type;
  int8_t           construction_device_index;
} TorchDevicePrivate;

G_DEFINE_TYPE_WITH_PRIVATE (TorchDevice, torch_device, G_TYPE_OBJECT)
#define TORCH_DEVICE_GET_PRIVATE(a) static_cast <TorchDevicePrivate *> (torch_device_get_instance_private ((a)))

enum {
  PROP_0,
  PROP_STRING,
  PROP_DEVICE_TYPE,
  PROP_DEVICE_INDEX,
  NPROPS
};

static GParamSpec *torch_device_props [NPROPS] = { NULL, };

c10::Device &
torch_device_get_real_device (TorchDevice *device)
{
  TorchDevicePrivate *priv = TORCH_DEVICE_GET_PRIVATE (device);

  return *priv->internal;
}

/**
 * torch_device_get_index:
 * @device: A #TorchDevice
 *
 * Get the index of the device
 *
 * Returns: The index of the device (if the system has many compute devices,
 *          which one this tensor lives on).
 */
short
torch_device_get_index (TorchDevice *device)
{
  TorchDevicePrivate *priv = TORCH_DEVICE_GET_PRIVATE (device);

  if (priv->internal != NULL)
    {
      return static_cast <short> (priv->internal->index ());
    }
  else
    {
      return priv->construction_device_index;
    }
}

/**
 * torch_device_get_device_type:
 * @device: A #TorchDevice
 *
 * Get the #TorchDeviceType of the device
 *
 * Returns: The #TorchDeviceType of the device.
 */
TorchDeviceType
torch_device_get_device_type (TorchDevice *device)
{
  TorchDevicePrivate *priv = TORCH_DEVICE_GET_PRIVATE (device);

  if (priv->internal != NULL)
    {
      return torch_device_type_from_real_device_type (priv->internal->type ());
    }

  return priv->construction_device_type;

}

/**
 * torch_device_get_string:
 * @device: A #TorchDevice
 *
 * Get a description of the device
 *
 * Returns: (transfer full): A string describing the device
 */
char *
torch_device_get_string (TorchDevice *device)
{
  TorchDevicePrivate *priv = TORCH_DEVICE_GET_PRIVATE (device);

  if (priv->internal != NULL)
    {
      auto str (priv->internal->str ());

      return g_strdup (str.c_str ());
    }

  return g_strdup (priv->construction_string);

}

static void
torch_device_init (TorchDevice *device)
{
  TorchDevicePrivate *priv = TORCH_DEVICE_GET_PRIVATE (device);
  priv->construction_string = NULL;
  priv->construction_device_type = TORCH_DEVICE_TYPE_CPU;
  priv->construction_device_index = -1;
}

static void
torch_device_constructed (GObject *object)
{
  TorchDevice        *device = TORCH_DEVICE (object);
  TorchDevicePrivate *priv = TORCH_DEVICE_GET_PRIVATE (device);

  if (priv->construction_string != NULL)
    {
      priv->internal = new c10::Device (std::string (priv->construction_string));
    }

  else
    {
      priv->internal = new c10::Device (torch_device_type_get_real_device_type (priv->construction_device_type),
                                        priv->construction_device_index);
    }

  g_clear_pointer (&priv->construction_string, g_free);
  priv->construction_device_index = -1;
  priv->construction_device_type = TORCH_DEVICE_TYPE_CPU;
}

static void
torch_device_get_property (GObject      *object,
                            unsigned int  prop_id,
                            GValue       *value,
                            GParamSpec   *pspec)
{
  TorchDevice *device = TORCH_DEVICE (object);

  switch (prop_id)
    {
      case PROP_DEVICE_TYPE:
        g_value_set_enum (value, torch_device_get_device_type (device));
        break;
      case PROP_DEVICE_INDEX:
        g_value_set_int (value, torch_device_get_index (device));
        break;
      case PROP_STRING:
        g_value_set_string (value, torch_device_get_string (device));
        break;
      default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
        break;
    }
}

static void
torch_device_set_property (GObject      *object,
                           unsigned int  prop_id,
                           const GValue *value,
                           GParamSpec   *pspec)
{
  TorchDevice *device = TORCH_DEVICE (object);
  TorchDevicePrivate *priv = TORCH_DEVICE_GET_PRIVATE (device);

  /* Properties only get set on construction */
  switch (prop_id)
    {
      case PROP_DEVICE_TYPE:
        priv->construction_device_type = static_cast <TorchDeviceType> (g_value_get_enum (value));
        break;
      case PROP_DEVICE_INDEX:
        priv->construction_device_index = g_value_get_uint64 (value);
        break;
      case PROP_STRING:
        priv->construction_string = g_value_dup_string (value);
        break;
      default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
        break;
    }
}

static void
torch_device_finalize (GObject *object)
{
  TorchDevice *device = TORCH_DEVICE (object);
  TorchDevicePrivate *priv = TORCH_DEVICE_GET_PRIVATE (device);

  g_clear_pointer (&priv->construction_string, g_free);

  if (priv->internal)
    {
      delete priv->internal;
      priv->internal = nullptr;
    }
}

static void
torch_device_class_init (TorchDeviceClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->constructed = torch_device_constructed;
  object_class->get_property = torch_device_get_property;
  object_class->set_property = torch_device_set_property;
  object_class->finalize = torch_device_finalize;

  torch_device_props[PROP_DEVICE_TYPE] =
    g_param_spec_enum ("type",
                       "Type",
                       "Device Type",
                       TORCH_TYPE_DEVICE_TYPE,
                       TORCH_DEVICE_TYPE_CPU,
                       static_cast <GParamFlags> (G_PARAM_READWRITE | G_PARAM_CONSTRUCT_ONLY));

  torch_device_props[PROP_DEVICE_INDEX] =
    g_param_spec_int ("index",
                      "Index",
                      "Device Index",
                      G_MINSHORT,
                      G_MAXSHORT,
                      0,
                      static_cast <GParamFlags> (G_PARAM_READWRITE | G_PARAM_CONSTRUCT_ONLY));

  torch_device_props[PROP_STRING] =
    g_param_spec_string ("string",
                         "String",
                         "String describing the device",
                         "cpu",
                         static_cast <GParamFlags> (G_PARAM_READWRITE | G_PARAM_CONSTRUCT_ONLY));


  g_object_class_install_properties (object_class,
                                     NPROPS,
                                     torch_device_props);
}

TorchDevice *
torch_device_new (void)
{
  return static_cast<TorchDevice *> (g_object_new (TORCH_TYPE_DEVICE, NULL));
}

TorchDevice *
torch_device_new_from_string (const char *string)
{
  return static_cast<TorchDevice *> (g_object_new (TORCH_TYPE_DEVICE, "string", string, NULL));
}

TorchDevice *
torch_device_new_from_type_index (TorchDeviceType type,
                                  short           index)
{
  return static_cast<TorchDevice *> (g_object_new (TORCH_TYPE_DEVICE, "type", type, "index", index, NULL));
}

TorchDevice *
torch_device_new_from_real_device (c10::Device const &real_device)
{
  g_autoptr (TorchDevice) device = torch_device_new ();
  TorchDevicePrivate *priv = TORCH_DEVICE_GET_PRIVATE (device);

  delete priv->internal;
  priv->internal = new c10::Device (real_device.type (), real_device.index ());

  return static_cast <TorchDevice *> (g_steal_pointer (&device));
}
