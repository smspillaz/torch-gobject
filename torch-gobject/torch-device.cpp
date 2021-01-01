/*
 * torch-gobject/torch-device.cpp
 *
 * Device abstraction for creating device.
 *
 * Copyright (C) 2020 Sam Spilsbury.
 *
 * torch-gobject is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * torch-gobject is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with eos-companion-app-service.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

#include <gio/gio.h>

#include <torch-gobject/torch-device.h>
#include <torch-gobject/torch-device-internal.h>
#include <torch-gobject/torch-device-type.h>
#include <torch-gobject/torch-device-type-internal.h>
#include <torch-gobject/torch-util.h>

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

static void initable_iface_init (GInitableIface *iface);

G_DEFINE_TYPE_WITH_CODE (TorchDevice, torch_device, G_TYPE_OBJECT,
                         G_ADD_PRIVATE (TorchDevice)
                         G_IMPLEMENT_INTERFACE (G_TYPE_INITABLE, initable_iface_init))
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
  g_autoptr (GError)    error = NULL;

  if (!torch_device_init_internal (device, static_cast <GError **> (&error)))
    torch_throw_error (error);

  return *priv->internal;
}

/**
 * torch_device_get_index:
 * @device: A #TorchDevice
 * @out_index: (out): A short indicating the which device (if the system has many devices)
 * @error: A #GError
 *
 * Get the index of the device
 *
 * Returns: %TRUE with @out_index set on success, %FALSE on failure.
 */
gboolean
torch_device_get_index (TorchDevice  *device,
                        short        *out_index,
                        GError      **error)
{
  TorchDevicePrivate *priv = TORCH_DEVICE_GET_PRIVATE (device);

  if (!torch_device_init_internal (device, error))
    return FALSE;

  try
    {
      *out_index = static_cast <short> (priv->internal->index ());
      return TRUE;
    }
  catch (const std::exception &e)
    {
      g_set_error (error,
                   G_IO_ERROR,
                   G_IO_ERROR_FAILED,
                   e.what ());
    }

  return FALSE;
}

/**
 * torch_device_get_device_type:
 * @device: A #TorchDevice
 * @out_device_type: (out): The #TorchDeviceType for this device
 * @error: A #GError
 *
 * Get the #TorchDeviceType of the device
 *
 * Returns: %TRUE with @out_index set on success, %FALSE on failure.
 */
gboolean
torch_device_get_device_type (TorchDevice      *device,
                              TorchDeviceType  *out_device_type,
                              GError          **error)
{
  TorchDevicePrivate *priv = TORCH_DEVICE_GET_PRIVATE (device);

  if (!torch_device_init_internal (device, error))
    return FALSE;

  try
    {
      *out_device_type = static_cast <TorchDeviceType> (
        torch_device_type_from_real_device_type (priv->internal->type ()
      ));
      return TRUE;
    }
  catch (const std::exception &e)
    {
      g_set_error (error,
                   G_IO_ERROR,
                   G_IO_ERROR_FAILED,
                   e.what ());
    }

  return FALSE;

}

/**
 * torch_device_get_string:
 * @device: A #TorchDevice
 * @error: A #GError
 *
 * Get a description of the device
 *
 * Returns: (transfer full): A string describing the device or %NULL
 *                           with @error set on failure.
 */
char *
torch_device_get_string (TorchDevice  *device,
                         GError      **error)
{
  TorchDevicePrivate *priv = TORCH_DEVICE_GET_PRIVATE (device);

  if (!torch_device_init_internal (device, error))
    return NULL;

  try
    {
      auto str (priv->internal->str ());

      return g_strdup (str.c_str ());
    }
  catch (const std::exception &e)
    {
      g_set_error (error,
                   G_IO_ERROR,
                   G_IO_ERROR_FAILED,
                   e.what ());
    }

  return NULL;
}

gboolean
torch_device_init_internal (TorchDevice  *device,
                            GError       **error)
{
  TorchDevicePrivate *priv = TORCH_DEVICE_GET_PRIVATE (device);

  g_return_val_if_fail (error == NULL || *error == NULL, FALSE);

  /* Even though we have a check in torch_device_initable_init,
   * check again here to avoid the vfunc calls */
  if (!priv->internal)
    return g_initable_init (G_INITABLE (device), NULL, error);

  return TRUE;
}

static gboolean
torch_device_initable_init (GInitable     *initable,
                            GCancellable  *cancellable,
                            GError       **error)
{
  TorchDevice        *device = TORCH_DEVICE (initable);
  TorchDevicePrivate *priv = TORCH_DEVICE_GET_PRIVATE (device);

  if (priv->internal)
    return TRUE;

  try
    {
      if (priv->construction_string != NULL)
        {
          priv->internal = new c10::Device (std::string (priv->construction_string));
        }

      else
        {
          priv->internal = new c10::Device (torch_device_type_get_real_device_type (priv->construction_device_type),
                                            priv->construction_device_index);
        }

      /* Once construction is complete, we clear the construction properties */
      g_clear_pointer (&priv->construction_string, g_free);
      priv->construction_device_index = -1;
      priv->construction_device_type = TORCH_DEVICE_TYPE_CPU;
    }
  catch (const std::exception &exp)
    {
      g_set_error (error,
                   G_IO_ERROR,
                   G_IO_ERROR_FAILED,
                   exp.what(),
                   nullptr);
      return FALSE;
    }

  return TRUE;
}

static void
initable_iface_init (GInitableIface *iface)
{
  iface->init = torch_device_initable_init;
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
torch_device_get_property (GObject      *object,
                           unsigned int  prop_id,
                           GValue       *value,
                           GParamSpec   *pspec)
{
  TorchDevice *device = TORCH_DEVICE (object);

  switch (prop_id)
    {
      case PROP_DEVICE_TYPE:
        g_value_set_enum (value,
                          call_and_warn_about_gerror ("get prop 'device-type'",
                                                      [](TorchDevice  *device,
                                                         GError       **error) -> gboolean {
                                                        TorchDeviceType device_type;
                                                        torch_device_get_device_type (device,
                                                                                      &device_type,
                                                                                      error);
                                                        return device_type;
                                                      },
                                                      device));
        break;
      case PROP_DEVICE_INDEX:
        g_value_set_int (value,
                         call_and_warn_about_gerror ("get prop 'index'",
                                                     [](TorchDevice  *device,
                                                        GError       **error) -> gboolean {
                                                       short index;
                                                       torch_device_get_index (device,
                                                                               &index,
                                                                               error);
                                                       return index;
                                                     },
                                                     device));
        break;
      case PROP_STRING:
        g_value_set_string (value,
                            call_and_warn_about_gerror ("get prop 'string'",
                                                        torch_device_get_string,
                                                        device));
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
        priv->construction_device_index = g_value_get_int (value);
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

  /**
   * TorchDevice::string: (transfer full)
   *
   * A string describing the device.
   */
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

/**
 * torch_device_new:
 * @error: A #GError
 *
 * Create a new #TorchDevice with no properties set. In practice,
 * this means that the device will be created with %TORCH_DEVICE_TYPE_CPU
 * and on index 0.
 *
 * Returns: (transfer full): A new #TorchDevice of %TORCH_DEVICE_TYPE_CPU
 *                           or %NULL with @error set on failure.
 */
TorchDevice *
torch_device_new (GError **error)
{
  return static_cast<TorchDevice *> (g_initable_new (TORCH_TYPE_DEVICE,
                                                     NULL,
                                                     error,
                                                     NULL));
}

/**
 * torch_device_new_from_string:
 * @string: A string describing the device
 * @error: A #GError
 *
 * Create a new #TorchDevice matching the description in @string.
 *
 * Returns: (transfer full): A new #TorchDevice matching the description
 *                           in @string or %NULL with @error set on failure.
 */
TorchDevice *
torch_device_new_from_string (const char  *string,
                              GError     **error)
{
  return static_cast<TorchDevice *> (g_initable_new (TORCH_TYPE_DEVICE,
                                                     NULL,
                                                     error,
                                                     "string", string,
                                                     NULL));
}

/**
 * torch_device_new_from_type_index:
 * @type: A #TorchDevicetype
 * @index: A short indicating which index in an array of compute devices.
 * @error: A #GError
 *
 * Create a new #TorchDevice matching the description given by
 * @type and @index
 *
 * Returns: (transfer full): A new #TorchDevice matching the description
 *                           in @type and @index or %NULL with @error set on failure.
 */
TorchDevice *
torch_device_new_from_type_index (TorchDeviceType   type,
                                  short             index,
                                  GError          **error)
{
  return static_cast<TorchDevice *> (g_initable_new (TORCH_TYPE_DEVICE,
                                                     NULL,
                                                     error,
                                                     "type", type,
                                                     "index", index,
                                                     NULL));
}

TorchDevice *
torch_device_new_from_real_device (c10::Device const &real_device)
{
  g_autoptr (TorchDevice) device = static_cast <TorchDevice *> (g_object_new (TORCH_TYPE_DEVICE, NULL));
  TorchDevicePrivate *priv = TORCH_DEVICE_GET_PRIVATE (device);

  /* This should not throw, since we were able to construct real_device */
  priv->internal = new c10::Device (real_device.type (), real_device.index ());

  return static_cast <TorchDevice *> (g_steal_pointer (&device));
}
