/*
 * torch-gobject/torch-generator.cpp
 *
 * Generator abstraction for creating RNGs.
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

#include <cstdint>

#include <gio/gio.h>

#include <torch-gobject/torch-device.h>
#include <torch-gobject/torch-device-internal.h>
#include <torch-gobject/torch-generator.h>
#include <torch-gobject/torch-generator.h>
#include <torch-gobject/torch-generator-internal.h>
#include <torch-gobject/torch-util.h>

#include <ATen/core/Generator.h>

struct _TorchGenerator
{
  GObject parent_instance;
};

typedef struct _TorchGeneratorPrivate
{
  at::Generator *internal;

  TorchDevice *device;
  uint64_t     construct_seed;
} TorchGeneratorPrivate;

static void initable_iface_init (GInitableIface *iface);

G_DEFINE_TYPE_WITH_CODE (TorchGenerator, torch_generator, G_TYPE_OBJECT,
                         G_ADD_PRIVATE (TorchGenerator)
                         G_IMPLEMENT_INTERFACE (G_TYPE_INITABLE, initable_iface_init))
#define TORCH_GENERATOR_GET_PRIVATE(a) static_cast <TorchGeneratorPrivate *> (torch_generator_get_instance_private ((a)))

enum {
  PROP_0,
  /* Important to put DEVICE before CURRENT_SEED
   * as that ensures that we don't wipe out the seed
   * as soon as the device re-initializes the implementation,
   * in case both properties are set. */
  PROP_DEVICE,
  PROP_CURRENT_SEED,
  NPROPS
};

static GParamSpec *torch_generator_props [NPROPS] = { NULL, };

at::Generator &
torch_generator_get_real_generator (TorchGenerator *generator)
{
  TorchGeneratorPrivate *priv = TORCH_GENERATOR_GET_PRIVATE (generator);
  g_autoptr (GError)     error = NULL;

  if (!torch_generator_init_internal (generator, &error))
    torch_throw_error (error);

  return *priv->internal;
}

TorchGenerator *
torch_generator_new_from_real_generator (at::Generator const &real_generator)
{
  g_autoptr (TorchGenerator) generator = static_cast <TorchGenerator *> (g_object_new (TORCH_TYPE_GENERATOR, NULL));
  TorchGeneratorPrivate *priv = TORCH_GENERATOR_GET_PRIVATE (generator);

  g_assert (priv->internal == NULL);
  priv->internal = new at::Generator (real_generator);

  return static_cast <TorchGenerator *> (g_steal_pointer (&generator));
}

/**
 * torch_generator_get_current_seed:
 * @generator: A #TorchGenerator
 *
 * Returns: The current seed value for the generator.
 */
uint64_t
torch_generator_get_current_seed (TorchGenerator *generator)
{
  TorchGeneratorPrivate *priv = TORCH_GENERATOR_GET_PRIVATE (generator);

  if (priv->internal)
    return priv->internal->current_seed ();

  return priv->construct_seed;
}

/**
 * torch_generator_set_current_seed:
 * @generator: A #TorchGenerator
 * @seed: The seed value to set
 * @error: An out-error parameter.
 *
 * Set a new seed value for this generator. Not threadsafe, you
 * should lock the generator with torch_generator_lock before doing this
 * if you're using threads.
 *
 * Returns: %TRUE on success, %FALSE with @error set on failure.
 */
gboolean
torch_generator_set_current_seed (TorchGenerator  *generator,
                                  uint64_t         seed,
                                  GError         **error)
{
  TorchGeneratorPrivate *priv = TORCH_GENERATOR_GET_PRIVATE (generator);

  if (!torch_generator_init_internal (generator, error))
    return FALSE;

  try
    {
      priv->internal->set_current_seed (seed);
      return TRUE;
    }
  catch (std::exception const &e)
    {
      g_set_error (error, G_IO_ERROR, G_IO_ERROR_FAILED, e.what ());
      return FALSE;
    }
}

/**
 * torch_generator_get_device:
 * @generator: A #TorchGenerator
 * @error: An error out-param
 *
 * Get the current device of the generator.
 *
 * Returns: (transfer full): The current seed value for the generator.
 */
TorchDevice *
torch_generator_get_device (TorchGenerator  *generator,
                            GError         **error)
{
  TorchGeneratorPrivate *priv = TORCH_GENERATOR_GET_PRIVATE (generator);

  if (priv->internal)
    return torch_device_new_from_real_device (priv->internal->device ());

  /* If the device is unset, we'll need to set it now */
  if (priv->device == NULL)
    {
      g_autoptr (TorchDevice) device = torch_device_new (error);

      if (device == NULL)
        return FALSE;

      priv->device = static_cast <TorchDevice *> (g_steal_pointer (&device));
    }

  return static_cast <TorchDevice *> (g_object_ref (priv->device));
}

/**
 * torch_generator_set_device:
 * @generator: A #TorchGenerator
 * @device: (transfer none): A #TorchDevice
 * @error: An error out-param
 *
 * Set the current device of the generator. This will reset the generator
 * state, so the generator should be re-seeded if that's necessary.
 *
 * Returns: %TRUE on success, %FALSE with @error set on failure.
 */
gboolean
torch_generator_set_device (TorchGenerator  *generator,
                            TorchDevice     *device,
                            GError         **error)
{
  TorchGeneratorPrivate *priv = TORCH_GENERATOR_GET_PRIVATE (generator);

  g_clear_object (&priv->device);
  priv->device = static_cast <TorchDevice *> (g_object_ref (device));

  if (priv->internal)
    {
      /* Keep old_internal around in case initialization fails for
       * some reason. */
      at::Generator *old_internal = priv->internal;

      priv->internal = nullptr;

      if (!torch_generator_init_internal (generator, error))
        {
          g_warning ("Failed to internally initialize generator on new generator, retaining old state");
          priv->internal = old_internal;
          return FALSE;
        }

      delete old_internal;
    }

  return TRUE;
}

/**
 * torch_generator_seed:
 * @generator: A #TorchGenerator
 * @out_seed: An out-param to return the new seed.
 * @error: An out-error parameter.
 *
 * Re-seeds the generator with a new value from the system RNG. Not threadsafe, you
 * should lock the generator with torch_generator_lock before doing this
 * if you're using threads.
 *
 * Returns: %TRUE on success, %FALSE with @error set on failure.
 */
gboolean
torch_generator_seed (TorchGenerator  *generator,
                      uint64_t        *out_seed,
                      GError         **error)
{
  TorchGeneratorPrivate *priv = TORCH_GENERATOR_GET_PRIVATE (generator);

  if (!torch_generator_init_internal (generator, error))
      return FALSE;

  try
    {
      uint64_t seed = priv->internal->seed ();

      if (out_seed != NULL)
        *out_seed = seed;

      return TRUE;
    }
  catch (std::exception const &e)
    {
      g_set_error (error, G_IO_ERROR, G_IO_ERROR_FAILED, e.what ());
      return FALSE;
    }
}

/**
 * torch_generator_lock:
 * @generator: A #TorchGenerator
 * @error: An out-error parameter.
 *
 * Locks the generator mutex.
 *
 * Returns: %TRUE on success, %FALSE with @error set on failure.
 */
gboolean
torch_generator_lock (TorchGenerator  *generator,
                      GError         **error)
{
  TorchGeneratorPrivate *priv = TORCH_GENERATOR_GET_PRIVATE (generator);

  if (!torch_generator_init_internal (generator, error))
      return FALSE;

  try
    {
      priv->internal->mutex ().lock ();
      return TRUE;
    }
  catch (std::exception const &e)
    {
      g_set_error (error, G_IO_ERROR, G_IO_ERROR_FAILED, e.what ());
      return FALSE;
    }
}

/**
 * torch_generator_unlock:
 * @generator: A #TorchGenerator
 * @error: An out-error parameter.
 *
 * Unlocks the generator mutex.
 *
 * Returns: %TRUE on success, %FALSE with @error set on failure.
 */
gboolean
torch_generator_unlock (TorchGenerator  *generator,
                        GError         **error)
{
  TorchGeneratorPrivate *priv = TORCH_GENERATOR_GET_PRIVATE (generator);

  /* If the generator isn't initialized at the point where we
   * unlock it, then it couldn't have been locked. */
  g_assert (priv->internal != NULL);

  try
    {
      priv->internal->mutex ().unlock ();
      return TRUE;
    }
  catch (std::exception const &e)
    {
      g_set_error (error, G_IO_ERROR, G_IO_ERROR_FAILED, e.what ());
      return FALSE;
    }
}

gboolean
torch_generator_init_internal (TorchGenerator  *generator,
                               GError         **error)
{
  TorchGeneratorPrivate *priv = TORCH_GENERATOR_GET_PRIVATE (generator);

  g_return_val_if_fail (error == NULL || *error == NULL, FALSE);

  /* Even though we have a check in torch_generator_initable_init,
   * check again here to avoid the vfunc calls */
  if (!priv->internal)
    return g_initable_init (G_INITABLE (generator), NULL, error);

  return TRUE;
}

static gboolean
torch_generator_initable_init (GInitable     *initable,
                               GCancellable  *cancellable,
                               GError       **error)
{
  TorchGenerator        *generator = TORCH_GENERATOR (initable);
  TorchGeneratorPrivate *priv = TORCH_GENERATOR_GET_PRIVATE (generator);

  if (priv->internal)
    return TRUE;

  /* If the device is unset, we'll need to set it now */
  if (priv->device == NULL)
    {
      g_autoptr (TorchDevice) device = torch_device_new (error);

      if (device == NULL)
        return FALSE;

      priv->device = static_cast <TorchDevice *> (g_steal_pointer (&device));
    }

  return call_set_error_on_exception (error, G_IO_ERROR, G_IO_ERROR_FAILED, FALSE, [&]() -> gboolean {
    TorchDeviceType device_type = TORCH_DEVICE_TYPE_CPU;
    g_autoptr (GError) local_error = NULL;

    g_assert (priv->device != NULL);

    if (!torch_device_get_device_type (priv->device, &device_type, &local_error))
      {
        torch_throw_error (local_error);
        return FALSE;
      }

    if (device_type == TORCH_DEVICE_TYPE_CPU)
      {
        priv->internal = new at::Generator (c10::make_intrusive<at::CPUGeneratorImpl> (priv->construct_seed));
      }
    else
      {
        std::stringstream ss;
        g_autofree char *device_string = NULL;
        
        device_string = torch_device_get_string (priv->device, &local_error);
        if (device_string == NULL)
          torch_throw_error (local_error);

        ss << "Unsupported device type '" << device_string << "' for making a Generator";
        throw std::logic_error (ss.str ());
      }

    priv->internal->set_current_seed (priv->construct_seed);

    /* Once we've constructed the internal, everything gets moved to
     * the internal generator (one canonical copy), so we can clear the construct
     * properties that we had in the meantime */
    g_clear_object (&priv->device);
    priv->construct_seed = 0;
    return TRUE;
  });
}

static void
initable_iface_init (GInitableIface *iface)
{
  iface->init = torch_generator_initable_init;
}

static void
torch_generator_init (TorchGenerator *generator)
{
}

static void
torch_generator_get_property (GObject      *object,
                              unsigned int  prop_id,
                              GValue       *value,
                              GParamSpec   *pspec)
{
  TorchGenerator *generator = TORCH_GENERATOR (object);

  switch (prop_id)
    {
      case PROP_CURRENT_SEED:
        g_value_set_uint64 (value,
                            torch_generator_get_current_seed (generator));
        break;
      case PROP_DEVICE:
        g_value_take_object (value,
                             call_and_warn_about_gerror ("get prop 'device'",
                                                         torch_generator_get_device,
                                                         generator));
        break;
      default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
        break;
    }
}

static void
torch_generator_set_property (GObject      *object,
                              unsigned int  prop_id,
                              const GValue *value,
                              GParamSpec   *pspec)
{
  TorchGenerator *generator = TORCH_GENERATOR (object);

  /* Properties only get set on construction */
  switch (prop_id)
    {
      case PROP_CURRENT_SEED:
        call_and_warn_about_gerror ("set prop 'current-seed'",
                                    torch_generator_set_current_seed,
                                    generator,
                                    g_value_get_uint64 (value));
        break;
      case PROP_DEVICE:
        call_and_warn_about_gerror ("set prop 'device'",
                                    torch_generator_set_device,
                                    generator,
                                    static_cast <TorchDevice *> (g_value_get_object (value)));
        break;
      default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
        break;
    }
}

static void
torch_generator_dispose (GObject *object)
{
  TorchGenerator *generator = TORCH_GENERATOR (object);
  TorchGeneratorPrivate *priv = TORCH_GENERATOR_GET_PRIVATE (generator);

  g_clear_object (&priv->device);
}

static void
torch_generator_finalize (GObject *object)
{
  TorchGenerator *generator = TORCH_GENERATOR (object);
  TorchGeneratorPrivate *priv = TORCH_GENERATOR_GET_PRIVATE (generator);

  if (priv->internal)
    {
      delete priv->internal;
      priv->internal = nullptr;
    }
}

static void
torch_generator_class_init (TorchGeneratorClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = torch_generator_get_property;
  object_class->set_property = torch_generator_set_property;
  object_class->dispose = torch_generator_dispose;
  object_class->finalize = torch_generator_finalize;

  torch_generator_props[PROP_DEVICE] =
    g_param_spec_object ("device",
                         "Device",
                         "Device to put the Generator on",
                         TORCH_TYPE_DEVICE,
                         static_cast <GParamFlags> (G_PARAM_READWRITE));

  torch_generator_props[PROP_CURRENT_SEED] =
    g_param_spec_uint64 ("current-seed",
                         "Current Seed",
                         "The current seed value for the generator",
                         0,
                         G_MAXUINT64,
                         0,
                         static_cast <GParamFlags> (G_PARAM_READWRITE));

  g_object_class_install_properties (object_class,
                                     NPROPS,
                                     torch_generator_props);

}
