/*
 * torch-gobject/torch-storage.cpp
 *
 * Storage abstraction for data to be passed to a tensor.
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

#include <gio/gio.h>

#include <torch-gobject/torch-allocator.h>
#include <torch-gobject/torch-allocator-internal.h>
#include <torch-gobject/torch-storage.h>
#include <torch-gobject/torch-storage-internal.h>
#include <torch-gobject/torch-util.h>

#include <c10/core/Storage.h>

struct _TorchStorage
{
  GObject parent_instance;
};

typedef struct _TorchStoragePrivate
{
  c10::Storage *internal;

  size_t          n_bytes;
  gpointer        data_ptr;
  GDestroyNotify  destroy_func;
  gboolean        resizable;
  TorchAllocator *allocator;
} TorchStoragePrivate;

static void initable_iface_init (GInitableIface *iface);

G_DEFINE_TYPE_WITH_CODE (TorchStorage, torch_storage, G_TYPE_OBJECT,
                         G_ADD_PRIVATE (TorchStorage)
                         G_IMPLEMENT_INTERFACE (G_TYPE_INITABLE, initable_iface_init))
#define TORCH_STORAGE_GET_PRIVATE(a) static_cast <TorchStoragePrivate *> (torch_storage_get_instance_private ((a)))

enum {
  PROP_0,
  PROP_RESIZABLE,
  PROP_N_BYTES,
  PROP_ALLOCATOR,
  NPROPS
};

static GParamSpec *torch_storage_props [NPROPS] = { NULL, };

c10::Storage &
torch_storage_get_real_storage (TorchStorage *storage)
{
  TorchStoragePrivate *priv = TORCH_STORAGE_GET_PRIVATE (storage);

  return *priv->internal;
}

gboolean
torch_storage_get_resizable (TorchStorage  *storage,
                             gboolean      *out_resizable,
                             GError       **error)
{
  TorchStoragePrivate *priv = TORCH_STORAGE_GET_PRIVATE (storage);

  g_return_val_if_fail (out_resizable != NULL, FALSE);

  if (!torch_storage_init_internal (storage, error))
    return FALSE;

  try
    {
      *out_resizable = priv->internal->resizable ();
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

gboolean
torch_storage_get_n_bytes (TorchStorage  *storage,
                           size_t        *out_n_bytes,
                           GError       **error)
{
  TorchStoragePrivate *priv = TORCH_STORAGE_GET_PRIVATE (storage);

  g_return_val_if_fail (out_n_bytes != NULL, FALSE);

  if (!torch_storage_init_internal (storage, error))
    return FALSE;

  try
    {
      *out_n_bytes = priv->internal->nbytes ();
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

const gpointer
torch_storage_get_data (TorchStorage  *storage,
                        GError       **error)
{
  TorchStoragePrivate *priv = TORCH_STORAGE_GET_PRIVATE (storage);

  if (!torch_storage_init_internal (storage, error))
    return NULL;

  try
    {
      return static_cast <gpointer> (priv->internal->data <char *> ());
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

GBytes *
torch_storage_get_bytes (TorchStorage  *storage,
                         GError       **error)
{
  TorchStoragePrivate *priv = TORCH_STORAGE_GET_PRIVATE (storage);

  if (!torch_storage_init_internal (storage, error))
    return NULL;

  try
    {
      const size_t n_bytes = priv->internal->nbytes();
      g_autofree char *data = static_cast <char *> (g_malloc (sizeof (char) * n_bytes));

      // XXX: This assumes that memcpy is even possible on the pointer,
      //      it may very well not be
      memcpy (static_cast <gpointer> (data), priv->internal->data <char *> (), n_bytes * sizeof (char));

      return g_bytes_new_with_free_func (g_steal_pointer (&data), n_bytes, g_free, NULL);
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

/**
 * torch_storage_get_allocator:
 * @storage: A #TorchStorage
 *
 * Get the allocator associated with this #TorchStorage
 *
 * Returns: (transfer none): The #TorchStorage
 */
TorchAllocator *
torch_storage_get_allocator (TorchStorage *storage)
{
  TorchStoragePrivate *priv = TORCH_STORAGE_GET_PRIVATE (storage);
  return priv->allocator;
}

namespace {
c10::DataPtr
data_ptr_from_data (gpointer data_ptr,
                    GDestroyNotify destroy_func)
{
  return c10::DataPtr (data_ptr, data_ptr, destroy_func, c10::DeviceType::CPU);
}

}

static gboolean
torch_storage_initable_init (GInitable     *initable,
                             GCancellable  *cancellable,
                             GError       **error)
{
  TorchStorage        *storage = TORCH_STORAGE (initable);
  TorchStoragePrivate *priv = TORCH_STORAGE_GET_PRIVATE (storage);

  if (priv->internal)
    return TRUE;

  try
    {
      if (priv->data_ptr)
        {
          priv->internal = new c10::Storage (c10::Storage::use_byte_size_t{},
                                             priv->n_bytes,
                                             data_ptr_from_data (priv->data_ptr,
                                                                 priv->destroy_func),
                                             nullptr,
                                             priv->resizable);
        }
      else if (priv->allocator)
        {
          priv->internal = new c10::Storage (c10::Storage::use_byte_size_t{},
                                             priv->n_bytes,
                                             &torch_allocator_get_real_allocator (priv->allocator),
                                             priv->resizable);
        }
      else
        {
          throw std::logic_error ("Need to provide either a data_ptr or allocator");
        }

      /* Once we've constructed the internal, everything gets moved to
       * the internal storage (one canonical copy), so we can clear the construct
       * properties that we had in the meantime */
      g_clear_pointer (&priv->allocator, g_object_unref);
      priv->n_bytes = 0;
      priv->resizable = 0;
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
  iface->init = torch_storage_initable_init;
}

static void
torch_storage_init (TorchStorage *storage)
{
}

static void
torch_storage_get_property (GObject      *object,
                            unsigned int  prop_id,
                            GValue       *value,
                            GParamSpec   *pspec)
{
  TorchStorage *storage = TORCH_STORAGE (object);

  /* Properties only get set on construction */
  switch (prop_id)
    {
      case PROP_RESIZABLE:
        g_value_set_boolean (value,
                             call_and_warn_about_gerror ("get prop 'resizable'",
                                                         [](TorchStorage  *storage,
                                                            GError       **error) -> gboolean {
                                                           gboolean resizable = FALSE;
                                                           torch_storage_get_resizable (storage,
                                                                                        &resizable,
                                                                                        error);
                                                           return resizable;
                                                         },
                                                         storage));
        break;
      case PROP_N_BYTES:
        g_value_set_uint64 (value,
                            call_and_warn_about_gerror ("get prop 'n-bytes'",
                                                        [](TorchStorage  *storage,
                                                           GError       **error) -> gboolean {
                                                          size_t n_bytes = 0;
                                                          torch_storage_get_n_bytes (storage,
                                                                                     &n_bytes,
                                                                                     error);
                                                          return n_bytes;
                                                        },
                                                        storage));
        break;
      case PROP_ALLOCATOR:
        g_value_set_object (value, torch_storage_get_allocator (storage));
        break;
      default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
        break;
    }
}

static void
torch_storage_set_property (GObject      *object,
                            unsigned int  prop_id,
                            const GValue *value,
                            GParamSpec   *pspec)
{
  TorchStorage *storage = TORCH_STORAGE (object);
  TorchStoragePrivate *priv = TORCH_STORAGE_GET_PRIVATE (storage);

  /* Properties only get set on construction */
  switch (prop_id)
    {
      case PROP_RESIZABLE:
        priv->resizable = g_value_get_boolean (value);
        break;
      case PROP_N_BYTES:
        priv->n_bytes = g_value_get_uint64 (value);
        break;
      case PROP_ALLOCATOR:
        priv->allocator = reinterpret_cast <TorchAllocator *> (g_value_dup_object (value));
        break;
      default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
        break;
    }
}

static void
torch_storage_dispose (GObject *object)
{
  TorchStorage *storage = TORCH_STORAGE (object);
  TorchStoragePrivate *priv = TORCH_STORAGE_GET_PRIVATE (storage);

  g_clear_pointer (&priv->allocator, g_object_unref);
}

static void
torch_storage_finalize (GObject *object)
{
  TorchStorage *storage = TORCH_STORAGE (object);
  TorchStoragePrivate *priv = TORCH_STORAGE_GET_PRIVATE (storage);

  if (priv->internal)
    {
      delete priv->internal;
      priv->internal = nullptr;
    }
}


static void
torch_storage_class_init (TorchStorageClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = torch_storage_get_property;
  object_class->set_property = torch_storage_set_property;
  object_class->dispose = torch_storage_dispose;
  object_class->finalize = torch_storage_finalize;

  torch_storage_props[PROP_RESIZABLE] =
    g_param_spec_boolean ("resizable",
                          "Resizable",
                          "Whether the storage is resizable",
                          FALSE,
                          static_cast <GParamFlags> (G_PARAM_READWRITE | G_PARAM_CONSTRUCT_ONLY));

  torch_storage_props[PROP_N_BYTES] =
    g_param_spec_uint64 ("n-bytes",
                         "N Bytes",
                         "Size of the storage in bytes",
                         0,
                         G_MAXUINT64,
                         0,
                         static_cast <GParamFlags> (G_PARAM_READWRITE | G_PARAM_CONSTRUCT_ONLY));

  torch_storage_props[PROP_ALLOCATOR] =
    g_param_spec_object ("allocator",
                         "Allocator",
                         "TorchAllocator that allocates this storage",
                         TORCH_TYPE_ALLOCATOR,
                         static_cast <GParamFlags> (G_PARAM_READWRITE | G_PARAM_CONSTRUCT_ONLY));

  g_object_class_install_properties (object_class,
                                     NPROPS,
                                     torch_storage_props);
}

gboolean
torch_storage_init_internal (TorchStorage  *storage,
                             GError       **error)
{
  TorchStoragePrivate *priv = TORCH_STORAGE_GET_PRIVATE (storage);

  g_return_val_if_fail (error == NULL || *error == NULL, FALSE);

  /* Even though we have a check in torch_storage_initable_init,
   * check again here to avoid the vfunc calls */
  if (!priv->internal)
    return g_initable_init (G_INITABLE (storage), NULL, error);
}

TorchStorage *
torch_storage_new_with_fixed_data (GBytes  *data,
                                   GError **error)
{
  g_return_val_if_fail (error == NULL || *error == NULL, NULL);

  return static_cast <TorchStorage *> (g_initable_new (TORCH_TYPE_STORAGE,
                                                       NULL,
                                                       error,
                                                       "n-bytes", g_bytes_get_size (data),
                                                       "resizable", FALSE,
                                                       "allocator", NULL,
                                                       NULL));
}

TorchStorage *
torch_storage_new_with_allocator (size_t           size_bytes,
                                  TorchAllocator  *allocator,
                                  gboolean         resizable,
                                  GError         **error)
{
  g_return_val_if_fail (error == NULL || *error == NULL, NULL);

  return static_cast <TorchStorage *> (g_initable_new (TORCH_TYPE_ALLOCATOR,
                                                       NULL,
                                                       error,
                                                       "n-bytes", size_bytes,
                                                       "resizable", resizable,
                                                       "allocator", allocator,
                                                       NULL));
}

TorchStorage *
torch_storage_new_with_reallocatable_data (size_t           size_bytes,
                                           gpointer         data,
                                           GDestroyNotify   destroy_func,
                                           TorchAllocator  *allocator,
                                           gboolean         resizable,
                                           GError         **error)
{
  g_return_val_if_fail (error == NULL || *error == NULL, NULL);

  g_autoptr(TorchStorage) storage = static_cast <TorchStorage *> (g_object_new (TORCH_TYPE_STORAGE,
                                                                  NULL,
                                                                  error,
                                                                  "n-bytes", size_bytes,
                                                                  "resizable", resizable,
                                                                  "allocator", allocator,
                                                                  NULL));

  TorchStoragePrivate *priv = TORCH_STORAGE_GET_PRIVATE (storage);

  priv->data_ptr = data;
  priv->destroy_func = destroy_func;

  if (!g_initable_init (G_INITABLE (storage),
                        NULL,
                        error))
    {
      return FALSE;
    }

  return reinterpret_cast <TorchStorage *> (g_steal_pointer (&storage));
}
