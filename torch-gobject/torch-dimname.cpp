/*
 * torch-gobject/torch-dimname.cpp
 *
 * Object representing a dimension name.
 *
 * Copyright (C) 2020 Sam Spilsbury.
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

#include <gio/gio.h>

#include <torch-gobject/torch-enums.h>
#include <torch-gobject/torch-dimname.h>
#include <torch-gobject/torch-dimname-internal.h>
#include <torch-gobject/torch-dimname-type.h>
#include <torch-gobject/torch-dimname-type-internal.h>
#include <torch-gobject/torch-enums.h>
#include <torch-gobject/torch-util.h>

#include <ATen/core/Dimname.h>

struct _TorchDimname
{
  GObject parent_instance;
};

typedef struct _TorchDimnamePrivate
{
  at::Dimname *internal;

  GQuark           q_name;
  TorchDimnameType symbol_type;
} TorchDimnamePrivate;

static void initable_iface_init (GInitableIface *iface);

G_DEFINE_TYPE_WITH_CODE (TorchDimname, torch_dimname, G_TYPE_OBJECT,
                         G_ADD_PRIVATE (TorchDimname)
                         G_IMPLEMENT_INTERFACE (G_TYPE_INITABLE, initable_iface_init))
#define TORCH_DIMNAME_GET_PRIVATE(a) static_cast <TorchDimnamePrivate *> (torch_dimname_get_instance_private ((a)))

enum {
  PROP_0,
  PROP_SYMBOL_TYPE,
  PROP_SYMBOL_NAME,
  NPROPS
};

static GParamSpec *torch_dimname_props [NPROPS] = { NULL, };

at::Dimname &
torch_dimname_get_real_dimname (TorchDimname *dimname)
{
  TorchDimnamePrivate  *priv = TORCH_DIMNAME_GET_PRIVATE (dimname);
  g_autoptr (GError)    error = NULL;

  if (!torch_dimname_init_internal (dimname, static_cast <GError **> (&error)))
    torch_throw_error (error);

  return *priv->internal;
}

/**
 * torch_dimname_get_symbol_type:
 * @dimname: A #TorchDimname
 *
 * The type of the #TorchDimname
 *
 * Returns: The #TorchDimnameType of the #TorchDimname.
 */
TorchDimnameType
torch_dimname_get_symbol_type (TorchDimname *dimname)
{
  TorchDimnamePrivate *priv = TORCH_DIMNAME_GET_PRIVATE (dimname);
  return priv->symbol_type;
}

/**
 * torch_dimname_get_symbol_name:
 * @dimname: A #TorchDimname
 *
 * The name of the #TorchDimname
 *
 * Returns: (transfer none): The name of the #TorchDimname
 */
const gchar *
torch_dimname_get_symbol_name (TorchDimname  *dimname)
{
  TorchDimnamePrivate *priv = TORCH_DIMNAME_GET_PRIVATE (dimname);
  return g_quark_to_string (priv->q_name);
}

/**
 * torch_dimname_matches:
 * @dimname: A #TorchDimname
 * @other: The other #TorchDimname
 * @out_matches: (out): Out-parameter for the match result
 * @error: A #GError
 *
 * Whether @dimname matches @other . Use this instead of strcmp, since it handles wildcards.
 *
 * Returns: %TRUE with @out_matches set on success, %FALSE with @error set on failure.
 */
gboolean
torch_dimname_matches (TorchDimname  *dimname,
                       TorchDimname  *other,
                       gboolean      *out_matches,
                       GError       **error)
{
  TorchDimnamePrivate *priv = TORCH_DIMNAME_GET_PRIVATE (dimname);

  g_return_val_if_fail (out_matches != NULL, FALSE);

  if (!torch_dimname_init_internal (dimname, error))
    return FALSE;

  return call_set_error_on_exception (error, G_IO_ERROR, G_IO_ERROR_FAILED, FALSE, [&]() -> gboolean {
    *out_matches = priv->internal->matches (torch_dimname_get_real_dimname (other));
    return TRUE;
  });
}

/**
 * torch_dimname_unify:
 * @dimname: A #TorchDimname
 * @other: The other #TorchDimname
 * @out_unified_dimname: (out) (transfer full): Out-parameter for the unified result
 * @error: A #GError
 *
 * Attempt to unify @dimname with @other by replacing wildcards. If the two dimnames don't
 * match, thne @out_unified_dimname will be set to %NULL. %FALSE is only returned on
 * error.
 *
 * Returns: %TRUE with @out_unified_dimname set on success, %FALSE with @error set on failure.
 */
gboolean
torch_dimname_unify (TorchDimname  *dimname,
                     TorchDimname  *other,
                     TorchDimname **out_unified_dimname,
                     GError       **error)
{
  TorchDimnamePrivate *priv = TORCH_DIMNAME_GET_PRIVATE (dimname);

  g_return_val_if_fail (out_unified_dimname != NULL, FALSE);

  if (!torch_dimname_init_internal (dimname, error))
    return FALSE;

  return call_set_error_on_exception (error, G_IO_ERROR, G_IO_ERROR_FAILED, FALSE, [&]() -> gboolean {
    c10::optional<at::Dimname> unified (priv->internal->unify (torch_dimname_get_real_dimname (other)));
    g_autoptr (TorchDimname) unified_wrapper = unified.has_value () ? torch_dimname_new_from_real_dimname (unified.value ()) : NULL;

    *out_unified_dimname = static_cast <TorchDimname *> (g_steal_pointer (&unified_wrapper));
    return TRUE;
  });
}

namespace {
  at::Dimname make_dimname(at::Symbol name, at::NameType type)
  {
    return type == at::NameType::WILDCARD ? at::Dimname::wildcard () : at::Dimname::fromSymbol (name);
  }
}

static gboolean
torch_dimname_initable_init (GInitable     *initable,
                             GCancellable  *cancellable,
                             GError       **error)
{
  TorchDimname        *Dimname = TORCH_DIMNAME (initable);
  TorchDimnamePrivate *priv = TORCH_DIMNAME_GET_PRIVATE (Dimname);

  if (priv->internal)
    return TRUE;

  return call_set_error_on_exception (error, G_IO_ERROR, G_IO_ERROR_FAILED, FALSE, [&]() -> gboolean {
    priv->internal = new at::Dimname (make_dimname (at::Symbol::dimname (g_quark_to_string (priv->q_name)),
                                                    torch_dimname_type_get_real_type (priv->symbol_type)));

    /* The internal copy is immutable, so we can keep the q_name and
     * symbol_type around */
    return TRUE;
  });
}

static void
initable_iface_init (GInitableIface *iface)
{
  iface->init = torch_dimname_initable_init;
}

static void
torch_dimname_init (TorchDimname *Dimname)
{
}

static void
torch_dimname_get_property (GObject      *object,
                            unsigned int  prop_id,
                            GValue       *value,
                            GParamSpec   *pspec)
{
  TorchDimname *dimname = TORCH_DIMNAME (object);

  /* Properties only get set on construction */
  switch (prop_id)
    {
      case PROP_SYMBOL_NAME:
        g_value_set_static_string (value,
                                   torch_dimname_get_symbol_name (dimname));
        break;
      case PROP_SYMBOL_TYPE:
        g_value_set_enum (value,
                          static_cast <int> (torch_dimname_get_symbol_type (dimname)));
        break;
      default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
        break;
    }
}

static void
torch_dimname_set_property (GObject      *object,
                            unsigned int  prop_id,
                            const GValue *value,
                            GParamSpec   *pspec)
{
  TorchDimname *dimname = TORCH_DIMNAME (object);
  TorchDimnamePrivate *priv = TORCH_DIMNAME_GET_PRIVATE (dimname);

  /* Properties only get set on construction */
  switch (prop_id)
    {
      case PROP_SYMBOL_NAME:
        {
          const char *name = g_value_get_string (value);
          priv->q_name = g_quark_from_string (g_value_get_string (value));

          /* If setting q_name to anything other than '*', then the type
           * shifts to BASIC */
          if (g_strcmp0 (name, "*") != 0)
            priv->symbol_type = TORCH_DIMNAME_TYPE_BASIC;

          break;
        }
      case PROP_SYMBOL_TYPE:
        priv->symbol_type = static_cast <TorchDimnameType> (g_value_get_enum (value));
        break;
      default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
        break;
    }
}

static void
torch_dimname_finalize (GObject *object)
{
  TorchDimname *dimname = TORCH_DIMNAME (object);
  TorchDimnamePrivate *priv = TORCH_DIMNAME_GET_PRIVATE (dimname);

  if (priv->internal)
    {
      delete priv->internal;
      priv->internal = nullptr;
    }
}

static void
torch_dimname_class_init (TorchDimnameClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = torch_dimname_get_property;
  object_class->set_property = torch_dimname_set_property;
  object_class->finalize = torch_dimname_finalize;

  torch_dimname_props[PROP_SYMBOL_NAME] =
    g_param_spec_string ("symbol-name",
                         "Symbol Name",
                         "The name of the Dimname",
                         "*",
                         static_cast <GParamFlags> (G_PARAM_READWRITE | G_PARAM_CONSTRUCT_ONLY));

  torch_dimname_props[PROP_SYMBOL_TYPE] =
    g_param_spec_enum ("symbol-type",
                       "Symbol type",
                       "The type of the Dimname",
                       TORCH_TYPE_DIMNAME_TYPE,
                       TORCH_DIMNAME_TYPE_WILDCARD,
                       static_cast <GParamFlags> (G_PARAM_READWRITE | G_PARAM_CONSTRUCT_ONLY));

  g_object_class_install_properties (object_class,
                                     NPROPS,
                                     torch_dimname_props);
}

gboolean
torch_dimname_init_internal (TorchDimname  *dimname,
                             GError       **error)
{
  TorchDimnamePrivate *priv = TORCH_DIMNAME_GET_PRIVATE (dimname);

  g_return_val_if_fail (error == NULL || *error == NULL, FALSE);

  /* Even though we have a check in torch_DIMNAME_initable_init,
   * check again here to avoid the vfunc calls */
  if (!priv->internal)
    return g_initable_init (G_INITABLE (dimname), NULL, error);

  return TRUE;
}

TorchDimname *
torch_dimname_new_from_real_dimname (const at::Dimname &dimname_real)
{
  g_autoptr (TorchDimname) dimname = static_cast <TorchDimname *> (g_object_new (TORCH_TYPE_DIMNAME,
                                                                                 "symbol-name",
                                                                                 dimname_real.symbol ().toUnqualString (),
                                                                                 "symbol-type",
                                                                                 torch_dimname_type_from_real_type (dimname_real.type ()),
                                                                                 NULL));
  g_autoptr (GError) error = NULL;

  if (!torch_dimname_init_internal (dimname, &error))
    torch_throw_error (error);

  return static_cast <TorchDimname *> (g_steal_pointer (&dimname));
}

/**
 * torch_dimname_new_wildcard:
 * @error: A #GError
 *
 * Create a new wildcard dimname.
 *
 * Returns: (transfer full): A #TorchDimname on success, %NULL with @error set on failure.
 */
TorchDimname *
torch_dimname_new_wildcard (GError **error)
{
  g_autoptr (TorchDimname) dimname = static_cast <TorchDimname *> (g_object_new (TORCH_TYPE_DIMNAME,
                                                                                 "symbol-type",
                                                                                 TORCH_DIMNAME_TYPE_WILDCARD,
                                                                                 NULL));

  if (!torch_dimname_init_internal (dimname, error))
    return NULL;

  return static_cast <TorchDimname *> (g_steal_pointer (&dimname));
}

/**
 * torch_dimname_new_with_name:
 * @name: The name for the #TorchDimname
 * @error: A #GError
 *
 * Create a new named dimname.
 *
 * Returns: (transfer full): A #TorchDimname on success, %NULL with @error set on failure.
 */
TorchDimname *
torch_dimname_new_with_name (const char  *name,
                             GError     **error)
{
  g_autoptr (TorchDimname) dimname = static_cast <TorchDimname *> (g_object_new (TORCH_TYPE_DIMNAME,
                                                                                 "symbol-name",
                                                                                 name,
                                                                                 "symbol-type",
                                                                                 TORCH_DIMNAME_TYPE_BASIC,
                                                                                 NULL));

  if (!torch_dimname_init_internal (dimname, error))
    return NULL;

  return static_cast <TorchDimname *> (g_steal_pointer (&dimname));
}
