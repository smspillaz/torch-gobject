/*
 * torch-gobject/torch-dimname.h
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

#pragma once

#include <glib-object.h>
#include <torch-gobject/torch-dimname-type.h>

G_BEGIN_DECLS

#define TORCH_TYPE_DIMNAME torch_dimname_get_type ()
G_DECLARE_FINAL_TYPE (TorchDimname, torch_dimname, TORCH, DIMNAME, GObject)

TorchDimnameType torch_dimname_get_symbol_type (TorchDimname *dimname);

const gchar * torch_dimname_get_symbol_name (TorchDimname  *dimname);

gboolean torch_dimname_matches (TorchDimname  *dimname,
                                TorchDimname  *other,
                                gboolean      *out_matches,
                                GError       **error);

gboolean torch_dimname_unify (TorchDimname  *dimname,
                              TorchDimname  *other,
                              TorchDimname **out_unified_dimname,
                              GError       **error);

TorchDimname * torch_dimname_new_wildcard (GError **error);

TorchDimname * torch_dimname_new_with_name (const char  *name,
                                            GError     **error);

G_END_DECLS
