/*
 * torch-gobject/torch-optional-value.h
 *
 * Utility boxed type to represent an optional value. A value of %NULL
 * means that the value is empty.
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

#pragma once

#include <glib-object.h>

G_BEGIN_DECLS

typedef struct _TorchOptionalValue TorchOptionalValue;

TorchOptionalValue * torch_optional_value_new_gtype (GType value);

TorchOptionalValue * torch_optional_value_new_double (double value);

TorchOptionalValue * torch_optional_value_new_int64_t (int64_t value);

TorchOptionalValue * torch_optional_value_copy (TorchOptionalValue *value);

GType torch_optional_value_get_gtype (TorchOptionalValue *value);

double torch_optional_value_get_double (TorchOptionalValue *value);

int64_t torch_optional_value_get_int64_t (TorchOptionalValue *value);

void torch_optional_value_free (TorchOptionalValue *value);

GType torch_optional_value_get_type (void);
#define TORCH_TYPE_OPTIONAL_VALUE (torch_optional_value_get_type ())


G_END_DECLS
