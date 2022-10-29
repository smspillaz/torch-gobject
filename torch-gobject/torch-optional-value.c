/*
 * torch-gobject/torch-optional-value.c
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

#include <torch-gobject/torch-optional-value.h>

struct _TorchOptionalValue {
    GValue internal_value;
};

/**
 * torch_optional_value_new_gtype:
 * @value: The #GType to put in the #TorchOptionalValue.
 *
 * Returns: (transfer full): A new #TorchOptionalValue with data @value and type %G_TYPE_DOUBLE
 */
TorchOptionalValue *
torch_optional_value_new_gtype (GType value)
{
    TorchOptionalValue *optional_value = g_new0 (TorchOptionalValue, 1);
    g_value_init (&optional_value->internal_value, G_TYPE_GTYPE);
    g_value_set_gtype (&optional_value->internal_value, value);

    return optional_value;
}

/**
 * torch_optional_value_new_double:
 * @value: The #double to put in the #TorchOptionalValue.
 *
 * Returns: (transfer full): A new #TorchOptionalValue with data @value and type %G_TYPE_DOUBLE
 */
TorchOptionalValue *
torch_optional_value_new_double (double value)
{
    TorchOptionalValue *optional_value = g_new0 (TorchOptionalValue, 1);
    g_value_init (&optional_value->internal_value, G_TYPE_DOUBLE);
    g_value_set_double (&optional_value->internal_value, value);

    return optional_value;
}

/**
 * torch_optional_value_new_int64_t:
 * @value: The #int64_t to put in the #TorchOptionalValue.
 *
 * Returns: (transfer full): A new #TorchOptionalValue with data @value and type %G_TYPE_INT64
 */
TorchOptionalValue *
torch_optional_value_new_int64_t (int64_t value)
{
    TorchOptionalValue *optional_value = g_new0 (TorchOptionalValue, 1);
    g_value_init (&optional_value->internal_value, G_TYPE_INT64);
    g_value_set_int64 (&optional_value->internal_value, value);

    return optional_value;
}

/**
 * torch_optional_value_get_gtype:
 * @value: The #TorchOptionalValue to get the internal value from.
 *
 * It is an error to use this function on any #TorchOptionalValue that does
 * not contain a #double.
 *
 * Returns: A #double with the internal value.
 */
GType
torch_optional_value_get_gtype (TorchOptionalValue *value)
{
    return g_value_get_gtype (&value->internal_value);
}


/**
 * torch_optional_value_get_double:
 * @value: The #TorchOptionalValue to get the internal value from.
 *
 * It is an error to use this function on any #TorchOptionalValue that does
 * not contain a #double.
 *
 * Returns: A #double with the internal value.
 */
double
torch_optional_value_get_double (TorchOptionalValue *value)
{
    return g_value_get_double (&value->internal_value);
}

/**
 * torch_optional_value_get_int64_t:
 * @value: The #TorchOptionalValue to get the internal value from.
 *
 * It is an error to use this function on any #TorchOptionalValue that does
 * not contain a #int64_t.
 *
 * Returns: A #int64_t with the internal value.
 */
int64_t
torch_optional_value_get_int64_t (TorchOptionalValue *value)
{
    return g_value_get_int64 (&value->internal_value);
}

/**
 * torch_optional_value_copy:
 * @value: (transfer none): The #TorchOptionalValue to copy.
 *
 * Returns: (transfer full): A new #TorchOptionalValue which is a copy of @value
 */
TorchOptionalValue *
torch_optional_value_copy (TorchOptionalValue *value)
{
    TorchOptionalValue *copy = g_new0 (TorchOptionalValue, 1);
    g_value_init (&copy->internal_value, G_VALUE_TYPE (&value->internal_value));
    g_value_copy (&value->internal_value, &copy->internal_value);

    return copy;
}

/**
 * torch_optional_value_free:
 * @value: The #TorchOptionalValue to free.
 */
void
torch_optional_value_free (TorchOptionalValue *value)
{
    g_value_unset (&value->internal_value);
    g_free (value);
}

G_DEFINE_BOXED_TYPE (TorchOptionalValue, torch_optional_value, (GBoxedCopyFunc) torch_optional_value_copy, (GBoxedFreeFunc) torch_optional_value_free)