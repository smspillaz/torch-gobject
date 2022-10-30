/*
 * torch-gobject/torch-callback-data.c
 *
 * Lightweight container object for callback data.
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

#include <glib-object.h>
#include <torch-gobject/torch-callback-data.h>
#include <torch-gobject/torch-callback-data-internal.h>

void
torch_callback_data_unref (TorchCallbackData *callback_data)
{
  g_assert (callback_data->ref_count > 0);

  if (--callback_data->ref_count == 0)
    {
      if (callback_data->user_data != NULL && callback_data->user_data_destroy != NULL)
        {
          callback_data->user_data_destroy (callback_data->user_data);
          callback_data->user_data = NULL;
        }

      g_free (callback_data);
    }
}

TorchCallbackData *
torch_callback_data_ref (TorchCallbackData *callback_data)
{
  ++callback_data->ref_count;

  return callback_data;
}

TorchCallbackData *
torch_callback_data_new (gpointer callback, gpointer user_data, GDestroyNotify user_data_destroy)
{
  TorchCallbackData *callback_data = g_new0 (TorchCallbackData, 1);

  callback_data->callback = callback;
  callback_data->user_data = user_data;
  callback_data->user_data_destroy = user_data_destroy;

  return callback_data;
}

G_DEFINE_BOXED_TYPE (TorchCallbackData, torch_callback_data, torch_callback_data_ref, torch_callback_data_unref);