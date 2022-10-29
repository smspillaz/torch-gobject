/*
 * torch-gobject/torch-callback-data.h
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

#pragma once

#include <glib.h>

G_BEGIN_DECLS

typedef struct _TorchCallbackData TorchCallbackData;

TorchCallbackData * torch_callback_data_new (gpointer callback, gpointer user_data, GDestroyNotify user_data_destroy);

TorchCallbackData * torch_callback_data_ref (TorchCallbackData *callback_data);

void torch_callback_data_unref (TorchCallbackData *callback_data);

#define TORCH_TYPE_CALLBACK_DATA (torch_callback_data_get_type());
GType torch_callback_data_get_type (void);

G_END_DECLS
