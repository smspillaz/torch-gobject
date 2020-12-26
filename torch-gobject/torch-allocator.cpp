/*
 * torch-gobject/torch-allocator.cpp
 *
 * Allocator abstraction for creating storage.
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

#include <torch-gobject/torch-allocator.h>
#include <torch-gobject/torch-allocator-internal.h>

#include <c10/core/Allocator.h>

struct _TorchAllocator
{
  GObject parent_instance;
};

typedef struct _TorchAllocatorPrivate
{
  c10::Allocator *internal;
} TorchAllocatorPrivate;

G_DEFINE_TYPE_WITH_PRIVATE (TorchAllocator, torch_allocator, G_TYPE_OBJECT)
#define TORCH_ALLOCATOR_GET_PRIVATE(x) static_cast <TorchAllocatorPrivate *> (torch_allocator_get_instance_private ((x)))

c10::Allocator &
torch_allocator_get_real_allocator (TorchAllocator *allocator)
{
  TorchAllocatorPrivate *priv = TORCH_ALLOCATOR_GET_PRIVATE (allocator);

  return *priv->internal;
}

namespace {
struct GLibAllocator:
  public c10::Allocator
{
  c10::DataPtr allocate(size_t n) const
  {
    gpointer mem = g_malloc (n);
    return c10::DataPtr (mem, mem, reinterpret_cast <c10::DeleterFnPtr> (g_free), c10::DeviceType::CPU);
  }
};
}

static void
torch_allocator_init (TorchAllocator *allocator)
{
  TorchAllocatorPrivate *priv = TORCH_ALLOCATOR_GET_PRIVATE (allocator);
  priv->internal = new GLibAllocator();
}

static void
torch_allocator_finalize (GObject *object)
{
  TorchAllocator *allocator = TORCH_ALLOCATOR (object);
  TorchAllocatorPrivate *priv = TORCH_ALLOCATOR_GET_PRIVATE (allocator);

  if (priv->internal)
    {
      delete priv->internal;
      priv->internal = nullptr;
    }
}

static void
torch_allocator_class_init (TorchAllocatorClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = torch_allocator_finalize;
}

TorchAllocator *
torch_allocator_new (void)
{
  return static_cast<TorchAllocator *> (g_object_new (TORCH_TYPE_ALLOCATOR, NULL));
}
