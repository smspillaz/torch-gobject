/*
 * torch-gobject/torch-slice.cpp
 *
 * Object representing a slice.
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

#include <torch/torch.h>

#include <torch-gobject/torch-slice.h>
#include <torch-gobject/torch-slice.h>

struct _TorchSlice {
  int64_t start;
  int64_t stop;
  int64_t step;
};

GType torch_slice_get_type ()
{
  static GType slice_type = 0;

  if (g_once_init_enter (&slice_type))
    {
      slice_type = g_boxed_type_register_static ("TorchSlice",
                                                 (GBoxedCopyFunc) torch_slice_copy,
                                                 (GBoxedFreeFunc) torch_slice_free);
    }

  return slice_type;
}

torch::indexing::Slice
torch_slice_get_real_slice (TorchSlice *slice)
{
  return torch::indexing::Slice (slice->start, slice->stop, slice->step);
}

TorchSlice *
torch_slice_new_from_real_slice (torch::indexing::Slice const &slice)
{
  return torch_slice_new (
    slice.start().expect_int(),
    slice.stop().expect_int(),
    slice.step().expect_int()
  );
}

/**
 * torch_slice_new:
 * @start: The starting position
 * @stop: The stopping position
 * @step: The step size
 *
 * Make a new #TorchSlice with starting at @start, stopping at @stop (exclusive)
 * with step @step.
 *
 * Returns: (transfer full): A new #TorchSlice.
 */
TorchSlice *
torch_slice_new (int64_t start, int64_t stop, int64_t step)
{
  TorchSlice *slice = static_cast <TorchSlice *> (g_new0 (TorchSlice, 1));

  slice->start = start;
  slice->stop = stop;
  slice->step = step;

  return slice;
}

/**
 * torch_slice_copy:
 * @slice: A #TorchSlice
 *
 * Make a copy of this #TorchSlice
 *
 * Returns: (transfer full): A new #TorchSlice with the same value as @slice
 */
TorchSlice *
torch_slice_copy (TorchSlice *slice)
{
  TorchSlice *new_slice = torch_slice_new (slice->start, slice->stop, slice->step);

  return new_slice;
}

/**
 * torch_slice_free:
 * @slice: (transfer full): A #TorchSlice
 *
 * Free this torch slice.
 */
void torch_slice_free (TorchSlice *slice)
{
  g_free (slice);
}
