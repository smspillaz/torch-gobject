# /torch-gobject/nn/meson.build
#
# Build the libtorch-gobject library, nn components.
#
# Copyright (C) 2020 Sam Spilsbury.
#
# torch-gobject is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or
# (at your option) any later version.
#
# torch-gobject is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with torch-gobject; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

torch_gobject_nn_headers_subdir = join_paths(torch_gobject_headers_subdir, 'nn')

subdir('options')

torch_gobject_nn_headers = files([
  'torch-nn-any-module.h',
  'torch-nn-any-module-castable.h',
  'torch-nn-transformer-decoder-layer.h',
  'torch-nn-transformer-encoder-layer.h',
  'torch-nn-module-base.h'
])
torch_gobject_nn_introspectable_sources = files([
  'torch-nn-any-module.cpp',
  'torch-nn-any-module-castable.cpp',
  'torch-nn-transformer-decoder-layer.cpp',
  'torch-nn-transformer-encoder-layer.cpp',
  'torch-nn-module-base.cpp'
])
torch_gobject_nn_private_headers = files([
  'torch-nn-any-module-internal.h',
  'torch-nn-transformer-decoder-layer-internal.h',
  'torch-nn-transformer-encoder-layer-internal.h'
])

torch_gobject_introspectable_sources += torch_gobject_nn_introspectable_sources
torch_gobject_headers += torch_gobject_nn_headers
torch_gobject_private_headers += torch_gobject_nn_private_headers
torch_gobject_include_directories += [include_directories('.')]

install_headers(
  torch_gobject_nn_headers,
  subdir: torch_gobject_nn_headers_subdir
)