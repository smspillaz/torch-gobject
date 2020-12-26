# /torch-gobject/meson.build
#
# Build the libtorch-gobject library.
#
# Copyright (C) 2020 Sam Spilsbury.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

python = import('python')
gnome = import('gnome')
cpp = meson.get_compiler('cpp')

api_version = '0'

torch_gobject_toplevel_headers = files([
  'torch-tensor.h',
  'torch-tensor-generated.h'
])
torch_gobject_toplevel_introspectable_sources = files([
  'torch-tensor.cpp',
  'torch-tensor-generated.cpp'
])
torch_gobject_toplevel_private_headers = files([
  'torch-tensor-internal.h',
])
torch_gobject_toplevel_private_sources = files([
  'torch-tensor-internal.cpp'
])

torch_gobject_introspectable_sources = files([])
torch_gobject_private_headers = files([])
torch_gobject_private_sources = files([])
torch_gobject_headers = files([])
torch_gobject_headers_subdir = 'torch-gobject'

torch_gobject_introspectable_sources += torch_gobject_toplevel_introspectable_sources
torch_gobject_private_sources += torch_gobject_toplevel_private_sources
torch_gobject_headers += torch_gobject_toplevel_headers
torch_gobject_private_headers += torch_gobject_toplevel_private_headers

install_headers(torch_gobject_toplevel_headers, subdir: torch_gobject_headers_subdir)

torch_gobject_sources = torch_gobject_introspectable_sources + torch_gobject_private_sources

glib = dependency('glib-2.0')
gobject = dependency('gobject-2.0')

torch_gobject_lib = shared_library(
  'torch-gobject',
  torch_gobject_sources,
  soversion: api_version,
  install: true,
  include_directories: [ torch_gobject_inc ],
  dependencies: [ glib, gobject, torch_dep ]
)

torch_gobject_dep = declare_dependency(
  link_with: torch_gobject_lib,
  include_directories: [ torch_gobject_inc ],
)

introspection_sources = [ torch_gobject_introspectable_sources, torch_gobject_headers ]

gnome = import('gnome')
torch_gobject_gir = gnome.generate_gir(
  torch_gobject_lib,
  extra_args: ['--warn-all'],
  identifier_prefix: 'Torch',
  include_directories: torch_gobject_inc,
  includes: ['GLib-2.0', 'GObject-2.0'],
  install: true,
  namespace: 'Torch',
  nsversion: api_version,
  sources: introspection_sources,
  symbol_prefix: 'torch',
  fatal_warnings: true,
)[0]

pkg = import('pkgconfig')
pkg.generate(
  description: 'Torch Differentiable Computing Library (GObject Binding)',
  name: 'libtorch-gobject',
  filebase: 'libtorch-gobject-' + api_version,
  version: meson.project_version(),
  libraries: torch_gobject_lib,
  install_dir: join_paths(get_option('libdir'), 'pkgconfig')
)
