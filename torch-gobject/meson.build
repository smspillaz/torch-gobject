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
torch_gobject_headers_subdir = 'torch-gobject'

python_installation = python.find_installation(required: true, modules: ['yaml'])
torch_gobject_aten_generated_source = custom_target('gen-torch-aten-src',
                                                    input : ['codegen.py'],
                                                    output : ['torch-tensor-generated.cpp'],
                                                    command : [python_installation.path(), '@INPUT@', '--output', '@OUTPUT0@', '/usr/local/share/ATen/Declarations.yaml'])
torch_gobject_aten_generated_header = custom_target('gen-torch-aten-header',
                                                    input : ['codegen.py'],
                                                    output : ['torch-tensor-generated.h'],
                                                    command : [python_installation.path(), '@INPUT@', '--header', '--output', '@OUTPUT0@', '/usr/local/share/ATen/Declarations.yaml'],
                                                    install: true,
                                                    install_dir: get_option('includedir') / torch_gobject_headers_subdir)


torch_gobject_toplevel_enums_headers = files([
  'torch-device-type.h',
  'torch-layout.h',
  'torch-errors.h',
  'torch-memory-format.h',
])

torch_gobject_enums = gnome.mkenums_simple(
  'torch-enums',
  sources: torch_gobject_toplevel_enums_headers,
  install_header: true,
  install_dir: get_option('includedir') / torch_gobject_headers_subdir
)

torch_gobject_toplevel_headers = files([
  'torch-allocator.h',
  'torch-device.h',
  'torch-storage.h',
  'torch-tensor.h',
  'torch-tensor-options.h'
]) + torch_gobject_toplevel_enums_headers
torch_gobject_toplevel_introspectable_sources = files([
  'torch-allocator.cpp',
  'torch-device.cpp',
  'torch-device-type.cpp',
  'torch-errors.c',
  'torch-layout.cpp',
  'torch-memory-format.cpp',
  'torch-storage.cpp',
  'torch-tensor.cpp',
  'torch-tensor-options.cpp'
])
torch_gobject_toplevel_private_headers = files([
  'torch-device-internal.h',
  'torch-device-type-internal.h',
  'torch-layout-internal.h',
  'torch-memory-format-internal.h',
  'torch-storage-internal.h',
  'torch-tensor-internal.h',
  'torch-tensor-options-internal.h',
  'torch-util.h'
])
torch_gobject_toplevel_private_sources = files([
  'torch-util.cpp'
])

torch_gobject_introspectable_sources = files([])
torch_gobject_private_headers = files([])
torch_gobject_private_sources = files([])
torch_gobject_headers = files([])


torch_gobject_introspectable_sources += torch_gobject_toplevel_introspectable_sources
torch_gobject_private_sources += torch_gobject_toplevel_private_sources
torch_gobject_headers += torch_gobject_toplevel_headers
torch_gobject_private_headers += torch_gobject_toplevel_private_headers

install_headers(
  torch_gobject_toplevel_headers,
  subdir: torch_gobject_headers_subdir
)

torch_gobject_sources = torch_gobject_introspectable_sources + torch_gobject_private_sources

c10 = cpp.find_library('c10')
torch_cpu = cpp.find_library('torch_cpu')
glib = dependency('glib-2.0')
gio = dependency('gio-2.0')
gobject = dependency('gobject-2.0')

torch_gobject_lib = shared_library(
  'torch-gobject',
  torch_gobject_sources,
  torch_gobject_aten_generated_source,
  torch_gobject_enums,
  soversion: api_version,
  install: true,
  include_directories: [ torch_gobject_inc ],
  dependencies: [ c10, gio, glib, gobject, torch_cpu, torch_dep ]
)

torch_gobject_dep = declare_dependency(
  link_with: torch_gobject_lib,
  include_directories: [ torch_gobject_inc ],
)

introspection_sources = [
  torch_gobject_introspectable_sources,
  torch_gobject_enums,
  torch_gobject_headers,
  torch_gobject_aten_generated_source,
  torch_gobject_aten_generated_header
]

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
