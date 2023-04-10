# /torch-gobject/nn/options/meson.build
#
# Build the libtorch-gobject library, nn options components.
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

torch_gobject_nn_options_headers_subdir = join_paths(torch_gobject_nn_headers_subdir, 'options')

torch_nn_options_defs = join_paths('definitions', 'options.json')
torch_gobject_nn_options_codegen_prog = join_paths(torch_gobject_codegen_dir, 'codegen-nn-options.py')
torch_gobject_nn_options_generated_header = custom_target('gen-torch-nn-options-header',
                                                          input : files([torch_gobject_nn_options_codegen_prog, torch_nn_options_defs]),
                                                          output : ['torch-nn-options-generated.h'],
                                                          command : [python_installation, '@INPUT@', '--header', '--output', '@OUTPUT0@'],
                                                          install: true,
                                                          install_dir: get_option('includedir') / torch_gobject_headers_subdir,
                                                          depend_files : torch_gobject_codegen_lib_files)

torch_gobject_nn_options_generated_source = custom_target('gen-torch-nn-options-source',
                                                          input : files([torch_gobject_nn_options_codegen_prog, torch_nn_options_defs]),
                                                          output : ['torch-nn-options-generated.cpp'],
                                                          command : [python_installation, '@INPUT@', '--source', '--output', '@OUTPUT0@'],
                                                          depend_files : torch_gobject_codegen_lib_files)
torch_gobject_nn_options_generated_introspectable_source = custom_target('gen-torch-nn-options-introspectable-source',
                                                          input : files([torch_gobject_nn_options_codegen_prog, torch_nn_options_defs]),
                                                          output : ['torch-nn-options-generated-introspectable.cpp'],
                                                          command : [python_installation, '@INPUT@', '--introspectable-source', '--output', '@OUTPUT0@'],
                                                          depend_files : torch_gobject_codegen_lib_files)


torch_gobject_nn_options_enums_headers = files([
  'torch-nn-conv-padding-mode.h',
  'torch-nn-conv-padding-type.h',
  'torch-nn-embedding-bag-mode.h',
  'torch-nn-grid-sample-mode.h',
  'torch-nn-grid-sample-padding-mode.h',
  'torch-nn-interpolate-mode.h',
  'torch-nn-loss-reduction-mode.h',
  'torch-nn-pad-mode.h',
  'torch-nn-rnn-nonlinearity-type.h',
  'torch-nn-transformer-activation-type.h',
  'torch-nn-upsample-mode.h',
])

torch_gobject_nn_options_enums = gnome.mkenums_simple(
  'torch-gobject-nn-options-enums',
  sources: torch_gobject_nn_options_enums_headers,
  install_header: true,
  install_dir: get_option('includedir') / torch_gobject_nn_options_headers_subdir
)

torch_gobject_nn_options_headers = files([
  'torch-nn-conv-padding-options.h',
  'torch-nn-namedshape-element.h',
]) + torch_gobject_toplevel_enums_headers
torch_gobject_nn_options_generated_headers = [
    torch_gobject_nn_options_generated_header
]
torch_gobject_nn_options_introspectable_sources = files([
  'torch-nn-conv-padding-mode.cpp',
  'torch-nn-conv-padding-options.c',
  'torch-nn-conv-padding-type.c',
  'torch-nn-embedding-bag-mode.cpp',
  'torch-nn-grid-sample-mode.cpp',
  'torch-nn-grid-sample-padding-mode.cpp',
  'torch-nn-interpolate-mode.cpp',
  'torch-nn-loss-reduction-mode.cpp',
  'torch-nn-namedshape-element.cpp',
  'torch-nn-pad-mode.cpp',
  'torch-nn-rnn-nonlinearity-type.cpp',
  'torch-nn-transformer-activation-type.cpp',
  'torch-nn-upsample-mode.cpp',
]) + [torch_gobject_nn_options_generated_introspectable_source]
torch_gobject_nn_options_private_sources = files([
]) + [torch_gobject_nn_options_generated_source]
torch_gobject_nn_options_private_headers = files([
  'torch-nn-conv-padding-mode-internal.h',
  'torch-nn-conv-padding-options-internal.h',
  'torch-nn-embedding-bag-mode-internal.h',
  'torch-nn-grid-sample-mode-internal.h',
  'torch-nn-grid-sample-padding-mode-internal.h',
  'torch-nn-interpolate-mode-internal.h',
  'torch-nn-loss-reduction-mode-internal.h',
  'torch-nn-namedshape-element-internal.h',
  'torch-nn-pad-mode-internal.h',
  'torch-nn-rnn-nonlinearity-type-internal.h',
  'torch-nn-transformer-activation-type-internal.h',
  'torch-nn-upsample-mode-internal.h',
])

torch_gobject_introspectable_sources += torch_gobject_nn_options_introspectable_sources
torch_gobject_private_sources += torch_gobject_nn_options_private_sources
torch_gobject_headers += torch_gobject_nn_options_headers
torch_gobject_generated_headers += torch_gobject_nn_options_generated_headers
torch_gobject_private_headers += torch_gobject_nn_options_headers
torch_gobject_enums += torch_gobject_nn_options_enums
torch_gobject_include_directories += [include_directories('.')]

install_headers(
  torch_gobject_nn_options_headers,
  subdir: torch_gobject_nn_options_headers_subdir
)

gnome = import('gnome')

generated_options_gresources = gnome.compile_resources(
  'torch_gobject_generated_options_tests_resources',
  'torch-gobject-generated-options-tests-resources.gresources.xml',
  build_by_default: true,
  c_name: 'torch_gobject_tests_resources'
)

torch_gobject_resources += generated_options_gresources