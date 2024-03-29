# tests/js/torch-gobject/meson.build
#
# Meson build file for the torch-gobject library javascript tests.
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

# We need to also build the resources
torch_gobject_tests_resources_lib_introspectable_sources = files([
  'init.c',
  'init.h'
])
torch_gobject_tests_resources_lib_sources = torch_gobject_tests_resources_lib_introspectable_sources + torch_gobject_resources
torch_gobject_tests_resources_lib = shared_library(
  'torch-gobject-tests-resources',
  torch_gobject_tests_resources_lib_sources,
  dependencies: [ gio, glib, gobject ],
  install: false
)
torch_gobject_tests_resources_typelib  = gnome.generate_gir(
  torch_gobject_tests_resources_lib,
  extra_args: ['--warn-all'],
  identifier_prefix: 'TorchGObjectTestsResources',
  includes: ['GLib-2.0', 'GObject-2.0'],
  install: false,
  namespace: 'TorchGObjectTestsResources',
  nsversion: '0',
  sources: torch_gobject_tests_resources_lib_introspectable_sources,
  symbol_prefix: 'torch_gobject_tests_resources',
  fatal_warnings: true,
)[1]

javascript_tests = [
  'testDevice.js',
  'testDimname.js',
  'testGenerator.js',
  'testTensor.js'
]

gjs = find_program('gjs', required: false)
jasmine = find_program('jasmine', required: false)
include_path = '@0@:@1@'.format(meson.source_root(), meson.build_root())
built_library_path = join_paths(meson.build_root(), meson.project_name())
built_js_tests_library_path = join_paths(meson.current_build_dir())
library_paths = '@0@:@1@'.format(built_library_path, built_js_tests_library_path)

tests_environment = environment()
tests_environment.set('GJS_PATH', include_path)
tests_environment.prepend('GI_TYPELIB_PATH', library_paths)
tests_environment.prepend('LD_LIBRARY_PATH', library_paths)
tests_environment.prepend('DYLD_LIBRARY_PATH', library_paths)

subdir('nn')

if jasmine.found()
  foreach test_file : javascript_tests
    test(test_file,
         gjs,
         args: [
           jasmine.full_path(),
           '--verbose',
           join_paths(meson.current_source_dir(), test_file)
         ],
         env: tests_environment,
         depends: [torch_gobject_tests_resources_typelib])
  endforeach
endif
