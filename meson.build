# /meson.build
#
# Toplevel meson build file for torch-gobject.
#
# Copyright (C) 2020 Sam Spilsbury
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

project('torch-gobject', ['c', 'cpp'],
        version: '0.0.0',
        default_options : ['cpp_std=c++17'],
        license: 'LGPL2+',
        meson_version: '>= 0.55.0')

#aten_dep = dependency('ATen', required: true, method: 'cmake')
torch_dep = dependency('Torch', required: true, method: 'cmake')

gtest_dep = dependency('gtest', required: false)
gtest_main_dep = dependency('gtest', main: true, required: false)
gmock_dep = dependency('gmock', required: false)

if not gtest_dep.found() or not gtest_main_dep.found() or not gmock_dep.found()
  gtest_project = subproject('googletest')
  gtest_dep = gtest_project.get_variable('gtest_dep')
  gtest_main_dep = gtest_project.get_variable('gtest_main_dep')
  gmock_dep = gtest_project.get_variable('gmock_dep')
endif

torch_gobject_inc = include_directories('.')
tests_inc = include_directories('tests')

subdir('torch-gobject')
subdir('tests')
