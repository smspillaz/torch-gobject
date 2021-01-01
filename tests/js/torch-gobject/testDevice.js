/*
 * tests/js/torch-gobject/testDevice.js
 *
 * Tests for the JavaScript Binding to the Device Object.
 *
 * Copyright (C) 2018 Sam Spilsbury.
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

const { GLib, GObject, Torch } = imports.gi;

describe('TorchDevice', function() {
  it('can be default-constructed', function() {
    let device = new Torch.Device({});
  });
});
