/*
 * tests/js/torch-gobject/testTensor.js
 *
 * Tests for the JavaScript Binding to the Tensor Object.
 *
 * Copyright (C) 2018 Sam Spilsbury.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

const { GLib, Torch } = imports.gi;

describe('TorchTensor', function() {
  it('can be constructed', function() {
    let tensor = new Torch.Tensor({});
  });
  
  it('can be constructed by Torch.zeros', function() {
    let tensor = Torch.zeros([1], new Torch.TensorOptions({}));
  });
  
  it('has default dimension of 1 when constructed with Torch.zeros', function() {
    let tensor = Torch.zeros([1], new Torch.TensorOptions({}));

    expect(tensor.get_dims()).toEqual([1]);
  });

  it('has a single zero constructed from Torch.zeros', function() {
    let opts = new Torch.TensorOptions({});
    let tensor = Torch.zeros([1], opts);

    expect(tensor.get_tensor_data().deep_unpack()).toEqual([0]);
  });

  /* Skipped, handling of array-like properties is currently
   * broken in gjs and pygi */
  xit('has default dimension prop of 0', function() {
    let tensor = new Torch.Tensor({});

    expect(tensor.dimensions).toEqual([0]);
  });

  /* Skipped, broken in gjs */
  xit('can be constructed with a dimension', function() {
    let tensor = new Torch.Tensor({ dimensions: [2] });

    expect(tensor.dimensions).toEqual([2]);
  });
});
