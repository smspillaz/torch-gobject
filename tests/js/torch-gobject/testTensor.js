/*
 * tests/js/torch-gobject/testTensor.js
 *
 * Tests for the JavaScript Binding to the Tensor Object.
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

  it('can do matrix multiplication', function() {
    let opts = new Torch.TensorOptions({});
    let matrix = Torch.eye([10], opts);
    let matrix2 = Torch.eye([10], opts);

    let result = matrix.mm(matrix2);
    let diag = matrix.diag(0);

    expect(diag.get_tensor_data().deep_unpack()).toEqual([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
  });

  it('can add scalars', function() {
    let opts = new Torch.TensorOptions({ dtype: GObject.TYPE_DOUBLE });
    let tensor = Torch.ones([2], opts);

    tensor.add_scalar_double_inplace(1.0, 1.0);

    expect(tensor.get_tensor_data().deep_unpack()).toEqual([2, 2]);
  });

  it('can check equality', function() {
    let opts = new Torch.TensorOptions({ dtype: GObject.TYPE_DOUBLE });
    let tensor = Torch.ones([2], opts);
    let tensor2 = Torch.ones([2], opts);

    let [status, equal] = tensor.equal(tensor2);

    expect(equal).toEqual(true);
  });

  it('can reshape', function() {
    let opts = new Torch.TensorOptions({ dtype: GObject.TYPE_DOUBLE });
    let tensor = Torch.linspace_double(0.0, 0.75, 4, opts);
    let tensor_reshaped = tensor.reshape([2, 2]);

    expect(tensor_reshaped.get_tensor_data().deep_unpack().map(v => v.deep_unpack())).toEqual([[0.0, 0.25], [0.5, 0.75]]);
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
