/*
 * tests/js/torch-gobject/testTensor.js
 *
 * Tests for the JavaScript Binding to the Tensor Object.
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

const { GLib, GObject, Torch } = imports.gi;

describe('TorchTensor', function() {
  it('can be constructed', function() {
    let tensor = new Torch.Tensor({});
  });

  it('can be constructed with data', function() {
    let tensor = new Torch.Tensor({
      data: new GLib.Variant("v", new GLib.Variant("ad", [2.0, 2.0]))
    });

    expect(tensor.get_tensor_data().deep_unpack()).toEqual([2.0, 2.0]);
  });

  // Broken: Passing empty index to index_put_ is invalid
  it('can be constructed with single value', function() {
    let tensor = new Torch.Tensor({
      data: new GLib.Variant("v", new GLib.Variant("d", 2.0))
    });

    expect(tensor.get_tensor_data().deep_unpack()).toEqual(2.0);
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

  it('can be indexed by ints', function() {
    let opts = new Torch.TensorOptions({ dtype: GObject.TYPE_DOUBLE });
    let tensor = Torch.linspace_double(1.0, 8.0, 8, opts);
    let tensor_reshaped = tensor.reshape([2, 2, 2]);
    let indices = [Torch.Index.new_int (1), Torch.Index.new_int (1)];

    let tensor_indexed = tensor_reshaped.index_list(indices);

    expect(tensor_indexed.get_tensor_data().deep_unpack()).toEqual([7.0, 8.0]);
  });

  // Broken: get_tensor_data can't handle single value
  xit('can be indexed by ints to get a single vlaue', function() {
    let opts = new Torch.TensorOptions({ dtype: GObject.TYPE_DOUBLE });
    let tensor = Torch.linspace_double(1.0, 8.0, 8, opts);
    let tensor_reshaped = tensor.reshape([2, 2, 2]);
    let indices = [Torch.Index.new_int (1), Torch.Index.new_int (1), Torch.Index.new_int (1)];

    let tensor_indexed = tensor_reshaped.index_list(indices);

    expect(tensor_indexed.get_tensor_data().deep_unpack()).toEqual(8.0);
  });

  it('can put a single value given by ints index', function() {
    let opts = new Torch.TensorOptions({ dtype: GObject.TYPE_DOUBLE });
    let tensor = Torch.linspace_double(1.0, 8.0, 8, opts);
    let tensor_reshaped = tensor.reshape([2, 2, 2]);
    let put_indices = [Torch.Index.new_int (1), Torch.Index.new_int (1), Torch.Index.new_int (0)];

    tensor_reshaped.index_list_put_inplace_double (put_indices, 10.0);

    let indices = [Torch.Index.new_int (1), Torch.Index.new_int (1)];
    let tensor_indexed = tensor_reshaped.index_list(indices);

    expect(tensor_indexed.get_tensor_data().deep_unpack()).toEqual([10.0, 8.0]);
  });

  it('can put a tensor given by none on last index', function() {
    let opts = new Torch.TensorOptions({ dtype: GObject.TYPE_DOUBLE });
    let tensor = Torch.linspace_double(1.0, 8.0, 8, opts);
    let tensor_reshaped = tensor.reshape([2, 2, 2]);
    let put_indices = [Torch.Index.new_int (1), Torch.Index.new_int (1), Torch.Index.new_none ()];

    tensor_reshaped.index_list_put_inplace_tensor(
      put_indices,
      new Torch.Tensor({ data: new GLib.Variant("v", new GLib.Variant("ad", [2.0, 2.0])) })
    );

    let indices = [Torch.Index.new_int (1), Torch.Index.new_int (1)];
    let tensor_indexed = tensor_reshaped.index_list(indices);

    expect(tensor_indexed.get_tensor_data().deep_unpack()).toEqual([2.0, 2.0]);
  });

  it('can be indexed by a slice', function() {
    let opts = new Torch.TensorOptions({ dtype: GObject.TYPE_DOUBLE });
    let tensor = Torch.linspace_double(1.0, 10.0, 10, opts);
    let tensor_reshaped = tensor.reshape([2, 5]);
    let indices = [Torch.Index.new_int (1), Torch.Index.new_range (1, 4, 1)];

    let tensor_indexed = tensor_reshaped.index_list(indices);

    expect(tensor_indexed.get_tensor_data().deep_unpack()).toEqual([7, 8, 9]);
  });

  /* Skipped, handling of GPtrArray broken on gjs */
  xit('can be array-indexed by ints', function() {
    let opts = new Torch.TensorOptions({ dtype: GObject.TYPE_DOUBLE });
    let tensor = Torch.linspace_double(1.0, 8.0, 8, opts);
    let tensor_reshaped = tensor.reshape([2, 2, 2]);
    let indices = [Torch.Index.new_int (1), Torch.Index.new_int (1)];

    let tensor_indexed = tensor_reshaped.index_array(indices);

    expect(tensor_indexed.get_tensor_data().deep_unpack()).toEqual([7.0, 8.0]);
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
