/*
 * tests/js/torch-gobject/nn/options/testGeneratedOptions.js
 *
 * Tests for the JavaScript Binding to the Generated Options.
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

const ByteArray = imports.byteArray;
const { GLib, GObject, Gio, Torch, TorchGObjectTestsResources } = imports.gi;

TorchGObjectTestsResources.init();
const options_json_contents = Gio.File.new_for_uri('resource:///org/torch/gobject/nn/options/definitions/options.json').load_contents(null)[1];
const options_json = JSON.parse(ByteArray.toString(options_json_contents));

const DEFAULT_VALUES = {
  'TorchNNConvPaddingOptions *': meta => Torch.NNConvPaddingOptions.new(
    Torch.NNConvPaddingType.SPECIFIED,
    [1]
  ),
  'TorchTensor *': meta => Torch.zeros([0], new Torch.TensorOptions({ dtype: GObject.TYPE_FLOAT })),
  'TorchOptionalValue *': meta => Torch.OptionalValue['new_' + meta['type'].toLowerCase()](DEFAULT_VALUES[meta['type']]({})),
  'TorchNNDistanceFunction': meta => (first, second) => Torch.zeros([0], new Torch.TensorOptions({ dtype: GObject.TYPE_FLOAT })),
  'int32_t *': meta => (Array(meta.length instanceof Number ? meta.length : 2).fill(1)),
  'int64_t *': meta => (Array(meta.length instanceof Number ? meta.length : 2).fill(1)),
  'double *': meta => (Array(meta.length instanceof Number ? meta.length : 2).fill(2.0)),
  'gboolean': meta => false,
  'bool': meta => false,
  'int32_t': meta => 1,
  'int64_t': meta => 1,
  'double': meta => 2.0,
  'GType': meta => GObject.TYPE_INT
}

function getDefaultValueForType(c_type, meta) {
  if (c_type in DEFAULT_VALUES) {
    return DEFAULT_VALUES[c_type](meta);
  }

  return null;
}

describe('Generated Options', () => {
  options_json.forEach(opt_info => describe(opt_info.name, () => {
    it('can be constructed', () => {
      const name = opt_info.name;
      const defaults = opt_info.opts.map(arg_info => getDefaultValueForType(arg_info['c_type'], arg_info['meta'] || {}));
      const options = Torch[name].new.apply(
        null,
        defaults
      );
    });
  }));
});