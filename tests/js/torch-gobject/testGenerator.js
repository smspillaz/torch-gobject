/*
 * tests/js/torch-gobject/testGenerator.js
 *
 * Tests for the JavaScript Binding to the Tensor Object.
 *
 * Copyright (C) 2021 Sam Spilsbury.
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

describe('TorchGenerator', function() {
  it('can be constructed', function() {
    let generator = new Torch.Generator({});
  });

  it('can be constructed with a seed', function() {
    let generator = new Torch.Generator({
      current_seed: 1
    });

    expect(generator.current_seed).toEqual(1);
  });

  it('can be constructed with a seed on an explicit device', function() {
    let generator = new Torch.Generator({
      device: new Torch.Device({
        type: Torch.DeviceType.CPU
      }),
      current_seed: 1
    });

    expect(generator.current_seed).toEqual(1);
  });

  it('by default, it is on the CPU', function() {
    let generator = new Torch.Generator({});

    expect(generator.device).not.toBeNull();
    expect(generator.device.type).toEqual(Torch.DeviceType.CPU);
  });

  it('by default, it has a seed of zero', function() {
    let generator = new Torch.Generator({});

    expect(generator.current_seed).toEqual(0);
  });

  it('can have its seed changed', function() {
    let generator = new Torch.Generator({});

    generator.set_current_seed(1);
    expect(generator.current_seed).toEqual(1);
  });

  it('gets a new seed when calling "seed"', function() {
    let generator = new Torch.Generator({});

    let newSeed = generator.seed()[1];
    expect(generator.current_seed).toEqual(newSeed);
  });

  it('has its state reset when the device is changed', function() {
    let generator = new Torch.Generator({});

    let oldSeed = generator.seed()[1];
    generator.device = new Torch.Device({
      type: Torch.DeviceType.CPU
    });
    expect(generator.current_seed).not.toEqual(oldSeed);
  });
});
