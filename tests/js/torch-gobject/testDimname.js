/*
 * tests/js/torch-gobject/testTensor.js
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

describe('TorchDimname', function() {
  it('can be constructed', function() {
    let dimname = new Torch.Dimname({});
  });

  it('by default, it is a wildcard', function() {
    let dimname = new Torch.Dimname({});

    expect(dimname.symbol_name).toEqual('*');
    expect(dimname.symbol_type).toEqual(Torch.DimnameType.WILDCARD);
  });

  it('can be constructed with an explicit name and gets type basic', function() {
    let dimname = new Torch.Dimname({
        symbol_name: 'F'
    });

    expect(dimname.symbol_name).toEqual('F');
    expect(dimname.symbol_type).toEqual(Torch.DimnameType.BASIC);
  });

  it('can be constructed as an explicit wildcard', function() {
    let dimname = new Torch.Dimname({
        symbol_type: Torch.DimnameType.WILDCARD
    });

    expect(dimname.symbol_name).toEqual('*');
    expect(dimname.symbol_type).toEqual(Torch.DimnameType.WILDCARD);
  });

  it('matches another dimname with the same name', function() {
    let dimname = new Torch.Dimname({
        symbol_name: 'F'
    });
    let other = new Torch.Dimname({
        symbol_name: 'F'
    });


    expect(dimname.matches(other)[1]).toBeTruthy();
  });

  it('matches another dimname with a wildcard', function() {
    let dimname = new Torch.Dimname({
        symbol_name: 'F'
    });
    let other = new Torch.Dimname({
        symbol_type: Torch.DimnameType.WILDCARD
    });


    expect(dimname.matches(other)[1]).toBeTruthy();
    expect(other.matches(dimname)[1]).toBeTruthy();
  });

  it('does not match a dimname of a different name', function() {
    let dimname = new Torch.Dimname({
        symbol_name: 'F'
    });
    let other = new Torch.Dimname({
        symbol_name: 'A'
    });

    expect(dimname.matches(other)[1]).toBeFalsy();
  });

  it('unifies with a dimname of the same name', function() {
    let dimname = new Torch.Dimname({
        symbol_name: 'F'
    });
    let other = new Torch.Dimname({
        symbol_name: 'F'
    });

    expect(dimname.unify(other)[1].symbol_name).toEqual('F');
  });

  it('does not unify with a dimname of a different name', function() {
    let dimname = new Torch.Dimname({
        symbol_name: 'F'
    });
    let other = new Torch.Dimname({
        symbol_name: 'A'
    });

    expect(dimname.unify(other)[1]).toBe(null);
  });

  it('unifies with a wildcard, replacing it', function() {
    let dimname = new Torch.Dimname({
        symbol_name: 'F'
    });
    let other = new Torch.Dimname({
        symbol_type: Torch.DimnameType.WILDCARD
    });

    expect(dimname.unify(other)[1].symbol_name).toEqual('F');
    expect(other.unify(dimname)[1].symbol_name).toEqual('F');
  });

  it('wildcards unify with each other', function() {
    let dimname = new Torch.Dimname({
        symbol_type: Torch.DimnameType.WILDCARD
    });
    let other = new Torch.Dimname({
        symbol_type: Torch.DimnameType.WILDCARD
    });

    expect(dimname.unify(other)[1].symbol_type).toEqual(Torch.DimnameType.WILDCARD);
  });
});
