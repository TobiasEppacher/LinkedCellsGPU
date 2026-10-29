#pragma once

#include "../util/DeviceDefinitions.hpp"
#include "../util/TypeDefinitions.hpp"

/**
 * Compile-time "interface" (concept) for mapping between 3D cell coordinates
 * and a 1D dense cell index.
 *
 * An indexing type must provide, callable on both host and device:
 *   LC_HOST_DEVICE_INLINE uint_type encode(const Vec3u& cell_coords) const;
 *   LC_HOST_DEVICE_INLINE Vec3u   decode(uint_type cell_idx) const;
 *
 * Implementations are plain, trivially-copyable structs (no virtual functions)
 * so they can be captured by value into device kernels -- runtime
 * virtual dispatch is not available on the device. Swap the CellIndex alias at
 * the bottom of this file to change the ordering everywhere at compile time.
 */

/**
 * Simple row-major flattening / unflattening of the cell indices.
 */
struct FlattenCellIndexing {
    Vec3u _num_cells_per_dim{};

    FlattenCellIndexing() = default;
    explicit FlattenCellIndexing(Vec3u num_cells_per_dim)
        : _num_cells_per_dim(num_cells_per_dim) {}

    LC_HOST_DEVICE_INLINE
    uint_type encode(const Vec3u& cell_coords) const {
        return cell_coords[0] * _num_cells_per_dim[1] * _num_cells_per_dim[2] +
               cell_coords[1] * _num_cells_per_dim[2] +
               cell_coords[2];
    }

    LC_HOST_DEVICE_INLINE
    Vec3u decode(uint_type cell_idx) const {
        const uint_type z = cell_idx % _num_cells_per_dim[2];
        const uint_type y = (cell_idx / _num_cells_per_dim[2]) % _num_cells_per_dim[1];
        const uint_type x = cell_idx / (_num_cells_per_dim[1] * _num_cells_per_dim[2]);
        return {x, y, z};
    }
};

/**
 * The cell ordering used throughout the containers and traversals.
 * Change this alias to switch the ordering at compile time.
 */
using CellIndex = FlattenCellIndexing;
