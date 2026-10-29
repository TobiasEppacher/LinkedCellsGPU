#pragma once

#include <algorithm>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../particle/Particle.hpp"
#include "../traversal/CellIndexing.hpp"
#include "TypeDefinitions.hpp"

/**
 * Writes particle snapshots as a numbered series of legacy-ASCII VTK files
 * (<directory>/<basename>_<step>.vtk). ParaView groups the numbered files
 * into a time series, one snapshot per timestep.
 */
class VTKWriter {
    std::filesystem::path _directory;
    std::string _basename;

   public:
    VTKWriter(std::filesystem::path directory, std::string basename)
        : _directory(std::move(directory)), _basename(std::move(basename)) {
        std::filesystem::create_directories(_directory);
    }

    void write(size_t step, std::vector<Particle> particles) const {
        // Stable id order so a point index maps to the same particle in every file
        std::sort(particles.begin(), particles.end(),
                  [](const Particle& a, const Particle& b) { return a.id < b.id; });

        const auto path = _directory / std::format("{}_{:04}.vtk", _basename, step);
        std::ofstream file(path);
        if (!file) {
            throw std::runtime_error("VTKWriter: could not open '" + path.string() + "'");
        }
        file.precision(std::numeric_limits<float_type>::max_digits10);

        const char* real_type = sizeof(float_type) == 8 ? "double" : "float";
        const char* id_type = sizeof(uint_type) == 8 ? "unsigned_long" : "unsigned_int";
        const size_t n = particles.size();

        file << "# vtk DataFile Version 3.0\n"
             << "Timestep " << step << "\n"
             << "ASCII\n"
             << "DATASET UNSTRUCTURED_GRID\n";

        file << "POINTS " << n << " " << real_type << "\n";
        for (const auto& p : particles) {
            file << p.r[0] << " " << p.r[1] << " " << p.r[2] << "\n";
        }

        // One VTK_VERTEX cell per particle so cell-based filters work
        file << "CELLS " << n << " " << 2 * n << "\n";
        for (size_t i = 0; i < n; ++i) file << "1 " << i << "\n";
        file << "CELL_TYPES " << n << "\n";
        for (size_t i = 0; i < n; ++i) file << "1\n";

        file << "POINT_DATA " << n << "\n";
        file << "SCALARS id " << id_type << " 1\n"
             << "LOOKUP_TABLE default\n";
        for (const auto& p : particles) file << p.id << "\n";

        file << "VECTORS velocity " << real_type << "\n";
        for (const auto& p : particles) {
            file << p.v[0] << " " << p.v[1] << " " << p.v[2] << "\n";
        }

        file << "VECTORS force " << real_type << "\n";
        for (const auto& p : particles) {
            file << p.f[0] << " " << p.f[1] << " " << p.f[2] << "\n";
        }
    }

    /**
     * Writes the static linked-cells grid as a single vtkImageData file
     * (<directory>/<basename>_grid.vti) for overlaying the cell boundaries on
     * the particle snapshots in ParaView. Carries two cell arrays: "halo"
     * (1 on the outer cell layer) and "cell_index" (the flat CellIndex id).
     */
    void writeCellGrid(const Vec3u& num_cells, const Vec3& cell_size, const Vec3& origin,
                       const CellIndex& indexing) const {
        const auto path = _directory / std::format("{}_grid.vti", _basename);
        std::ofstream file(path);
        if (!file) {
            throw std::runtime_error("VTKWriter: could not open '" + path.string() + "'");
        }
        file.precision(std::numeric_limits<float_type>::max_digits10);

        const char* id_type = sizeof(uint_type) == 8 ? "UInt64" : "UInt32";
        const std::string extent = std::format("0 {} 0 {} 0 {}", num_cells[0], num_cells[1], num_cells[2]);

        file << "<?xml version=\"1.0\"?>\n"
             << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\">\n"
             << "  <ImageData WholeExtent=\"" << extent << "\" Origin=\"" << origin[0] << " "
             << origin[1] << " " << origin[2] << "\" Spacing=\"" << cell_size[0] << " "
             << cell_size[1] << " " << cell_size[2] << "\">\n"
             << "    <Piece Extent=\"" << extent << "\">\n"
             << "      <CellData Scalars=\"halo\">\n";

        // vtkImageData cells run x-fastest, unlike the CellIndex ordering
        file << "        <DataArray type=\"UInt8\" Name=\"halo\" format=\"ascii\">\n";
        for (uint_type z = 0; z < num_cells[2]; ++z) {
            for (uint_type y = 0; y < num_cells[1]; ++y) {
                for (uint_type x = 0; x < num_cells[0]; ++x) {
                    const bool halo = x == 0 || y == 0 || z == 0 || x + 1 == num_cells[0] ||
                                      y + 1 == num_cells[1] || z + 1 == num_cells[2];
                    file << (halo ? 1 : 0) << " ";
                }
                file << "\n";
            }
        }
        file << "        </DataArray>\n";

        file << "        <DataArray type=\"" << id_type << "\" Name=\"cell_index\" format=\"ascii\">\n";
        for (uint_type z = 0; z < num_cells[2]; ++z) {
            for (uint_type y = 0; y < num_cells[1]; ++y) {
                for (uint_type x = 0; x < num_cells[0]; ++x) {
                    file << indexing.encode({x, y, z}) << " ";
                }
                file << "\n";
            }
        }
        file << "        </DataArray>\n";

        file << "      </CellData>\n"
             << "    </Piece>\n"
             << "  </ImageData>\n"
             << "</VTKFile>\n";
    }
};
