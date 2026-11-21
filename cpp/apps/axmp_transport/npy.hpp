// SPDX-License-Identifier: BSD-3-Clause
// NPY (format version 1.0, little endian, C order) files and SHA-256, with no dependency beyond the C++ library.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace axtr {

// Hex SHA-256 (FIPS 180-4) of a byte string.
std::string sha256_hex(const unsigned char* data, std::size_t n);

// Writes `data` (bytes = product(shape) * element size, host order, which must be little endian) as an NPY file of
// numpy dtype string `descr` (for example "<u8") and returns the hex SHA-256 of the file's bytes.
// Throws std::runtime_error when the file cannot be written.
std::string write_npy(const std::filesystem::path& file, const std::string& descr,
                      const std::vector<std::size_t>& shape, const void* data, std::size_t bytes);

// A little-endian float64 array in C order read from an NPY file (format 1.0 to 3.0), and the hex SHA-256 of the
// file's bytes.
struct NpyDoubles {
  std::vector<std::size_t> shape;
  std::vector<double> data;
  std::string sha256;
};

// Reads an NPY file of dtype '<f8' in C order. Throws std::runtime_error naming the file for anything else.
NpyDoubles read_npy_doubles(const std::filesystem::path& file);

}  // namespace axtr
