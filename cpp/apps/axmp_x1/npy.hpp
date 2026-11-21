// SPDX-License-Identifier: BSD-3-Clause
// Minimal writer of NumPy .npy files (format version 1.0, C order, little-endian).
#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace x1 {

// descr is the dtype string ("<f8", "<u8", "<i4"); data holds the elements of the given shape in C order.
void write_npy(const std::filesystem::path& file, const std::string& descr, const std::vector<std::size_t>& shape,
               const void* data, std::size_t bytes);

}  // namespace x1
