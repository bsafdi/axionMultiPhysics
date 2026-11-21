// SPDX-License-Identifier: BSD-3-Clause
#include "npy.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace x1 {

void write_npy(const std::filesystem::path& file, const std::string& descr, const std::vector<std::size_t>& shape,
               const void* data, std::size_t bytes) {
  std::ostringstream dict;
  dict << "{'descr': '" << descr << "', 'fortran_order': False, 'shape': (";
  for (std::size_t k = 0; k < shape.size(); ++k) dict << (k ? ", " : "") << shape[k];
  if (shape.size() == 1) dict << ",";
  dict << "), }";
  std::string header = dict.str();
  // magic (6) + version (2) + length (2) + header + newline must be a multiple of 64
  const std::size_t total = 10 + header.size() + 1;
  header.append((64 - total % 64) % 64, ' ');
  header.push_back('\n');
  std::ofstream f(file, std::ios::binary | std::ios::trunc);
  const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
  f.write(reinterpret_cast<const char*>(magic), 8);
  const unsigned char len[2] = {static_cast<unsigned char>(header.size() & 0xFF),
                                static_cast<unsigned char>((header.size() >> 8) & 0xFF)};
  f.write(reinterpret_cast<const char*>(len), 2);
  f.write(header.data(), static_cast<std::streamsize>(header.size()));
  if (bytes) f.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  f.flush();
  if (!f) throw std::runtime_error("cannot write " + file.string());
}

}  // namespace x1
