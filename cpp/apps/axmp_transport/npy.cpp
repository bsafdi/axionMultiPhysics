// SPDX-License-Identifier: BSD-3-Clause
#include "npy.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <stdexcept>

namespace axtr {

namespace {

constexpr std::array<std::uint32_t, 64> kK = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

inline std::uint32_t rotr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

void compress(std::array<std::uint32_t, 8>& h, const unsigned char* p) {
  std::uint32_t w[64];
  for (int i = 0; i < 16; ++i)
    w[i] = (std::uint32_t(p[4 * i]) << 24) | (std::uint32_t(p[4 * i + 1]) << 16) |
           (std::uint32_t(p[4 * i + 2]) << 8) | std::uint32_t(p[4 * i + 3]);
  for (int i = 16; i < 64; ++i) {
    const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
    const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
  for (int i = 0; i < 64; ++i) {
    const std::uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
    const std::uint32_t ch = (e & f) ^ (~e & g);
    const std::uint32_t t1 = hh + S1 + ch + kK[i] + w[i];
    const std::uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
    const std::uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
    const std::uint32_t t2 = S0 + mj;
    hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
  }
  h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

class Sha256 {
 public:
  void update(const unsigned char* p, std::size_t n) {
    total_ += n;
    while (n) {
      const std::size_t take = std::min<std::size_t>(n, 64 - fill_);
      std::memcpy(buf_ + fill_, p, take);
      fill_ += take; p += take; n -= take;
      if (fill_ == 64) { compress(h_, buf_); fill_ = 0; }
    }
  }
  std::string hex() {
    const std::uint64_t bits = total_ * 8;
    unsigned char pad[72] = {0x80};
    const std::size_t padlen = (fill_ < 56) ? 56 - fill_ : 120 - fill_;
    update(pad, padlen);
    unsigned char len[8];
    for (int i = 0; i < 8; ++i) len[i] = static_cast<unsigned char>(bits >> (56 - 8 * i));
    update(len, 8);
    static const char* digits = "0123456789abcdef";
    std::string out;
    for (std::uint32_t v : h_)
      for (int s = 28; s >= 0; s -= 4) out.push_back(digits[(v >> s) & 0xF]);
    return out;
  }

 private:
  std::array<std::uint32_t, 8> h_ = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  unsigned char buf_[64] = {};
  std::size_t fill_ = 0;
  std::uint64_t total_ = 0;
};

}  // namespace

std::string sha256_hex(const unsigned char* data, std::size_t n) {
  Sha256 s;
  s.update(data, n);
  return s.hex();
}

std::string write_npy(const std::filesystem::path& file, const std::string& descr,
                      const std::vector<std::size_t>& shape, const void* data, std::size_t bytes) {
#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "write_npy writes host byte order and declares little endian"
#endif
  std::string shp = "(";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    shp += std::to_string(shape[i]);
    if (shape.size() == 1 || i + 1 < shape.size()) shp += ",";
    if (i + 1 < shape.size()) shp += " ";
  }
  shp += ")";
  std::string header = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shp + ", }";
  // magic (6) + version (2) + length field (2) + header + '\n' is a multiple of 64
  const std::size_t unpadded = 10 + header.size() + 1;
  header.append((64 - unpadded % 64) % 64, ' ');
  header.push_back('\n');
  std::string head = "\x93NUMPY";
  head.push_back(1);
  head.push_back(0);
  head.push_back(static_cast<char>(header.size() & 0xFF));
  head.push_back(static_cast<char>((header.size() >> 8) & 0xFF));
  head += header;

  Sha256 sha;
  sha.update(reinterpret_cast<const unsigned char*>(head.data()), head.size());
  sha.update(static_cast<const unsigned char*>(data), bytes);
  std::ofstream f(file, std::ios::binary | std::ios::trunc);
  f.write(head.data(), static_cast<std::streamsize>(head.size()));
  f.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  f.flush();
  if (!f) throw std::runtime_error("cannot write " + file.string());
  return sha.hex();
}

NpyDoubles read_npy_doubles(const std::filesystem::path& file) {
  auto fail = [&](const std::string& what) { throw std::runtime_error(file.string() + ": " + what); };
  std::ifstream f(file, std::ios::binary);
  if (!f) fail("cannot be read");
  std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (bytes.size() < 10 || bytes.compare(0, 6, "\x93NUMPY") != 0) fail("is not an NPY file");
  const int major = static_cast<unsigned char>(bytes[6]);
  std::size_t hlen = 0, start = 0;
  if (major == 1) {
    hlen = static_cast<unsigned char>(bytes[8]) | (std::size_t(static_cast<unsigned char>(bytes[9])) << 8);
    start = 10;
  } else if (major == 2 || major == 3) {
    if (bytes.size() < 12) fail("is truncated");
    for (int i = 0; i < 4; ++i) hlen |= std::size_t(static_cast<unsigned char>(bytes[8 + i])) << (8 * i);
    start = 12;
  } else {
    fail("has an NPY format version this program does not read");
  }
  if (bytes.size() < start + hlen) fail("is truncated");
  const std::string header = bytes.substr(start, hlen);
  auto value_of = [&](const std::string& key) {
    const std::size_t k = header.find("'" + key + "'");
    if (k == std::string::npos) fail("its header has no '" + key + "'");
    std::size_t v = header.find(':', k);
    if (v == std::string::npos) fail("its header is malformed");
    ++v;
    while (v < header.size() && header[v] == ' ') ++v;
    return v;
  };
  {
    const std::size_t v = value_of("descr");
    if (header.compare(v, 6, "'<f8'") != 0 && header.compare(v, 6, "'<f8',") != 0 &&
        header.compare(v, 5, "'<f8'") != 0)
      fail("is not little-endian float64 ('<f8')");
  }
  {
    const std::size_t v = value_of("fortran_order");
    if (header.compare(v, 5, "False") != 0) fail("is in Fortran order");
  }
  NpyDoubles out;
  {
    std::size_t v = value_of("shape");
    if (v >= header.size() || header[v] != '(') fail("its shape is malformed");
    const std::size_t e = header.find(')', v);
    if (e == std::string::npos) fail("its shape is malformed");
    std::string inner = header.substr(v + 1, e - v - 1);
    std::size_t pos = 0;
    while (pos < inner.size()) {
      while (pos < inner.size() && (inner[pos] == ' ' || inner[pos] == ',')) ++pos;
      if (pos >= inner.size()) break;
      std::size_t end = pos;
      while (end < inner.size() && inner[end] >= '0' && inner[end] <= '9') ++end;
      if (end == pos) fail("its shape is malformed");
      out.shape.push_back(static_cast<std::size_t>(std::stoull(inner.substr(pos, end - pos))));
      pos = end;
    }
  }
  std::size_t n = 1;
  for (std::size_t s : out.shape) n *= s;
  const std::size_t off = start + hlen;
  if (bytes.size() != off + 8 * n) fail("holds " + std::to_string(bytes.size() - off) + " data bytes, not " +
                                        std::to_string(8 * n));
  out.data.resize(n);
  if (n) std::memcpy(out.data.data(), bytes.data() + off, 8 * n);
  out.sha256 = sha256_hex(reinterpret_cast<const unsigned char*>(bytes.data()), bytes.size());
  return out;
}

}  // namespace axtr
