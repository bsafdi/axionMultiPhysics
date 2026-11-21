// SPDX-License-Identifier: BSD-3-Clause
// Per-event seed words of axmp-transport.
//
// The words of event e of a chunk with seed s are the outputs 2e and 2e+1 of the SplitMix64 sequence that starts at
// the state s (Steele, Lea and Flood, "Fast splittable pseudorandom number generators", 2014; the finaliser below is
// the one of Vigna's public-domain splitmix64.c), each split into its low and high 32 bits, low half first.
#pragma once

#include <cstdint>

namespace axtr {

inline constexpr std::uint64_t kGolden = 0x9E3779B97F4A7C15ULL;

inline constexpr std::uint64_t splitmix64_mix(std::uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// Four 32-bit words (held in longs) for (chunk seed, event index in the chunk). Unsigned arithmetic wraps modulo 2^64.
inline void event_seed_words(std::uint64_t chunk_seed, std::uint64_t event, long words[4]) {
  const std::uint64_t o0 = splitmix64_mix(chunk_seed + (2 * event + 1) * kGolden);
  const std::uint64_t o1 = splitmix64_mix(chunk_seed + (2 * event + 2) * kGolden);
  words[0] = static_cast<long>(o0 & 0xFFFFFFFFULL);
  words[1] = static_cast<long>(o0 >> 32);
  words[2] = static_cast<long>(o1 & 0xFFFFFFFFULL);
  words[3] = static_cast<long>(o1 >> 32);
}

}  // namespace axtr
