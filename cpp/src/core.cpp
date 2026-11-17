#include "axionmultiphysics/core.hpp"

#include <iostream>

namespace axmp {

int& log_level() {
  static int level = 1;
  return level;
}

void log(int level, const std::string& msg) {
  if (level <= log_level()) std::cerr << "[axmp] " << msg << std::endl;
}

}  // namespace axmp
