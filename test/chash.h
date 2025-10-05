#pragma once

#include <vector>
#include <string>

bool hash_file(const std::string &fname, const char *algo, std::vector<uint8_t> &md);