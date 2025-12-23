#pragma once

#include <list>
#include <vector>

struct my_phdr {
  uint32_t type;
  uint64_t addr;
  uint64_t memsz;
  const std::string *name_ref;
  // check if other is inside this item
  bool inside(const my_phdr &other) const {
    return (other.addr >= addr) && (other.addr + other.memsz) <= (addr + memsz);
  }
#if __cplusplus >= 202002L
  bool operator==(const my_phdr &other) = default;
#else
  bool operator==(const my_phdr &other) {
    return addr == other.addr && memsz == other.memsz;
  }
#endif
};

class rtmem_storage {
 public:
  rtmem_storage() = default;
  int read();
  const std::string *find(uint64_t addr);
  const my_phdr *check(uint64_t addr);
 protected:
  static int iterate_cb(struct dl_phdr_info *info, size_t size, void *data);
  std::list<std::string> m_names;
  std::vector<my_phdr> m_mem;
};
