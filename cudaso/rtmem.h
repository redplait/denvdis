#pragma once

struct my_phdr {
  uint32_t type;
  uint64_t addr;
  uint64_t memsz;
  const std::string *name_ref;
};

class rtmem_storage {
 public:
  rtmem_storage() = default;
  int read();
  const std::string *find(uint64_t addr);
 protected:
  static int iterate_cb(struct dl_phdr_info *info, size_t size, void *data);
  std::list<std::string> m_names;
  std::vector<my_phdr> m_mem;
};
