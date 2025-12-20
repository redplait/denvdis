#include "types.h"
#include <link.h>
#include <string>
#include <vector>
#include <list>
#include <algorithm>
#include "rtmem.h"

int rtmem_storage::iterate_cb(struct dl_phdr_info *info, size_t size, void *data)
{
  rtmem_storage *rs = (rtmem_storage *)data;
  // check if we add such name
  if ( !rs->m_names.empty() ) {
    if ( rs->m_names.back() != info->dlpi_name )
      rs->m_names.push_back(info->dlpi_name);
  } else {
    rs->m_names.push_back(info->dlpi_name);
  }
  const std::string *mod_name = &rs->m_names.back();
  // from https://man7.org/linux/man-pages/man3/dl_iterate_phdr.3.html
  for (size_t j = 0; j < info->dlpi_phnum; j++) {
    rs->m_mem.push_back({ info->dlpi_phdr[j].p_type, info->dlpi_addr + info->dlpi_phdr[j].p_vaddr, info->dlpi_phdr[j].p_memsz, mod_name });
  }
  return 0;
}

int rtmem_storage::read()
{
  dl_iterate_phdr( &iterate_cb, this );
  // finalize
  std::sort(m_mem.begin(), m_mem.end(), [](const my_phdr &a, const my_phdr &b) { return a.addr < b.addr; });
  return !m_mem.empty();
}

const std::string *rtmem_storage::find(uint64_t addr) {
  if ( m_mem.empty() ) return nullptr;
  const auto it = std::lower_bound(m_mem.begin(), m_mem.end(), addr,
   [](auto &what, uint64_t off) { return what.addr < off; });
  if ( it == m_mem.end() ) return nullptr;
  // check if addr really inside found region
  if ( addr >= it->addr && addr < (it->addr + it->memsz) )
   return it->name_ref;
  return nullptr;
};