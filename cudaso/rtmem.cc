#include "types.h"
#include <link.h>
#include <string>
#include <algorithm>
#include "rtmem.h"
#include <list>

extern int opt_d;

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
  // there are 2 problems
  // 1) items from dlpi_phdr are unsorted
  // 2) there are lots of intersected areas
  // so first sort them in sorted array (using dirty hack to store only pointers)
  // and then cache added areas in ord list and check for overlapping
  std::list<my_phdr> ord;
  using T_sorted = std::remove_reference<decltype(info->dlpi_phdr[0])>::type;
  std::vector<T_sorted *> sorted;
  for (size_t j = 0; j < info->dlpi_phnum; j++) {
    // skip items with zero length
    if ( !info->dlpi_phdr[j].p_memsz ) continue;
    sorted.push_back(&info->dlpi_phdr[j]);
  }
  // sort
  std::sort(sorted.begin(), sorted.end(), [](const T_sorted *a, const T_sorted *b) { return a->p_vaddr < b->p_vaddr; });
  for ( auto s: sorted ) {
    // make new my_phdr
    auto curr_addr = info->dlpi_addr + s->p_vaddr;
    my_phdr tmp{ s->p_type, curr_addr, s->p_memsz, mod_name };
    bool skip = false;
    for ( auto &ord_it: ord ) {
     // check if we already have such start address
      if ( ord_it.addr == tmp.addr ) {
        if ( s->p_memsz > ord_it.memsz ) {
          ord_it.memsz = s->p_memsz;
          ord_it.type = s->p_type;
          skip = true;
          break;
        }
      }
      // check for overlapping
      if ( ord_it.inside(tmp) ) {
        skip = true;
        break;
      }
      if ( tmp.inside(ord_it) ) {
        ord_it = std::move(tmp);
        skip = true;
        break;
      }
    }
    if ( opt_d )
      printf("skip %d %lX-%lX size %lX %s\n", skip, tmp.addr, tmp.addr + tmp.memsz, tmp.memsz, tmp.name_ref->c_str());
    if ( !skip )
     ord.push_back( std::move(tmp) );
  }
  std::copy(ord.begin(), ord.end(), std::back_inserter(rs->m_mem));
  return 0;
}

int rtmem_storage::read()
{
  dl_iterate_phdr( &iterate_cb, this );
  // finalize
  std::sort(m_mem.begin(), m_mem.end(), [](const my_phdr &a, const my_phdr &b) { return a.addr < b.addr; });
  // dump
  if ( opt_d )
    for ( const auto &it: m_mem ) {
      if ( it.name_ref->empty() ) continue;
      printf("%lX-%lX size %lX - %X %s\n", it.addr, it.addr + it.memsz, it.memsz, it.type, it.name_ref->c_str());
    }
  return !m_mem.empty();
}

// from https://en.cppreference.com/w/cpp/algorithm/lower_bound.html
// returns true if the first argument is ordered before the second
static bool for_lower_bound(const my_phdr &what, uint64_t off) {
  return ( what.addr + what.memsz < off );
}

const std::string *rtmem_storage::find(uint64_t addr) {
  if ( m_mem.empty() ) return nullptr;
  const auto it = std::lower_bound(m_mem.begin(), m_mem.end(), addr, for_lower_bound);
  if ( it == m_mem.end() ) return nullptr;
  // check if addr really inside found region
  if ( addr >= it->addr && addr < (it->addr + it->memsz) )
   return it->name_ref;
  return nullptr;
}

const my_phdr *rtmem_storage::check(uint64_t addr) {
  if ( m_mem.empty() ) return nullptr;
  auto it = std::lower_bound(m_mem.begin(), m_mem.end(), addr, for_lower_bound);
  if ( it == m_mem.end() ) return nullptr;
#ifdef DEBUG
printf("check found base %lX - %lX\n", it->addr, it->addr + it->memsz);
#endif
  // check if addr really inside found region
  if ( addr >= it->addr && addr < (it->addr + it->memsz) )
   return &*it;
  return nullptr;
}