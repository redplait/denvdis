#include "ced_base.h"

std::regex CEd_base::rs_digits("^\\d+");

int CEd_base::flush_buf()
{
  if ( !m_cubin_fp || !block_dirty ) return 1;
  fseek(m_cubin_fp, m_buf_off, SEEK_SET);
  if ( opt_h ) HexDump(m_out, buf, block_size);
  if ( 1 != fwrite(buf, block_size, 1, m_cubin_fp) ) {
    Err("fwrite at %lX failed, error %d (%s)\n", m_buf_off, errno, strerror(errno));
    return 0;
  }
  flush_cnt++;
  block_dirty = 0;
  return 1;
}

int CEd_base::setup_labels(int idx)
{
  if ( idx == (int)m_idx ) return 1;
  // get appr attributes section
  m_labels.clear();
  auto si = m_code_sects.find(idx);
  if ( si == m_code_sects.end() || !si->second ) return 0;
  section *sec = m_reader->sections[si->second];
  // similar to nv_dis::_parse_attrs
  const char *data = sec->get_data();
  const char *start = data, *end = data + sec->get_size();
  while( data < end )
  {
    if ( end - data < 2 ) {
      Err("bad attrs data. section %d\n", si->second);
      return 0;
    }
    char format = data[0];
    char attr = data[1];
    unsigned short a_len;
    int ltype = 0;
    switch (format)
    {
      case 1: data += 2;
        // check align
        if ( (data - start) & 0x3 ) data += 4 - ((data - start) & 0x3);
        break;
      case 2:
        data += 3;
        // check align
        if ( (data - start) & 0x1 ) data++;
       break;
      case 3:
        data += 4;
       break;
      case 4:
        a_len = *(unsigned short *)(data + 2);
        if ( attr == 0x28 ) // EIATTR_COOP_GROUP_INSTR_OFFSETS
          ltype = NVLType::Coop_grp;
        else if ( attr == 0x1c ) // EIATTR_EXIT_INSTR_OFFSETS
          ltype = NVLType::Exit;
        else if ( attr == 0x1d ) // EIATTR_S2RCTAID_INSTR_OFFSETS
          ltype = NVLType::S2Rctaid;
        else if ( attr == 0x25 ) // EIATTR_LD_CACHEMOD_INSTR_OFFSETS
          ltype = NVLType::Ld_cachemode;
        else if ( attr == 0x31 ) // EIATTR_INT_WARP_WIDE_INSTR_OFFSETS
          ltype = NVLType::Warp_wide;
        else if ( attr == 0x39 ) // EIATTR_MBARRIER_INSTR_OFFSETS
          ltype = NVLType::MBarier;
        else if ( attr == 0x47 ) // EIATTR_SW_WAR_MEMBAR_SYS_INSTR_OFFSETS
          ltype = NVLType::War_membar;
        // read offsets
        if ( ltype ) {
          fill_eaddrs(&m_labels, ltype, data, a_len);
        } else if ( attr == 0x34 ) { // EIATTR_INDIRECT_BRANCH_TARGETS
          parse_branch_targets(data + 4, a_len, [&](const one_indirect_branch &ibt) {
            for ( auto l: ibt.labels ) {
              m_labels[l] = NVLType::Ind_BT;
            }
            // store addr with type 0
            m_labels[ibt.addr] = 0;
          });
        }
        data += 4 + a_len;
        break;
      default: Err("unknown format %d, section %d off %lX (%s)\n",
        format, idx, data - start, sec->get_name().c_str());
         return 0;
    }
  }
  return 1;
}

int CEd_base::prepare(const char *fn)
{
  if ( !init_guts() ) return 0;
  n_sec = m_reader->sections.size();
  // iterate on sections to collect section with attributes
  std::unordered_map<int, int> attrs; // key - link section, value - index of section with attributes
  for ( Elf_Half i = 0; i < n_sec; ++i ) {
    section *sec = m_reader->sections[i];
    auto st = sec->get_type();
    if ( st == SHT_NOBITS || !sec->get_size() ) continue;
    if ( st == 0x70000000 ) attrs[sec->get_info()] = i;
  }
  // iterate on sections to collect code sections
  for ( Elf_Half i = 0; i < n_sec; ++i ) {
   section *sec = m_reader->sections[i];
   if ( sec->get_type() == SHT_NOBITS || !sec->get_size() ) continue;
   auto sname = sec->get_name();
   if ( !strncmp(sname.c_str(), ".text.", 6) ) {
     auto ai = attrs.find(i);
     if ( ai == attrs.end() )
       m_code_sects[i] = 0;
     else
       m_code_sects[i] = ai->second;
     m_named_cs[sname] = i;
     if ( opt_v ) {
       if ( ai == attrs.end() )
         printf("section %d: %s, size %lX\n", i, sname.c_str(), sec->get_size());
       else
         printf("section %d: %s, size %lX attrs in section %d\n", i, sname.c_str(), sec->get_size(), ai->second);
     }
   }
  }
  if ( m_code_sects.empty() ) {
   Err("cannot find code sections in %s\n", fn);
   return 0;
  }
  // init block
  switch(m_width) {
    case 64: mask_size = 6; block_size = 64; break;
    case 88: mask_size = 6; block_size = 32; break;
    case 128: mask_size = 7; block_size = 16; break;
    default:
     Err("Unknown width %d\n", m_width);
     return 0;
  }
  mask_size -= 3; // align to byte
  memset(buf, 0, block_size);
  fill_rels();
  // open file
  m_cubin_fp = fopen(fn, "r+b");
  if ( !m_cubin_fp ) {
    Err("Cannot open %s, error %d (%s)\n", fn, errno, strerror(errno));
    return 0;
  }
  // lets try to find NOP
  auto il = m_dis->get_instrs();
  auto nop = std::lower_bound(il->begin(), il->end(), "NOP"s, [](const auto &pair, const std::string &w) {
   return pair.first < w;
   });
  if ( nop == il->end() ) {
   Err("Warning: cannot find NOP\n");
  } else {
    m_nop = nop->second.at(0);
    m_nop_rend = m_dis->get_rend(m_nop->n);
    if ( !m_nop_rend ) {
      Err("Warning: cannot find NOP render\n");
      m_nop = nullptr;
    }
  }
  // read symbols
  return _read_symbols(opt_t, [&](asymbol &sym) {
   auto find_cs = m_code_sects.find(sym.section);
   bool add = find_cs != m_code_sects.end() && (sym.type != STT_SECTION) && (sym.type != STT_FILE);
     m_syms.push_back(std::move(sym));
     if ( add ) {
       auto *last = &m_syms.back();
       m_named[last->name] = last;
     }
   });
}

int CEd_base::setup_f(Ced_named::const_iterator &fiter, const char *fname)
{
  auto s_idx = fiter->second->section;
  auto siter = m_code_sects.find(s_idx);
  if ( siter == m_code_sects.end() ) {
    Err("section %d don't have code, %s\n", s_idx, fname);
    return 0;
  }
  setup_labels(s_idx);
  section *sec = m_reader->sections[s_idx];
  m_idx = s_idx;
  m_obj_off = fiter->second->addr;
  m_obj_size = fiter->second->size;
  m_file_off = sec->get_offset() + m_obj_off;
  setup_srelocs(s_idx);
  m_state = WantOff;
  return 1;
}

int CEd_base::setup_s(int s_idx)
{
  setup_labels(s_idx);
  section *sec = m_reader->sections[s_idx];
  m_idx = s_idx;
  m_obj_off = 0;
  m_obj_size = sec->get_size();
  m_file_off = sec->get_offset();
  setup_srelocs(s_idx);
  m_state = WantOff;
  return 1;
}

// try to reuse as much code from base ParseSASS as possible
// actual value in m_v
int CEd_base::parse_num(NV_Format fmt, std::string_view &tail)
{
  if ( fmt == NV_BITSET && tail.at(0) == '{' ) {
    parse_bitset(1, tail);
    return 1;
  }
  m_minus = 0;
  int idx = 0;
  if ( tail.at(0) == '-' ) {
    idx++;
    m_minus = 1;
  }
  if ( fmt < NV_F64Imm ) {
    int i = 0;
    parse_digit(tail.data() + idx, i);
    if ( m_minus ) i = -i;
    m_v = i;
    return 1;
  }
  // check for inf & nan
  int tidx = idx;
  if ( tail.at(idx) == '+' ) tidx++;;
  float fl;
  if ( !strcasecmp(tail.data() + tidx, "inf") ) {
   if ( fmt == NV_F64Imm ) {
     *(double *)&m_v = m_minus ? -INFINITY: INFINITY;
     return 1;
   }
   const uint32_t positive_infinity_f32 = uint32_t(0x7F800000);
   const uint32_t negative_infinity_f32 = uint32_t(0xFF800000);
   fl = *(float *)( m_minus ? &negative_infinity_f32 : &positive_infinity_f32 );
   if ( fmt == NV_F32Imm ) {
     *(float *)&m_v = fl;
     return 1;
   } else if ( fmt == NV_F16Imm ) {
     m_v = fp16_ieee_from_fp32_value(fl);
     return 1;
   }
  } else if ( !strcasecmp(tail.data() + tidx, "nan") || !strcasecmp(tail.data() + tidx, "qnan") ) {
   if ( fmt == NV_F64Imm ) {
     *(double *)&m_v = m_minus ? -NAN: NAN;
     return 1;
   }
    const uint32_t positive_nan_f32 = uint32_t(0x7FFFFFFF);
    const uint32_t negative_nan_f32 = uint32_t(0xFFFFFFFF);
   fl = *(float *)( m_minus ? &negative_nan_f32 : &positive_nan_f32 );
   if ( fmt == NV_F32Imm ) {
     *(float *)&m_v = fl;
     return 1;
   } else if ( fmt == NV_F16Imm ) {
     m_v = fp16_ieee_from_fp32_value(fl);
     return 1;
   }
  }
  // this is floating value
  parse_float_tail(idx, tail);
  if ( m_minus ) m_d = -m_d;
  if ( fmt == NV_F64Imm )
  {
    m_v = *(uint64_t *)&m_d;
  } else if ( fmt == NV_F32Imm ) {
    float fl = (float)this->m_d;
    *(float *)&m_v = fl;
  } else if ( fmt == NV_F16Imm ) {
    m_v = fp16_ieee_from_fp32_value(float(m_d));
  } else return 0;
  return 1;
}

int CEd_base::generic_ins(const nv_instr *ins, NV_extracted &kv)
{
  m_inc_tabs.clear();
  if ( !m_dis->set_mask(ins->mask) ) {
    Err("set_mask for %s %d failed\n", ins->name, ins->line);
    return 0;
  }
  // enum fields
  for ( auto &f: ins->fields ) {
    unsigned long v = 0;
    auto kvi = kv.find(f.name);
    if ( kvi != kv.end() )
     v = kvi->second;
    else
     v = get_def_value(ins, f.name);
    if ( f.scale ) v /= f.scale;
    m_dis->put(f.mask, f.mask_size, v);
  }
  if ( opt_d ) printf("end enums\n");
  // tabs
  if ( ins->tab_fields.size() ) {
    std::vector<unsigned short> row;
    int row_idx = 0;
    for ( auto tf: ins->tab_fields ) {
      for ( auto &sv: tf->fields ) {
        unsigned long v;
        auto kvi = kv.find(sv);
        if ( kvi != kv.end() )
         v = kvi->second;
        else
         v = get_def_value(ins, sv);
        row.push_back((unsigned short)v);
      }
      int res_val = 0;
      if ( !ins->check_tab(tf->tab, row, res_val) ) {
        Err("check_tab index %d for %s %d failed\n", row_idx, ins->name, ins->line);
        return 0;
      }
      m_dis->put(tf->mask, tf->mask_size, res_val);
      row.clear();
      row_idx++;
    }
  }
  if ( opt_d ) printf("end tabs\n");
  // const bank
  if ( ins->cb_field )
  {
    unsigned long c1, c2;
    auto kvi = kv.find(ins->cb_field->f1);
    if ( kvi != kv.end() )
     c1 = kvi->second;
    else
     c1 = get_def_value(ins, ins->cb_field->f1);
    kvi = kv.find(ins->cb_field->f2);
    if ( kvi != kv.end() )
     c2 = kvi->second;
    else
     c2 = get_def_value(ins, ins->cb_field->f2);
    generic_cb(ins, c1, c2);
  }
  if ( opt_h )
    HexDump(m_out, buf, block_size);
  m_dis->flush();
  block_dirty = 1;
  return 1;
}

int CEd_base::generic_cb(const nv_instr *ins, unsigned long c1, unsigned long c2) {
  if ( ins->cb_field->scale )
    c2 /= ins->cb_field->scale;
  // mask can have size 2 or 3. see details in ina.cc kv_field::patch method
  if ( ins->cb_field->mask3 ) {
    auto lo = c1 & 0xf;
    auto hi = (c1 >> 4) & 0xf;
    m_dis->put(ins->cb_field->mask1, ins->cb_field->mask1_size, hi);
    m_dis->put(ins->cb_field->mask2, ins->cb_field->mask2_size, lo);
    m_dis->put(ins->cb_field->mask3, ins->cb_field->mask3_size, c2);
  } else {
    // simple 2 mask
    m_dis->put(ins->cb_field->mask1, ins->cb_field->mask1_size, c1);
    m_dis->put(ins->cb_field->mask2, ins->cb_field->mask2_size, c2);
  }
  return 1;
}

unsigned long CEd_base::value_or_def(const nv_instr *ins, const std::string_view &s, const NV_extracted &kv)
{
  auto kvi = kv.find(s);
  if ( kvi != kv.end() ) return kvi->second;
  return get_def_value(ins, s);
}

unsigned long CEd_base::get_def_value(const nv_instr *ins, const std::string_view &s)
{
  if ( ins->vas ) {
    auto va = find(ins->vas, s);
    if ( va ) return va->dval;
  }
  auto ea = find_ea(ins, s);
  if ( ea && ea->has_def_value ) return ea->def_value;
  // hz - lets return zero
  return 0;
}

int CEd_base::_next_off()
{
  unsigned long off = block_offset();
  int new_block = 0;
  switch(m_width) {
    case 64: if ( m_bidx >= 6 ) {
        new_block = 1;
        off += block_size + 8;
      } else
        m_bidx++;
      break;
    case 88: if ( m_bidx >= 2 ) {
        new_block = 1;
        off += block_size + 8;
      } else
        m_bidx++;
      break;
    case 128: off += block_size; new_block = 1; break;
    default: return 0; // wtf?
  }
  if ( new_block )
  {
    if ( !flush_buf() ) return 0;
    return _verify_off(off);
  }
  // check if we have reloc on real offset
  check_rel(off);
  check_off(off);
  return _disasm(off);
}

int CEd_base::_verify_off(unsigned long off)
{
  m_inc_tabs.clear();
  // check that offset is valid
  if ( off < m_obj_off || off >= (m_obj_off + m_obj_size) ) {
    Err("invalid offset %lX, should be within %lX - %lX\n",
       off, m_obj_off, m_obj_off + m_obj_size);
    return 0;
  }
  // check if offset is properly aligned
  unsigned long off_mask = (1 << mask_size) - 1;
  if ( off & off_mask ) {
    Err("warning: offset %lX is not aligned on 2 ^ %d (off_mask %lX)\n", off, mask_size, off_mask);
    off &= ~off_mask;
  }
  // extract index inside block
  m_bidx = 0;
  unsigned long block_off = m_file_off + (off - m_obj_off);
  if ( m_width != 128 ) {
    auto b_off = off & ~(block_size - 1);
    if ( b_off == off ) {
      Err("warning: offset %lX points to Ctrl Word, change to %lX\n", off, off + 8);
      off += 8;
      m_bidx = 0;
    } else {
      m_bidx = (off - 8 - b_off) / 8;
    }
    block_off = m_file_off + (b_off - m_obj_off);
    if ( opt_d )
      fprintf(m_out, "block_off %lX off %lX block_idx %d\n", b_off, off, m_bidx);
  }
  // check if we have reloc on real offset
  check_rel(off);
  check_off(off);
  if ( block_off != m_buf_off ) {
    // need to read new buffer
    fseek(m_cubin_fp, block_off, SEEK_SET);
    if ( 1 != fread(buf, block_size, 1, m_cubin_fp) ) {
      Err("fread at %lX failed, error %d (%s)\n", m_buf_off, errno, strerror(errno));
      return 0;
    }
    rdr_cnt++;
  }
  m_buf_off = block_off;
  if ( opt_h ) HexDump(m_out, buf, block_size);
  return _disasm(off);
}

int CEd_base::_disasm(unsigned long off)
{
  if ( !m_dis->init(buf, block_size, off, m_bidx) ) {
    Err("dis init failed\n");
    return 0;
  }
  // disasm instruction at offset
  NV_res res;
  int what = 1;
  if ( m_width > 64 ) what = 2;
  int get_res = m_dis->get(res, what);
  if ( get_res < 0 || res.empty() ) {
    Err("cannot disasm at offset %lX\n", off);
    return 0;
  }
  int res_idx = 0;
  if ( res.size() > 1 ) res_idx = calc_index(res, m_dis->rz);
  if ( -1 == res_idx ) {
    Err("warning: ambigious instruction at %lX, has %ld formst\n", off, res.size());
    // lets choose 1st
    res_idx = 0;
  }
  if ( opt_d ) printf("res_idx %d\n", res_idx);
  // store disasm result
  curr_dis = std::move(res[res_idx]);
  m_rend = m_dis->get_rend(curr_dis.first->n);
  if ( !m_rend ) {
    Err("cannot get render at %lX, n %d\n", off, curr_dis.first->n);
    return 0;
  }
  return 1;
}
