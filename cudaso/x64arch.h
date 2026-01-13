#pragma once

#include <cstddef>
#include <unordered_map>
#include "elfio/elfio.hpp"
#include "types.h"

#define __UD_STANDALONE__
#include "libudis86/types.h"
#include "libudis86/extern.h"
#include "libudis86/itab.h"

extern int opt_d;

typedef enum
{
  jo = 1,
  jno,
  jb,
  jae,
  jz,
  jnz,
  jbe,
  ja,
  js,
  jns,
  jp,
  jnp,
  jl,
  jge,
  jle,
  jg,
  jcx,
  jc,  // for arm64 CS
  jnc, // for arm64 CC
  jmp, // just direct jump
} jmp_tag;


template <typename V>
class used_regs
{
  public:
    void add(ud_type reg, V value)
    {
      try
      {
        m_regs[reg] = value;
      } catch(std::bad_alloc)
      { }
    }
    int add_off(ud_type reg_dst, ud_type reg_src, V value)
    {
      auto iter = m_regs.find(reg_src);
      if ( iter == m_regs.end() )
        return 0;
      if ( reg_dst == reg_src )
        add(reg_dst, value);
      else
        add(reg_dst, iter->second + value);
      return 1;
    }
    V add_zero(ud_type reg_dst, ud_type reg_src, V value)
    {
      auto iter = m_regs.find(reg_src);
      if ( iter == m_regs.end() )
      {
        add(reg_dst, value);
        return value;
      }
      if ( reg_dst == reg_src )
        add(reg_dst, value);
      else
        add(reg_dst, iter->second + value);
      return iter->second + value;
    }
    void erase(ud_type reg)
    {
      auto iter = m_regs.find(reg);
      if ( iter != m_regs.end() )
        m_regs.erase(iter);
    }
    int asgn(ud_type reg, V &out_value)
    {
      auto iter = m_regs.find(reg);
      if ( iter != m_regs.end() )
      {
        out_value = iter->second;
        return 1;
      }
      return 0;
    }
    int mov(ud_type src, ud_type dst)
    {
      auto iter = m_regs.find(src);
      if ( iter == m_regs.end() )
        return 0;
      add(dst, iter->second);
      return 1;
    }
    int exists(ud_type reg) const
    {
      auto iter = m_regs.find(reg);
      if ( iter == m_regs.end() )
        return 0;
      return 1;
    }
    inline void clear()
    {
      m_regs.clear();
    }
    inline int empty() const
    {
      return m_regs.empty();
    }
    inline size_t size() const
    {
      return m_regs.size();
    }
    inline int asgn_first(V &out_value) const
    {
      if ( m_regs.empty() )
        return 0;
      out_value = m_regs.begin()->second;
      return 1;
    }
    // graph machinery
    bool operator<(const used_regs &outer) const
    {
      return ( m_regs.size() < outer.m_regs.size() );
    }
  protected:
    std::unordered_map<ud_type, V> m_regs;
};

template <typename S, typename T = uint64_t>
struct regs_state: public used_regs<T>
{
  S state;
  regs_state() = default;
  regs_state(S &&s): used_regs<T>(), state(s)
  { };
  regs_state(regs_state<S, T> &&) = default;
  regs_state(const regs_state<S, T> &) = default;
  // graph machinery
  bool operator<(const regs_state<S, T> &outer) const
  {
    if ( state == outer.state )
      return (used_regs<T>)*this < (used_regs<T>)outer;
    return ( state < outer.state );
  }
  regs_state<S, T>& operator=(const regs_state<S, T>&) = default;
};

// main workhorse
struct diter
{
  // data
  ud_t ud_obj;
  ULONG curr_len, total;
  ELFIO::section *m_s;
  ptrdiff_t m_psp = 0;
  int m_kind = 64;

  inline void reset()
  {
    total = curr_len = 0;
    m_psp = 0;
  }
  diter(ELFIO::section *s, int kind = 64)
   : m_s(s), m_kind(kind)
  {
    ud_init(&ud_obj);
    ud_set_mode(&ud_obj, kind);
    reset();
    if ( opt_d )
      ud_set_syntax(&ud_obj, UD_SYN_INTEL);
  }
  ULONG next()
  {
     total += curr_len;
     curr_len = ud_disassemble(&ud_obj);
     return curr_len;
  }
  int setup(ptrdiff_t off)
  {
    reset();
    m_psp = off;
    // sanity check
    auto sa = m_s->get_address();
    auto size = m_s->get_size();
    if ( off < sa || off >= (sa + size) )
      return 0;
    DWORD avail = DWORD(sa + size - off);
    ud_set_input_buffer(&ud_obj, (uint8_t *)m_s->get_data() + off - sa, avail);
    ud_set_pc(&ud_obj, (uint64_t)m_psp);
    return 1;
  }
  int setup(ptrdiff_t off, DWORD limit)
  {
    reset();
    m_psp = off;
    // sanity check
    auto sa = m_s->get_address();
    auto size = m_s->get_size();
    if ( off < sa || off >= (sa + size) )
      return 0;
    DWORD avail = DWORD(sa + size - off);
    if ( limit < avail )
      avail = limit;
    ud_set_input_buffer(&ud_obj, (uint8_t *)m_s->get_data() + off - sa, avail);
    ud_set_pc(&ud_obj, (uint64_t)m_psp);
    return 1;
  }
  ptrdiff_t get(int idx) const
  {
    return ud_obj.pc + ud_obj.operand[idx].lval.sdword;
  }
  ptrdiff_t pc() const
  {
    return ud_obj.pc;
  }
  ptrdiff_t prev_pc() const
  {
    return ud_obj.pc - curr_len;
  }
  ptrdiff_t get_jval(int idx) const
  {
    if (ud_obj.operand[idx].size == 8)
      return ud_obj.operand[idx].lval.sbyte;
    else if (ud_obj.operand[idx].size == 16)
      return ud_obj.operand[idx].lval.sword;
    else
      return ud_obj.operand[idx].lval.sdword;
  }
  ptrdiff_t get_jmp(int idx) const
  {
    if (ud_obj.operand[idx].size == 8)
      return ud_obj.pc + ud_obj.operand[idx].lval.sbyte;
    else if (ud_obj.operand[idx].size == 16)
      return ud_obj.pc + ud_obj.operand[idx].lval.sword;
    else
      return ud_obj.pc + ud_obj.operand[idx].lval.sdword;
  }
   int is_jmp(jmp_tag &tag) const;
   int is_jmp() const;
   int jxx_jimm(jmp_tag &tag) const;
   int is_jxx_jimm() const;
   inline int is_jxx_jimm(ud_mnemonic_code op) const
   {
     return (ud_obj.mnemonic == op) &&
            (ud_obj.operand[0].type == UD_OP_JIMM)
     ;
   }
   // variadic version
   template <typename T>
   int is_jxx(T op) const
   {
     return (ud_obj.mnemonic == op);
   }
   template <typename T, typename... Args>
   int is_jxx(T op, Args... args) const
   {
     return (ud_obj.mnemonic == op) || is_jxx(args...);
   }
   template <typename T, typename... Args>
   int is_jxx_jimm(T op, Args... args)
   {
     if ( ud_obj.operand[0].type != UD_OP_JIMM )
       return 0;
     return is_jxx(op, args...);
   }
   inline int is_call_reg() const
   {
     return (ud_obj.mnemonic == UD_Icall) &&
            (ud_obj.operand[0].type == UD_OP_REG)
     ;
   }
   inline int is_call_jimm() const
   {
     return (ud_obj.mnemonic == UD_Icall) &&
            (ud_obj.operand[0].type == UD_OP_JIMM)
     ;
   }
   inline int is_call_mrip() const
   {
     return (ud_obj.mnemonic == UD_Icall) &&
            (ud_obj.operand[0].type == UD_OP_MEM) &&
            (ud_obj.operand[0].base == UD_R_RIP)
     ;
   }
   inline int is_lea(ud_type r) const
   {
     return (ud_obj.mnemonic == UD_Ilea) &&
            (ud_obj.operand[0].type == UD_OP_REG) &&
            (ud_obj.operand[0].base == r)
     ;
   }
   inline int is_push_imm() const
   {
     return (ud_obj.mnemonic == UD_Ipush) &&
            (ud_obj.operand[0].type == UD_OP_IMM)
     ;
   }
   inline int is_push_reg() const
   {
     return (ud_obj.mnemonic == UD_Ipush) &&
            (ud_obj.operand[0].type == UD_OP_REG)
     ;
   }
   inline int is_push_mem() const
   {
     return (ud_obj.mnemonic == UD_Ipush) &&
            (ud_obj.operand[0].type == UD_OP_MEM)
     ;
   }
   inline DWORD get_const32(int op_idx) const
   {
      if ( ud_obj.operand[op_idx].size == 32 )
        return ud_obj.operand[op_idx].lval.sdword;
      else if ( ud_obj.operand[op_idx].size == 8 )
        return ud_obj.operand[op_idx].lval.sbyte;
      else if ( ud_obj.operand[op_idx].size == 16 )
        return ud_obj.operand[op_idx].lval.sword;
      else
        return 0;
   }
   inline int64_t get_const64(int op_idx) const
   {
      if ( ud_obj.operand[op_idx].size == 64 )
        return ud_obj.operand[op_idx].lval.sqword;
      if ( ud_obj.operand[op_idx].size == 32 )
        return ud_obj.operand[op_idx].lval.sdword;
      else if ( ud_obj.operand[op_idx].size == 8 )
        return ud_obj.operand[op_idx].lval.sbyte;
      else if ( ud_obj.operand[op_idx].size == 16 )
        return ud_obj.operand[op_idx].lval.sword;
      else
        return 0;
   }
   template <typename T>
   inline int adjust_call(T &call_addr, int op_idx) const
   {
      if ( ud_obj.operand[op_idx].size == 32 )
        call_addr += ud_obj.operand[op_idx].lval.sdword;
      else if ( ud_obj.operand[op_idx].size == 8 )
        call_addr += ud_obj.operand[op_idx].lval.sbyte;
      else if ( ud_obj.operand[op_idx].size == 16 )
        call_addr += ud_obj.operand[op_idx].lval.sword;
      else
        return 0;
      return 1;
   }
   static int is_r64(ud_type);
   // for x64 64 -> 32 reg conversion
   static int get64to32reg(ud_type, ud_type &);
   // 32 -> 64
   static int get32to64reg(ud_type, ud_type &);
   // 16 -> 32 reg conversion
   static int get16to32reg(ud_type, ud_type &);
   // 32 -> 16 reg conversion
   static int get32to16reg(ud_type, ud_type &);
   // 16 -> 64 reg conversion
   static int get16to64reg(ud_type, ud_type &);
   // for used_regs - use only 32 or 64bit regs
   ud_type normalize_reg(ud_type, uint8_t size);
   inline int is_ret3() const
   {
     return (ud_obj.mnemonic == UD_Iint3) ||
            (ud_obj.mnemonic == UD_Iret)
     ;
   }
   inline int is_ret() const
   {
     return (ud_obj.mnemonic == UD_Iint3) || (ud_obj.mnemonic == UD_Iretf) ||
            ((ud_obj.mnemonic == UD_Iint) && (ud_obj.operand[0].lval.sdword != 0x2c)) ||
            (ud_obj.mnemonic == UD_Iret) || (ud_obj.mnemonic == UD_Iiretq)
     ;
   }
   inline int is_end() const
   {
     return (ud_obj.mnemonic == UD_Iint3)  ||
            (ud_obj.mnemonic == UD_Iret)   ||
            (ud_obj.mnemonic == UD_Iretf)  ||
            (ud_obj.mnemonic == UD_Iiretq) ||
            (ud_obj.mnemonic == UD_Ihlt)   ||
            (ud_obj.mnemonic == UD_Ijmp)   ||
            (ud_obj.mnemonic == UD_Iinvalid)
     ;
   }
   // same as is_end but keep jmp $+2
   inline int is_end2() const
   {
     if ( (ud_obj.mnemonic == UD_Iint3) ||
          (ud_obj.mnemonic == UD_Iret)
        )
       return 1;
     if ( ud_obj.mnemonic == UD_Ijmp )
     {
       if ( (ud_obj.operand[0].type == UD_OP_JIMM) &&
            !ud_obj.operand[0].lval.sdword
          )
         return 0;
       return 1;
     }
     return 0;
   }
   inline int is_jmp_reg() const
   {
     return (ud_obj.mnemonic == UD_Ijmp) &&
            (ud_obj.operand[0].type == UD_OP_REG)
     ;
   }
   int is_setXX() const;
   int is_cmovXX() const;
   inline int is_shr_rimm() const
   {
     return (ud_obj.mnemonic == UD_Ishr) &&
            (ud_obj.operand[0].type == UD_OP_REG) &&
            (ud_obj.operand[1].type == UD_OP_IMM)
     ;
   }
   inline int is_xor_rr() const
   {
     return (ud_obj.mnemonic == UD_Ixor) &&
            (ud_obj.operand[0].type == UD_OP_REG) &&
            (ud_obj.operand[1].type == UD_OP_REG)
     ;
   }
   inline int is_mov_rr() const
   {
     return (ud_obj.mnemonic == UD_Imov) &&
            (ud_obj.operand[0].type == UD_OP_REG) &&
            (ud_obj.operand[1].type == UD_OP_REG)
     ;
   }
   inline int is_and_rimm() const
   {
     return (ud_obj.mnemonic == UD_Iand) &&
            (ud_obj.operand[0].type == UD_OP_REG) &&
            (ud_obj.operand[1].type == UD_OP_IMM)
     ;
   }
   inline int is_add_rimm() const
   {
     return (ud_obj.mnemonic == UD_Iadd) &&
            (ud_obj.operand[0].type == UD_OP_REG) &&
            (ud_obj.operand[1].type == UD_OP_IMM)
     ;
   }
   inline int is_add_rimm(ud_type reg) const
   {
     return (ud_obj.mnemonic == UD_Iadd) &&
            (ud_obj.operand[0].type == UD_OP_REG) &&
            (ud_obj.operand[0].base == reg)       &&
            (ud_obj.operand[1].type == UD_OP_IMM)
     ;
   }
   inline int is_sub_rimm() const
   {
     return (ud_obj.mnemonic == UD_Isub) &&
            (ud_obj.operand[0].type == UD_OP_REG) &&
            (ud_obj.operand[1].type == UD_OP_IMM)
     ;
   }
   inline int is_sub_rimm(ud_type reg) const
   {
     return (ud_obj.mnemonic == UD_Isub) &&
            (ud_obj.operand[0].type == UD_OP_REG) &&
            (ud_obj.operand[1].type == UD_OP_IMM) &&
            (ud_obj.operand[0].base == reg)
     ;
   }
   inline int is_add_rr() const
   {
     return (ud_obj.mnemonic == UD_Iadd) &&
            (ud_obj.operand[0].type == UD_OP_REG) &&
            (ud_obj.operand[1].type == UD_OP_REG)
     ;
   }
   inline int is_movzx_rr() const
   {
     return (ud_obj.mnemonic == UD_Imovzx) &&
            (ud_obj.operand[0].type == UD_OP_REG) &&
            (ud_obj.operand[1].type == UD_OP_REG)
     ;
   }
   inline int is_cmp_mimm() const
   {
     return (ud_obj.mnemonic == UD_Icmp) &&
            (ud_obj.operand[0].type == UD_OP_MEM) &&
            (ud_obj.operand[1].type == UD_OP_IMM)
     ;
   }
   inline int is_cmp_rimm() const
   {
     return (ud_obj.mnemonic == UD_Icmp) &&
            (ud_obj.operand[0].type == UD_OP_REG) &&
            (ud_obj.operand[1].type == UD_OP_IMM)
     ;
   }
   inline int is_cmp_rimm(ud_type reg) const
   {
     return (ud_obj.mnemonic == UD_Icmp) &&
            (ud_obj.operand[0].type == UD_OP_REG) &&
            (ud_obj.operand[0].base == reg) &&
            (ud_obj.operand[1].type == UD_OP_IMM)
     ;
   }
   inline int is_cmp_rr() const
   {
     return (ud_obj.mnemonic == UD_Icmp) &&
            (ud_obj.operand[0].type == UD_OP_REG) &&
            (ud_obj.operand[1].type == UD_OP_REG)
     ;
   }
   inline int is_test_rr() const
   {
     return (ud_obj.mnemonic == UD_Itest) &&
            (ud_obj.operand[0].type == UD_OP_REG) &&
            (ud_obj.operand[1].type == UD_OP_REG)
     ;
   }
   // predicate methods for 32bit
   inline int is_rmem() const
   {
     return (ud_obj.operand[0].type == UD_OP_REG) &&
            (ud_obj.operand[1].type == UD_OP_MEM)
     ;
   }
   inline int is_rmem(ud_type reg) const
   {
     return (ud_obj.operand[0].type == UD_OP_REG) &&
            (ud_obj.operand[0].base == reg) &&
            (ud_obj.operand[1].type == UD_OP_MEM)
     ;
   }
   inline int is_memr() const
   {
     return (ud_obj.operand[1].type == UD_OP_REG) &&
            (ud_obj.operand[0].type == UD_OP_MEM)
     ;
   }
   inline int is_memr(ud_type reg) const
   {
     return (ud_obj.operand[1].type == UD_OP_REG) &&
            (ud_obj.operand[1].base == reg) &&
            (ud_obj.operand[0].type == UD_OP_MEM)
     ;
   }
   inline int is_movm32c() const
   {
     return (ud_obj.mnemonic == UD_Imov) &&
            (ud_obj.operand[0].type == UD_OP_MEM) &&
            (ud_obj.operand[1].type == UD_OP_IMM) &&
            (ud_obj.operand[1].size == 32)
     ;
   }
   inline int is_mov_rimm() const
   {
     return (ud_obj.mnemonic == UD_Imov) &&
            (ud_obj.operand[0].type == UD_OP_REG) &&
            (ud_obj.operand[1].type == UD_OP_IMM)
     ;
   }
   inline int is_mov_rimm(ud_type reg) const
   {
     return (ud_obj.mnemonic == UD_Imov) &&
            (ud_obj.operand[0].type == UD_OP_REG) &&
            (ud_obj.operand[1].type == UD_OP_IMM) &&
            (ud_obj.operand[0].base == reg)
     ;
   }
   // predicate functions for 64bit
   template <typename T, typename... Args>
   int is_mrip(int idx, T op, Args... args) const
   {
     if ( ud_obj.operand[idx].type != UD_OP_MEM ||
          ud_obj.operand[idx].base != UD_R_RIP )
       return 0;
     return is_jxx(op, args...);
   }
   inline int is_mrip(ud_mnemonic_code op, int idx) const
   {
     return (ud_obj.mnemonic == op) && 
            (ud_obj.operand[idx].type == UD_OP_MEM) &&
            (ud_obj.operand[idx].base == UD_R_RIP)
     ;
   }
   inline int is_imm(ud_mnemonic_code op, int idx) const
   {
     return (ud_obj.mnemonic == op) &&
            (ud_obj.operand[idx].type == UD_OP_IMM)
     ;
   }
   inline int is_mrip(int idx) const
   {
     return (ud_obj.operand[idx].type == UD_OP_MEM) &&
            (ud_obj.operand[idx].base == UD_R_RIP)
     ;
   }
   inline int is_r0() const
   {
     return (ud_obj.operand[1].type == UD_OP_REG) &&
            (ud_obj.operand[0].type == UD_OP_MEM) &&
            (ud_obj.operand[0].base == UD_R_RIP)
     ;
   }
   inline int is_r0(ud_type reg) const
   {
     return (ud_obj.operand[1].type == UD_OP_REG) &&
            (ud_obj.operand[0].type == UD_OP_MEM) &&
            (ud_obj.operand[0].base == UD_R_RIP)  &&
            (ud_obj.operand[1].base == reg)
     ;
   }
   inline int is_r1() const
   {
     return (ud_obj.operand[0].type == UD_OP_REG) &&
            (ud_obj.operand[1].type == UD_OP_MEM) &&
            (ud_obj.operand[1].base == UD_R_RIP)
     ;
   }
   inline int is_r1(ud_type reg) const
   {
     return (ud_obj.operand[0].type == UD_OP_REG) &&
            (ud_obj.operand[1].type == UD_OP_MEM) &&
            (ud_obj.operand[1].base == UD_R_RIP)  &&
            (ud_obj.operand[0].base == reg)
     ;
   }
   inline int is_lea_stack(ud_type r) const
   {
     return (ud_obj.mnemonic == UD_Ilea) &&
            (ud_obj.operand[0].type == UD_OP_REG) &&
            (ud_obj.operand[1].base == r)
     ;
   }
   int op_offset(int idx)
   {
     if (ud_obj.operand[idx].offset == 8)
       return ud_obj.operand[idx].lval.sbyte;
     if (ud_obj.operand[idx].offset == 16)
       return ud_obj.operand[idx].lval.sword;
     if (ud_obj.operand[idx].offset == 32)
       return ud_obj.operand[idx].lval.sdword;
     return 0;
   }
   inline int is_mov_stack(ud_type reg) const
   {
     return (ud_obj.mnemonic == UD_Imov) &&
            (ud_obj.operand[0].type == UD_OP_MEM) &&
            (ud_obj.operand[0].base == reg) &&
            (ud_obj.operand[1].type == UD_OP_IMM)
     ;
   }
   inline int is_mov64() const {
     return (ud_obj.mnemonic == UD_Imov) &&
            (ud_obj.operand[1].size == 64);
   }
   inline int is_mov32() const {
     return (ud_obj.mnemonic == UD_Imov) &&
            (ud_obj.operand[1].size == 32);
   }
   inline int is_movr(ud_type r) const
   {
     return (ud_obj.mnemonic == UD_Imov) &&
            (ud_obj.operand[0].type == UD_OP_REG) &&
            (ud_obj.operand[0].base == r)
     ;
   }
   inline int is_popr(ud_type r) const
   {
     return (ud_obj.mnemonic == UD_Ipop) &&
            (ud_obj.operand[0].type == UD_OP_REG) &&
            (ud_obj.operand[0].base == r)
     ;
   }
   // debugging dump
   void dasm();
   void dasm(int state);
};
