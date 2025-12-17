#include "x64arch.h"

int diter::jxx_jimm(jmp_tag &tag) const
{
  if ( !is_jmp(tag) )
    return 0;
  return (ud_obj.operand[0].type == UD_OP_JIMM);
}

int diter::is_jxx_jimm() const
{
  if ( !is_jmp() )
    return 0;
  return (ud_obj.operand[0].type == UD_OP_JIMM);
}

int diter::is_setXX() const
{
  switch(ud_obj.mnemonic)
  {
    case UD_Iseto:
    case UD_Isetno:
    case UD_Isetb:
    case UD_Isetnb:
    case UD_Isetz:
    case UD_Isetnz:
    case UD_Isetbe:
    case UD_Iseta:
    case UD_Isets:
    case UD_Isetns:
    case UD_Isetp:
    case UD_Isetnp:
    case UD_Isetl:
    case UD_Isetge:
    case UD_Isetle:
    case UD_Isetg:
      return 1;
  }
  return 0;
}

int diter::is_cmovXX() const
{
  switch(ud_obj.mnemonic)
  {
    case UD_Icmovo:
    case UD_Icmovno:
    case UD_Icmovb:
    case UD_Icmovae:
    case UD_Icmovz:
    case UD_Icmovnz:
    case UD_Icmovbe:
    case UD_Icmova:
    case UD_Icmovs:
    case UD_Icmovns:
    case UD_Icmovp:
    case UD_Icmovnp:
    case UD_Icmovl:
    case UD_Icmovge:
    case UD_Icmovle:
    case UD_Icmovg:
     return 1;
  }
  return 0;
}

int diter::is_jmp(jmp_tag &tag) const
{
  switch(ud_obj.mnemonic)
  {
    case UD_Ijo:
      tag = jo; return 1;
    case UD_Ijno:
      tag = jno; return 1;
    case UD_Ijb:
      tag = jb; return 1;
    case UD_Ijae:
      tag = jae; return 1;
    case UD_Ijz:
      tag = jz; return 1;
    case UD_Ijnz:
      tag = jnz; return 1;
    case UD_Ijbe:
      tag = jbe; return 1;
    case UD_Ija:
      tag = ja; return 1;
    case UD_Ijs:
      tag = js; return 1;
    case UD_Ijns:
      tag = jns; return 1;
    case UD_Ijp:
      tag = jp; return 1;
    case UD_Ijnp:
      tag = jnp; return 1;
    case UD_Ijl:
      tag = jl; return 1;
    case UD_Ijge:
      tag = jge; return 1;
    case UD_Ijle:
      tag = jle; return 1;
    case UD_Ijg:
      tag = jg; return 1;
    case UD_Ijcxz:
    case UD_Ijecxz:
    case UD_Ijrcxz:
      tag = jcx; return 1;
    case UD_Ijmp:
      tag = jmp; return 1;
  }
  return 0;
}

int diter::is_jmp() const
{
  switch(ud_obj.mnemonic)
  {
    case UD_Ijo:
    case UD_Ijno:
    case UD_Ijb:
    case UD_Ijae:
    case UD_Ijz:
    case UD_Ijnz:
    case UD_Ijbe:
    case UD_Ija:
    case UD_Ijs:
    case UD_Ijns:
    case UD_Ijp:
    case UD_Ijnp:
    case UD_Ijl:
    case UD_Ijge:
    case UD_Ijle:
    case UD_Ijg:
    case UD_Ijcxz:
    case UD_Ijecxz:
    case UD_Ijrcxz:
    case UD_Ijmp:
     return 1;
  }
  return 0;
}

int diter::get32to16reg(ud_type from, ud_type &res)
{
  switch(from)
  {
    case UD_R_EAX: res = UD_R_AX; return 1;
    case UD_R_ECX: res = UD_R_CX; return 1;
    case UD_R_EDX: res = UD_R_DX; return 1;
    case UD_R_EBX: res = UD_R_BX; return 1;
    case UD_R_ESP: res = UD_R_SP; return 1;
    case UD_R_EBP: res = UD_R_BP; return 1;
    case UD_R_ESI: res = UD_R_SI; return 1;
    case UD_R_EDI: res = UD_R_DI; return 1;
  }
  return 0;
}

int diter::get16to32reg(ud_type from, ud_type &res)
{
  switch(from)
  {
    case UD_R_AX: res = UD_R_EAX; return 1;
    case UD_R_CX: res = UD_R_ECX; return 1;
    case UD_R_DX: res = UD_R_EDX; return 1;
    case UD_R_BX: res = UD_R_EBX; return 1;
    case UD_R_SP: res = UD_R_ESP; return 1;
    case UD_R_BP: res = UD_R_EBP; return 1;
    case UD_R_SI: res = UD_R_ESI; return 1;
    case UD_R_DI: res = UD_R_EDI; return 1;
  }
  return 0;
}

int diter::get16to64reg(ud_type from, ud_type &res)
{
  switch(from)
  {
    case UD_R_AX:   res = UD_R_RAX; return 1;
    case UD_R_CX:   res = UD_R_RCX; return 1;
    case UD_R_DX:   res = UD_R_RDX; return 1;
    case UD_R_BX:   res = UD_R_RBX; return 1;
    case UD_R_SP:   res = UD_R_RSP; return 1;
    case UD_R_BP:   res = UD_R_RBP; return 1;
    case UD_R_SI:   res = UD_R_RSI; return 1;
    case UD_R_DI:   res = UD_R_RDI; return 1;
    case UD_R_R8W:  res = UD_R_R8;  return 1;
    case UD_R_R9W:  res = UD_R_R9;  return 1;
    case UD_R_R10W: res = UD_R_R10; return 1;
    case UD_R_R11W: res = UD_R_R11; return 1;
    case UD_R_R12W: res = UD_R_R12; return 1;
    case UD_R_R13W: res = UD_R_R13; return 1;
    case UD_R_R14W: res = UD_R_R14; return 1;
    case UD_R_R15W: res = UD_R_R15; return 1;
  }
  return 0;
}

int diter::get32to64reg(ud_type from, ud_type &res)
{
  switch(from)
  {
    case UD_R_EAX: res = UD_R_RAX; return 1;
    case UD_R_ECX: res = UD_R_RCX; return 1;
    case UD_R_EDX: res = UD_R_RDX; return 1;
    case UD_R_EBX: res = UD_R_RBX; return 1;
    case UD_R_ESP: res = UD_R_RSP; return 1;
    case UD_R_EBP: res = UD_R_RBP; return 1;
    case UD_R_ESI: res = UD_R_RSI; return 1;
    case UD_R_EDI: res = UD_R_RDI; return 1;
    case UD_R_R8D:  res = UD_R_R8; return 1;
    case UD_R_R9D:  res = UD_R_R9; return 1;
    case UD_R_R10D: res = UD_R_R10; return 1;
    case UD_R_R11D: res = UD_R_R11; return 1;
    case UD_R_R12D: res = UD_R_R12; return 1;
    case UD_R_R13D: res = UD_R_R13; return 1;
    case UD_R_R14D: res = UD_R_R14; return 1;
    case UD_R_R15D: res = UD_R_R15; return 1;
  }
  return 0;
}

int diter::is_r64(ud_type reg)
{
  switch(reg)
  {
    case UD_R_RAX:
    case UD_R_RCX:
    case UD_R_RDX:
    case UD_R_RBX:
    case UD_R_RSP:
    case UD_R_RBP:
    case UD_R_RSI:
    case UD_R_RDI:
    case UD_R_R8:
    case UD_R_R9:
    case UD_R_R10:
    case UD_R_R11:
    case UD_R_R12:
    case UD_R_R13:
    case UD_R_R14:
    case UD_R_R15:
     return 1;
  }
  return 0;
}

int diter::get64to32reg(ud_type from, ud_type &res)
{
  switch(from)
  {
    case UD_R_RAX: res = UD_R_EAX; return 1;
    case UD_R_RCX: res = UD_R_ECX; return 1;
    case UD_R_RDX: res = UD_R_EDX; return 1;
    case UD_R_RBX: res = UD_R_EBX; return 1;
    case UD_R_RSP: res = UD_R_ESP; return 1;
    case UD_R_RBP: res = UD_R_EBP; return 1;
    case UD_R_RSI: res = UD_R_ESI; return 1;
    case UD_R_RDI: res = UD_R_EDI; return 1;
    case UD_R_R8:  res = UD_R_R8D; return 1;
    case UD_R_R9:  res = UD_R_R9D; return 1;
    case UD_R_R10: res = UD_R_R10D; return 1;
    case UD_R_R11: res = UD_R_R11D; return 1;
    case UD_R_R12: res = UD_R_R12D; return 1;
    case UD_R_R13: res = UD_R_R13D; return 1;
    case UD_R_R14: res = UD_R_R14D; return 1;
    case UD_R_R15: res = UD_R_R15D; return 1;
  }
  return 0;
}

ud_type diter::normalize_reg(ud_type reg, uint8_t size)
{
 if ( 64 == m_kind ) {
  // use 64bit registers
  if ( 64 == size )
    return reg;
  if ( 32 == size )
  {
    ud_type tmp;
    if ( get32to64reg(reg, tmp) )
      return tmp;
  } else if ( 16 == size )
  {
    ud_type tmp;
    if ( get16to64reg(reg, tmp) )
      return tmp;
  }
 } else {
  // use 32bit registers
  if ( 32 == size )
    return reg;
  if ( 16 == size )
  {
    ud_type tmp;
    if ( get16to32reg(reg, tmp) )
      return tmp;
  }
 }
 return reg;
}

void diter::dasm()
{
  if ( !opt_d ) return;
  printf("%p %s (I: %d size %d, II: %d size %d)\n", (PBYTE)ud_insn_off(&ud_obj), ud_insn_asm(&ud_obj),
    ud_obj.operand[0].type, ud_obj.operand[0].size,
    ud_obj.operand[1].type, ud_obj.operand[1].size
  );
}

void diter::dasm(int state)
{
  if ( !opt_d ) return;
  printf("%p %s (I: %d size %d, II: %d size %d) state %d\n",
    (PBYTE)ud_insn_off(&ud_obj), ud_insn_asm(&ud_obj),
    ud_obj.operand[0].type, ud_obj.operand[0].size,
    ud_obj.operand[1].type, ud_obj.operand[1].size, state
  );
}
