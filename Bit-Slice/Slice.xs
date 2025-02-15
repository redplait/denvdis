// #define PERL_NO_GET_CONTEXT
#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"
#include <vector>
#include <string>

#include "ppport.h"

static const char *s_package = "Bit::Slice";
static const char *hexes = "0123456789abcdef";

/* make CHAR_BIT 8 if it's not defined in limits.h */
#ifndef CHAR_BIT
#warning CHAR_BIT not defined.  Assuming 8 bits.
#define CHAR_BIT 8
#endif

#define BIT_CHAR(bit)         ((bit) / CHAR_BIT)
/* position of bit within character */
#define BIT_IN_CHAR(bit)      (1 << (CHAR_BIT - 1 - ((bit)  % CHAR_BIT)))

class bit_slice {
 public:
  bit_slice() = default;
  bit_slice(AV* array) {
   for ( int i = 0; i <= av_len(array); i++ ) {
     SV** elem = av_fetch(array, i, 0);
     if (elem != NULL)
      m_array.push_back( (unsigned char)SvUV(*elem) );
   }
  }
  void to_str(std::string &res) const {
    for ( auto c: m_array ) {
      res.push_back( hexes[c >> 4] );
      res.push_back( hexes[c & 0xf] );
    }
  }
  bool test(int idx) const
  {
    if ( idx >= m_array.size() * CHAR_BIT ) return false;
    return m_array[BIT_CHAR(idx)] & BIT_IN_CHAR(idx);
  }
  bool extract(int from, int len, uint64_t &res) const
  {
    auto lim = m_array.size() * CHAR_BIT;
    if ( from + len > lim ) {
    }
    int pos = BIT_CHAR(from);
    from -= pos * CHAR_BIT;
    for ( int i = pos; i < m_array.size() && len; i++ )
    {
      int next = process(m_array[i], from, len, res);
      if ( len ) res <<= next;
    }
    return true;
  }
  bool extract2(int from1, int len1, int from2, int len2, uint64_t &res) const
  {
    // check total length
    if ( len1 + len2 > sizeof(uint64_t) * CHAR_BIT) {
      croak("len1 %d + len2 %d > sizeof(uint64_t)\n", len1, len2);
      return false;
    }
    uint64_t res2 = 0;
    if ( !extract(from1, len1, res) || !extract(from2, len2, res2)) return false;
    res = (res << len2) | res2;
    return true;
  }
  // looks very clumsy
  bool extract3(int from1, int len1, int from2, int len2, int from3, int len3, uint64_t &res) const
  {
    // check total length
    if ( len1 + len2 + len3 > sizeof(uint64_t) * CHAR_BIT) {
      croak("len1 %d + len2 %d + len3 %d > sizeof(uint64_t)\n", len1, len2, len3);
      return false;
    }
    uint64_t res2 = 0;
    if ( !extract(from1, len1, res) || !extract(from2, len2, res2)) return false;
    res = (res << len2) | res2;
    res2 = 0;
    if ( !extract(from3, len3, res2) ) return false;
    res = (res << len3) | res2;
    return true;
  }
  // generalized extractN, expects array in form fromI, lenI
  bool extractN(AV* array, uint64_t &res) const
  {
    std::vector<std::pair<int, int> > tmp;
    int old, comp = 0;
    for ( int i = 0; i <= av_len(array); i++ ) {
     SV** elem = av_fetch(array, i, 0);
     if (elem != NULL) {
       if ( !comp ) old = SvIV(*elem);
       else tmp.push_back(std::make_pair(old, SvIV(*elem)));
       comp ^= 1;
     }
    }
    // check if we have even list items
    if ( comp || tmp.empty() ) return false;
    // and bitsize don't exceed uint64_t
    int total = 0;
    for ( const auto &p: tmp ) total += p.second;
    if ( total > sizeof(uint64_t) * CHAR_BIT) {
      croak("lenN %d > sizeof(uint64_t)\n", total);
      return false;
    }
    // extract bits
    for ( int i = 0; i < tmp.size(); ++i ) {
      uint64_t res2 = 0;
      if ( !extract(tmp[i].first, tmp[i].second, res2) ) return false;
      if ( !i ) res = res2;
      else res = (res << tmp[i-1].second) | res2;
    }
    return true;
  }
 protected:
  inline int process(unsigned char b, int &pos, int &len, uint64_t &res) const
  {
    int f = pos & 0x7;
    if ( !f ) {
      if ( len >= CHAR_BIT ) {
        // return whole byte
        res |= b;
        len -= CHAR_BIT;
        return CHAR_BIT;
      }
      // some part of current byte and it is last
      res |= extractBits(b, f, len);
      len = 0;
      return 0;
    }
    int av = std::min(CHAR_BIT - f, len);
    res |= extractBits(b, f, av);
    len -= av;
    pos -= f;
    return av;
  }
  inline unsigned int extractBits(unsigned char num, unsigned int pos, unsigned int k) const
  {
    // Right shift 'num' by 'pos' bits
    unsigned int shifted = num >> pos;

    // Create a mask with 'k' bits set to 1
    unsigned int mask = (1 << k) - 1;

    // Apply the mask to the shifted number
    return shifted & mask;
  }
  std::vector<unsigned char> m_array;
};

// all boring stuff like in Elf::Reader
static bit_slice *get_magic(SV *obj, int die, MGVTBL *tab)
{
  SV *sv;
  MAGIC* magic;
 
  if (!sv_isobject(obj)) {
     if (die)
        croak("Not an object");
        return NULL;
  }
  sv= SvRV(obj);
  if (SvMAGICAL(sv)) {
     /* Iterate magic attached to this scalar, looking for one with our vtable */
     for (magic= SvMAGIC(sv); magic; magic = magic->mg_moremagic)
        if (magic->mg_type == PERL_MAGIC_ext && magic->mg_virtual == tab)
          /* If found, the mg_ptr points to the fields structure. */
            return (bit_slice*) magic->mg_ptr;
    }
  return NULL;
}

template <typename T>
static int magic_free(pTHX_ SV* sv, MAGIC* mg) {
    if (mg->mg_ptr) {
        T *t = (T *)mg->mg_ptr;
        delete t;
        mg->mg_ptr= NULL;
    }
    return 0; // ignored anyway
}

// magic table for Bit::Slice
static MGVTBL bitslice_magic_vt = {
        0, /* get */
        0, /* write */
        0, /* length */
        0, /* clear */
        magic_free<bit_slice>,
        0, /* copy */
        0 /* dup */
#ifdef MGf_LOCAL
        ,0
#endif
};


MODULE = Bit::Slice		PACKAGE = Bit::Slice

void
new(SV *obj_or_pkg, SV *aref)
 INIT:
  AV *array;
  HV *pkg = NULL;
  SV *msv;
  SV *objref= NULL;
  MAGIC* magic;
  bit_slice *t = NULL;
 PPCODE:
  if (!SvROK(aref) || SvTYPE(SvRV(aref)) != SVt_PVAV)
    croak("expected ARRAY ref");
  array = (AV*) SvRV(aref);
  if (SvPOK(obj_or_pkg) && (pkg= gv_stashsv(obj_or_pkg, 0))) {
    if (!sv_derived_from(obj_or_pkg, s_package))
        croak("Package %s does not derive from %s", SvPV_nolen(obj_or_pkg), s_package);
    msv = newSViv(0);
    objref= sv_2mortal(newRV_noinc(msv));
    sv_bless(objref, pkg);
    ST(0)= objref;
  } else
        croak("new: first arg must be package name or blessed object");
  t = new bit_slice(array);
  magic = sv_magicext(msv, NULL, PERL_MAGIC_ext, &bitslice_magic_vt, (const char*)t, 0);
#ifdef USE_ITHREADS
  magic->mg_flags |= MGf_DUP;
#endif
  XSRETURN(1);

void
to_str(SV *arg)
 INIT:
  struct bit_slice *t= get_magic(arg, 1, &bitslice_magic_vt);
  std::string res;
 PPCODE:
  t->to_str(res);
  ST(0)= sv_2mortal( newSVpv(res.c_str(), res.size()) );
  XSRETURN(1);

void
test(SV *arg, int idx)
 INIT:
  struct bit_slice *t= get_magic(arg, 1, &bitslice_magic_vt);
 PPCODE:
  if ( t->test(idx) )
    ST(0)= &PL_sv_yes;
  else
    ST(0) = &PL_sv_no;
  XSRETURN(1);

void
get(SV *arg, int from, int len)
 INIT:
  struct bit_slice *t= get_magic(arg, 1, &bitslice_magic_vt);
  uint64_t res = 0;
 PPCODE:
  if ( t->extract(from, len, res) )
    ST(0)= sv_2mortal( newSVuv(res) );
  else
    ST(0) = &PL_sv_undef;
  XSRETURN(1);

void
get2(SV *arg, int from1, int len1, int from2, int len2)
 INIT:
  struct bit_slice *t= get_magic(arg, 1, &bitslice_magic_vt);
  uint64_t res = 0;
 PPCODE:
  if ( t->extract2(from1, len1, from2, len2, res) )
    ST(0)= sv_2mortal( newSVuv(res) );
  else
    ST(0) = &PL_sv_undef;
  XSRETURN(1);

void
get3(SV *arg, int from1, int len1, int from2, int len2, int from3, int len3)
 INIT:
  struct bit_slice *t= get_magic(arg, 1, &bitslice_magic_vt);
  uint64_t res = 0;
 PPCODE:
  if ( t->extract3(from1, len1, from2, len2, from3, len3, res) )
    ST(0)= sv_2mortal( newSVuv(res) );
  else
    ST(0) = &PL_sv_undef;
  XSRETURN(1);

void
getN(SV *arg, SV *aref)
 INIT:
  struct bit_slice *t= get_magic(arg, 1, &bitslice_magic_vt);
  uint64_t res = 0;
  AV *array;
 PPCODE:
  if (!SvROK(aref) || SvTYPE(SvRV(aref)) != SVt_PVAV)
    croak("getN expects ARRAY ref");
  array = (AV*) SvRV(aref);
  if ( t->extractN(array, res) )
    ST(0)= sv_2mortal( newSVuv(res) );
  else
    ST(0) = &PL_sv_undef;
  XSRETURN(1);
