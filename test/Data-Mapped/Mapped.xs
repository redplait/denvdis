// #define PERL_NO_GET_CONTEXT
#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"

#include "ppport.h"
#include <sys/mman.h>

void my_warn(const char * pat, ...) {
 va_list args;
 va_start(args, pat);
 vwarn(pat, &args);
 va_end(args);
}

struct DataMapped {
  void clean() {
    if ( m_mapped ) {
      munmap((void *)m_mapped, m_size);
      m_mapped = nullptr;
    }
    if ( m_fp ) {
      fclose(m_fp);
      m_fp = NULL;
    }
  }
  ~DataMapped() {
    clean();
  }
  int open(const char *fname) {
    if ( m_fp ) clean();
    m_fp = fopen(fname, "rb");
    if ( !m_fp ) {
      my_warn("Data::Mapped: cannot open %s, error %d (%s)\n", fname, errno, strerror(errno));
      return 0;
    }
    // seek to end to get size of file
    if ( fseek(m_fp, 0, SEEK_END) ) {
      my_warn("Data::Mapped: cannot seek %s, error %d (%s)\n", fname, errno, strerror(errno));
      fclose(m_fp);
      m_fp = NULL;
      return 0;
    }
    m_size = ftell(m_fp);
    // mmap
    m_mapped = (const char *)mmap(NULL, m_size, PROT_READ, MAP_PRIVATE, fileno(m_fp), 0);
    if ( !m_mapped ) {
      my_warn("Data::Mapped: cannot mmap(%d) %s, error %d (%s)\n", m_size, fname, errno, strerror(errno));
      clean();
      return 0;
    }
    return 1;
  }
  SV *at(int off) {
    if ( !m_mapped ) return &PL_sv_undef;
    if ( off < 0 || off >= m_size ) {
      my_warn("Data::Mapped: bad offset %d\n", off);
      return &PL_sv_undef;
    }
    auto body = m_mapped + off;
    int res = 0;
    for ( auto curr = body; *curr && off < m_size; ++curr, ++off ) ++res;
    return newSVpv(body, res);
  }
  // data
  FILE *m_fp = NULL;
  off_t m_size = 0;
  const char *m_mapped = nullptr;
};

static U32 my_len(pTHX_ SV *sv, MAGIC* mg);

template <typename T>
static int magic_del(pTHX_ SV* sv, MAGIC* mg) {
    if (mg->mg_ptr) {
        auto *m = (T *)mg->mg_ptr;
        if ( m ) delete m;
        mg->mg_ptr= NULL;
    }
    return 0; // ignored anyway
}

#ifdef MGf_LOCAL
#define TAB_TAIL ,0
#else
#define TAB_TAIL
#endif

static const char *s_dm = "Data::Mapped";
static HV *s_dm_pkg = nullptr;
static MGVTBL dm_magic_vt = {
        0, /* get */
        0, /* write */
        my_len, /* length */
        0, /* clear */
        magic_del<DataMapped>,
        0, /* copy */
        0 /* dup */
        TAB_TAIL
};

static U32 my_len(pTHX_ SV *sv, MAGIC* mg)
{
  DataMapped *d = nullptr;
  if (SvMAGICAL(sv)) {
    MAGIC* magic;
    for (magic= SvMAGIC(sv); magic; magic = magic->mg_moremagic)
      if ( magic->mg_virtual == &dm_magic_vt) {
        d = (DataMapped*) magic->mg_ptr;
        break;
      }
  }
  if ( !d ) {
    my_warn("my_len %d\n", SvTYPE(sv));
    return 0;
  }
  return (U32)d->m_size;
}

template <typename T>
static T *get_magic_ext(SV *obj, int die, MGVTBL *tab)
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
            return (T*) magic->mg_ptr;
    }
  return NULL;
}


MODULE = Data::Mapped		PACKAGE = Data::Mapped

void
new(obj_or_pkg, const char *fname)
  SV *obj_or_pkg
 INIT:
  HV *pkg = NULL;
  SV *msv;
  SV *objref= NULL;
  MAGIC* magic;
  DataMapped *res = nullptr;
 PPCODE:
  if (SvPOK(obj_or_pkg) && (pkg= gv_stashsv(obj_or_pkg, 0))) {
    if (!sv_derived_from(obj_or_pkg, s_dm))
        croak("Package %s does not derive from %s", SvPV_nolen(obj_or_pkg), s_dm);
  } else
    croak("new: first arg must be package name or blessed object");
  // make new DataMapped
  res = new DataMapped();
  if ( !res->open(fname) ) {
    delete res;
    ST(0) = &PL_sv_undef;
  } else {
     msv = newSViv(0);
     objref = newRV_noinc((SV*)msv);
     sv_bless(objref, pkg);
     magic = sv_magicext((SV*)msv, NULL, PERL_MAGIC_ext, &dm_magic_vt, (const char *)res, 0);
     SvREADONLY_on((SV*)msv);
     ST(0) = objref;
  }
  XSRETURN(1);

SV *
at(SV *self, int idx)
 INIT:
  auto *d = get_magic_ext<DataMapped>(self, 1, &dm_magic_vt);
 CODE:
  RETVAL = d->at(idx);
 OUTPUT:
  RETVAL

BOOT:
 s_dm_pkg = gv_stashpv(s_dm, 0);
 if ( !s_dm_pkg )
    croak("Package %s does not exists", s_dm);