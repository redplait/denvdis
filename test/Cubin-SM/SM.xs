// #define PERL_NO_GET_CONTEXT
#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"

#include "ppport.h"
#include <dlfcn.h>
#include "../../scripts/include/nv_types.h"

static void my_warn(const char * pat, ...) {
 va_list args;
 va_start(args, pat);
 vwarn(pat, &args);
 va_end(args);
}

typedef INV_disasm *(*Dproto)(void);

struct sm_guts {
  void *sm_mod = nullptr;
  INV_disasm *ds = nullptr;
  const NV_sorted *m_ns = nullptr;
  int load(const char *);
  int _load(const char *);
  void clean() {
    // call delete before dlclose - else destructor will point into nowhere
    if ( ds ) { delete ds; ds = nullptr; }
    if ( sm_mod ) { dlclose(sm_mod); sm_mod = nullptr; }
    m_ns = nullptr;
  }
 ~sm_guts() { clean(); }
};

int sm_guts::load(const char *fn) {
  sm_mod = dlopen(fn, RTLD_NOW);
  if ( sm_mod ) return _load(fn);
  // try to use SM_DIR
  auto sm_dir = getenv("SM_DIR");
  if ( !sm_dir ) {
    my_warn("sm_guts: cannot load %s, error %d", fn, dlerror());
    return 0;
  }
  auto s = strrchr(fn, '/');
  if ( s ) s++;
  else s = fn;
  std::string path = sm_dir;
  if ( !path.ends_with('/') ) path += '/';
  path += s;
  sm_mod = dlopen(path.c_str(), RTLD_NOW);
  if ( sm_mod ) return _load(path.c_str());
  my_warn("sm_guts: cannot load %s, error %d", fn, dlerror());
  return 0;
}

int sm_guts::_load(const char *fn) {
  Dproto f = (Dproto)dlsym(sm_mod, "get_sm");
  if ( !f ) {
    my_warn("no get_sm in %s", fn);
    return 0;
  }
  ds = f();
  if ( !ds ) {
    my_warn("cannot get_sm in %s", fn);
    return 0;
  }
  m_ns = ds->get_instrs();
  return m_ns != nullptr;
}

#ifdef MGf_LOCAL
#define TAB_TAIL ,0
#else
#define TAB_TAIL
#endif

template <typename T>
static int magic_del(pTHX_ SV* sv, MAGIC* mg) {
    if (mg->mg_ptr) {
        auto *m = (T *)mg->mg_ptr;
        if ( m ) delete m;
        mg->mg_ptr= NULL;
    }
    return 0; // ignored anyway
}

static U32 my_len(pTHX_ SV *sv, MAGIC* mg);

// magic table for Cubin::Attrs
static const char *s_sm = "Cubin::SM";
static HV *s_sm_pkg = nullptr;
static MGVTBL sm_magic_vt = {
        0, /* get */
        0, /* write */
        my_len, /* length */
        0, /* clear */
        magic_del<sm_guts>,
        0, /* copy */
        0 /* dup */
        TAB_TAIL
};

template <typename T>
static T *magic_tied(SV *obj, int die, MGVTBL *tab)
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
        if (magic->mg_type == PERL_MAGIC_tied && magic->mg_virtual == tab)
          /* If found, the mg_ptr points to the fields structure. */
            return (T*) magic->mg_ptr;
    }
  return NULL;
}

sm_guts *get_sm(SV *self) {
  return magic_tied<sm_guts>(self, 1, &sm_magic_vt);
}

static U32 my_len(pTHX_ SV *sv, MAGIC* mg)
{
  sm_guts *d = nullptr;
  if (SvMAGICAL(sv)) {
    MAGIC* magic;
    for (magic= SvMAGIC(sv); magic; magic = magic->mg_moremagic)
      if (magic->mg_type == PERL_MAGIC_tied && magic->mg_virtual == &sm_magic_vt) {
        d = (sm_guts*) magic->mg_ptr;
        break;
      }
  }
  if ( !d && mg->mg_ptr )
    d = (sm_guts*)mg->mg_ptr;
  if ( !d ) {
    my_warn("Cubin::SM: my_type %d\n", SvTYPE(sv));
    return 0;
  }
  if ( d->m_ns ) {
    return (U32)d->m_ns->size()-1;
  } else
   return 0;
}


MODULE = Cubin::SM		PACKAGE = Cubin::SM

void
new(SV *obj_or_pkg, const char *fname = nullptr)
 INIT:
  HV *pkg = NULL;
  SV *msv;
  SV *objref= NULL;
  MAGIC* magic;
  AV *fake;
 PPCODE:
  if (SvPOK(obj_or_pkg) && (pkg= gv_stashsv(obj_or_pkg, 0))) {
    if (!sv_derived_from(obj_or_pkg, s_sm))
        croak("Package %s does not derive from %s", SvPV_nolen(obj_or_pkg), s_sm);
  } else
    croak("new: first arg must be package name or blessed object");
  // make new CAttrs
  auto res = new sm_guts();
  if ( fname ) {
    if ( !res->load(fname) ) {
      delete res;
      ST(0) = &PL_sv_undef;
      XSRETURN(1);
      return;
    }
  }
  fake = newAV();
  objref = newRV_noinc((SV*)fake);
  sv_bless(objref, s_sm_pkg);
  magic = sv_magicext((SV*)fake, NULL, PERL_MAGIC_tied, &sm_magic_vt, (const char *)res, 0);
  SvREADONLY_on((SV*)fake);
  ST(0) = objref;
  XSRETURN(1);

int
load(SV *obj, const char *fname)
 INIT:
  sm_guts *sm = get_sm(obj);
 CODE:
  sm->clean();
  RETVAL = sm->load(fname);
 OUTPUT:
  RETVAL

SV *
FETCH(SV *self, int idx)
 INIT:
  sm_guts *sm = get_sm(self);
 CODE:
  if ( !sm->m_ns )
    RETVAL = &PL_sv_undef;
  else {
    if ( idx < 0 || idx >= sm->m_ns->size() )
      RETVAL = &PL_sv_undef;
    else {
      const auto &item = sm->m_ns->at(idx);
      RETVAL = newSVpv(item.first.data(), item.first.size());
    }
  }
 OUTPUT:
  RETVAL


BOOT:
 s_sm_pkg = gv_stashpv(s_sm, 0);
 if ( !s_sm_pkg )
    croak("Package %s does not exists", s_sm);