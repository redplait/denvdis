// #define PERL_NO_GET_CONTEXT
#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"

#include "ppport.h"

// to avoid conflicts with std regex
#undef do_open
#undef do_close

#include "ced_base.h"
#include "elf.inc"

int opt_d = 0,
  opt_h = 0,
  opt_m = 0,
  skip_final_cut = 0,
  skip_op_parsing = 0,
  opt_t = 0,
  opt_k = 0,
  opt_v = 0;

class Perl_ELog: public NV_ELog {
  virtual void verr(const char *format, va_list *ap) {
    vwarn(format, ap);
  }
};

class Ced_perl: public CEd_base {
 public:
  Ced_perl(IElf *e) {
    m_elog = new Perl_ELog;
    m_e = e;
    e->add_ref();
  }
  virtual ~Ced_perl() {
    if ( m_e ) m_e->release();
    if ( m_elog ) delete m_elog;
  }
  // patch virtual methods
   virtual void patch_error(const char *what) override {
     my_warn("cannot patch %s\n", what);
   };
   virtual void patch_error(const std::string_view &what) override {
     int w_len = int(what.length());
     my_warn("cannot patch %.*s\n", w_len, what.data());
   }
   virtual void patch_tab_error(const char *what) override {
     my_warn("cannot patch tab value %s\n", what);
   }
  // interface to perl
  int sef_func(const char *fname) {
    reset_ins();
    Ced_named::const_iterator fiter = m_named.find({ fname, strlen(fname) });
    if ( fiter == m_named.end() ) {
      Err("unknown fn: %s\n", fname);
      return 0;
    }
    return setup_f(fiter, fname);
  }
  int set_section(int idx) {
    reset_ins();
    auto siter = m_code_sects.find(idx);
    if ( siter == m_code_sects.end() ) {
      Err("section %d don't have code\n", idx);
      return 0;
    }
    return setup_s(idx);
  }
  int set_section(const char *sname, STRLEN len) {
    reset_ins();
    // try to find section by sname
    auto siter = m_named_cs.find({ sname, len });
    if ( siter == m_named_cs.end() ) {
      Err("section %.*s don't have code\n", len, sname);
      return 0;
    }
    return setup_s(siter->second);
  }
 protected:
  void reset_ins() {
    curr_dis.first = nullptr;
    curr_dis.second.clear();
  }
  IElf *m_e;
};

#ifdef MGf_LOCAL
#define TAB_TAIL ,0
#else
#define TAB_TAIL
#endif

// magic table for Cubin::Attrs
static const char *s_ca = "Cubin::Ced";
static HV *s_ca_pkg = nullptr;
static MGVTBL ca_magic_vt = {
        0, /* get */
        0, /* write */
        0, /* length */
        0, /* clear */
        magic_del<Ced_perl>,
        0, /* copy */
        0 /* dup */
        TAB_TAIL
};

#define DWARF_EXT(vtab, pkg, what) \
  fake = newAV(); \
  objref = newRV_noinc((SV*)fake); \
  sv_bless(objref, pkg); \
  magic = sv_magicext((SV*)fake, NULL, PERL_MAGIC_ext, &vtab, (const char *)what, 0); \
  SvREADONLY_on((SV*)fake); \
  ST(0) = objref; \
  XSRETURN(1);

MODULE = Cubin::Ced		PACKAGE = Cubin::Ced

void
new(obj_or_pkg, SV *elsv)
  SV *obj_or_pkg
 INIT:
  HV *pkg = NULL;
  SV *msv;
  SV *objref= NULL;
  struct IElf *e= extract(elsv);
  Ced_perl *res = NULL;
  int ok = 1;
 PPCODE:
  if (SvPOK(obj_or_pkg) && (pkg= gv_stashsv(obj_or_pkg, 0))) {
    if (!sv_derived_from(obj_or_pkg, s_ca)) {
      ok = 0;
      croak("Package %s does not derive from %s", SvPV_nolen(obj_or_pkg), s_ca);
    }
  } else {
    ok = 0;
    croak("new: first arg must be package name or blessed object");
  }
  if ( !ok ) {
    ST(0) = &PL_sv_undef;
  } else {
    res = new Ced_perl(e);
    if ( !res->open(e->rdr, e->fname.c_str()) || !res->prepare(e->fname.c_str()) ) {
      delete res;
      ST(0) = &PL_sv_undef;
    } else {
      msv = newSViv(0);
      objref= sv_2mortal(newRV_noinc(msv));
      sv_bless(objref, pkg);
      ST(0)= objref;
      // attach magic
      sv_magicext(msv, NULL, PERL_MAGIC_ext, &ca_magic_vt, (const char*)res, 0);
    }
  }
  XSRETURN(1);

SV *
set_f(SV *obj, const char *fname)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->sef_func(fname) ? &PL_sv_yes : &PL_sv_no;
 OUTPUT:
  RETVAL

SV *
set_s(SV *obj, SV *sv)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   int res = 0;
   if ( SvPOK(sv) ) {
     STRLEN len;
     auto p =  SvPVbyte(sv, len);
     res = e->set_section(p, len);
   } else if ( SvIOK(sv) ) res = e->set_section(SvIV(sv));
   else my_warn("set_s: unknown arg type");
   RETVAL = res ? &PL_sv_yes : &PL_sv_no;
 OUTPUT:
  RETVAL

BOOT:
 s_ca_pkg = gv_stashpv(s_ca, 0);
 if ( !s_ca_pkg )
    croak("Package %s does not exists", s_ca);