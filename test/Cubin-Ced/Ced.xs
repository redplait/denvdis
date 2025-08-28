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
    curr_dis.first = nullptr;
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
  int width() const {
    return m_width;
  }
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
  int set_off(UV off) {
    if ( m_state < WantOff ) return 0;
    int res = _verify_off(off);
    if ( !res ) reset_ins();
    return res;
  }
  // instruction properties
  SV *ins_line() const {
    if ( !curr_dis.first ) return &PL_sv_undef;
    return newSViv(curr_dis.first->line);
  }
  SV *ins_alt() const {
    if ( !curr_dis.first ) return &PL_sv_undef;
    return newSViv(curr_dis.first->alt);
  }
  SV *ins_setp() const {
    if ( !curr_dis.first ) return &PL_sv_undef;
    return newSViv(curr_dis.first->setp);
  }
  SV *ins_brt() const {
    if ( !curr_dis.first ) return &PL_sv_undef;
    return newSViv(curr_dis.first->brt);
  }
  SV *ins_name() const {
    if ( !curr_dis.first ) return &PL_sv_undef;
    return newSVpv(curr_dis.first->name, strlen(curr_dis.first->name));
  }
  SV *ins_class() const {
    if ( !curr_dis.first ) return &PL_sv_undef;
    return newSVpv(curr_dis.first->cname, strlen(curr_dis.first->cname));
  }
  bool has_ins() const {
    return (m_rend != nullptr) && (curr_dis.first != nullptr);
  }
  bool ins_mask(std::string &res) {
    if ( !has_ins() ) return false;
    return m_dis->gen_mask(res);
  }
  bool ins_text(std::string &res) {
    if ( !has_ins() ) return false;
    render(m_rend, res, curr_dis.first, cex(), nullptr, 1);
    return !res.empty();
  }
  // return ref to hash where key is predicate name
  SV *ins_pred() {
    if ( !has_ins() ) return &PL_sv_undef;
    if ( !curr_dis.first->predicated ) return &PL_sv_undef;
    HV *hv = newHV();
    for ( auto &pred: *curr_dis.first->predicated ) {
      int res = pred.second(cex());
      if ( res >= 0 && m_vq && cmp(pred.first, "VQ") ) {
        auto name = m_vq(res);
        if ( name ) {
          hv_store(hv, "VQ", 2, newSVpv(name, strlen(name)), 0 );
          continue;
        }
      }
      hv_store(hv, pred.first.data(), pred.first.size(), newSViv(res), 0);
    }
    return newRV_noinc((SV*)hv);
  }
 protected:
  void reset_ins() {
    m_rend = nullptr;
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

#define EXPORT_ENUM(e,x) newCONSTSUB(stash, #x, new_enum_dualvar(aTHX_ e::x, newSVpvs_share(#x)));
static SV * new_enum_dualvar(pTHX_ IV ival, SV *name) {
        SvUPGRADE(name, SVt_PVNV);
        SvIV_set(name, ival);
        SvIOK_on(name);
        SvREADONLY_on(name);
        return name;
}

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

SV *
off(SV *obj, UV off)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->set_off(off) ? &PL_sv_yes : &PL_sv_no;
 OUTPUT:
  RETVAL

int
width(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->width();
 OUTPUT:
  RETVAL

SV *
ins_name(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->ins_name();
 OUTPUT:
  RETVAL

SV *
ins_class(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->ins_class();
 OUTPUT:
  RETVAL

SV *
ins_line(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->ins_line();
 OUTPUT:
  RETVAL

SV *
ins_alt(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->ins_alt();
 OUTPUT:
  RETVAL

SV *
ins_mask(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
   std::string mask;
 CODE:
   bool res = e->ins_mask(mask);
   if ( !res )
     RETVAL = &PL_sv_undef;
   else
     RETVAL = newSVpv(mask.c_str(), mask.size());
 OUTPUT:
  RETVAL

SV *
ins_text(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
   std::string mask;
 CODE:
   bool res = e->ins_text(mask);
   if ( !res )
     RETVAL = &PL_sv_undef;
   else
     RETVAL = newSVpv(mask.c_str(), mask.size());
 OUTPUT:
  RETVAL

SV *
ins_pred(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->ins_pred();
 OUTPUT:
  RETVAL

BOOT:
 s_ca_pkg = gv_stashpv(s_ca, 0);
 if ( !s_ca_pkg )
    croak("Package %s does not exists", s_ca);
 // add enums from nv_types.h
 HV *stash = gv_stashpvn(s_ca, 10, 1);
 EXPORT_ENUM(NVP_ops, IDEST)
 EXPORT_ENUM(NVP_ops, IDEST2)
 EXPORT_ENUM(NVP_ops, ISRC_A)
 EXPORT_ENUM(NVP_ops, ISRC_B)
 EXPORT_ENUM(NVP_ops, ISRC_C)
 EXPORT_ENUM(NVP_ops, ISRC_E)
 EXPORT_ENUM(NVP_type, INTEGER)
 EXPORT_ENUM(NVP_type, SIGNED_INTEGER)
 EXPORT_ENUM(NVP_type, UNSIGNED_INTEGER)
 EXPORT_ENUM(NVP_type, FLOAT)
 EXPORT_ENUM(NVP_type, DOUBLE)
 EXPORT_ENUM(NVP_type, GENERIC_ADDRESS)
 EXPORT_ENUM(NVP_type, SHARED_ADDRESS)
 EXPORT_ENUM(NVP_type, LOCAL_ADDRESS)
 EXPORT_ENUM(NVP_type, TRAM_ADDRESS)
 EXPORT_ENUM(NVP_type, LOGICAL_ATTR_ADDRESS)
 EXPORT_ENUM(NVP_type, PHYSICAL_ATTR_ADDRESS)
 EXPORT_ENUM(NVP_type, GENERIC)
 EXPORT_ENUM(NVP_type, CONSTANT_ADDRESS)
 EXPORT_ENUM(NVP_type, VILD_INDEX)
 EXPORT_ENUM(NVP_type, VOTE_INDEX)
 EXPORT_ENUM(NVP_type, STP_INDEX)
 EXPORT_ENUM(NVP_type, PIXLD_INDEX)
 EXPORT_ENUM(NVP_type, PATCH_OFFSET_ADDRESS)
 EXPORT_ENUM(NVP_type, RAW_ISBE_ACCESS)
 EXPORT_ENUM(NVP_type, GLOBAL_ADDRESS)
 EXPORT_ENUM(NVP_type, TEX)
 EXPORT_ENUM(NVP_type, GS_STATE)
 EXPORT_ENUM(NVP_type, SURFACE_COORDINATES)
 EXPORT_ENUM(NVP_type, FP16SIMD)
 EXPORT_ENUM(NVP_type, BINDLESS_CONSTANT_ADDRESS)
 EXPORT_ENUM(NVP_type, VERTEX_HANDLE)
 EXPORT_ENUM(NVP_type, MEMORY_DESCRIPTOR)
 EXPORT_ENUM(NVP_type, FP8SIMD)
 EXPORT_ENUM(NV_Format, NV_BITSET)
 EXPORT_ENUM(NV_Format, NV_UImm)
 EXPORT_ENUM(NV_Format, NV_SImm)
 EXPORT_ENUM(NV_Format, NV_SSImm)
 EXPORT_ENUM(NV_Format, NV_RSImm)
 EXPORT_ENUM(NV_Format, NV_F64Imm)
 EXPORT_ENUM(NV_Format, NV_F16Imm)
 EXPORT_ENUM(NV_Format, NV_F32Imm)
 EXPORT_ENUM(NV_Brt, BRT_CALL)
 EXPORT_ENUM(NV_Brt, BRT_RETURN)
 EXPORT_ENUM(NV_Brt, BRT_BRANCH)
 EXPORT_ENUM(NV_Brt, BRT_BRANCHOUT)
