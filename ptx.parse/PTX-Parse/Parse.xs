// #define PERL_NO_GET_CONTEXT
#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"

#include "ppport.h"

#include "../ptx_types.h"

class Perl_ELog: public NV_ELog {
  virtual void verr(const char *format, va_list *ap) {
    vwarn(format, ap);
  }
};

static Perl_ELog s_log;

static int dummy_free(pTHX_ SV* sv, MAGIC* mg) {
 return 0;
}

template <typename T>
static int magic_del(pTHX_ SV* sv, MAGIC* mg) {
    if (mg->mg_ptr) {
        auto *m = (T *)mg->mg_ptr;
        if ( m ) delete m;
        mg->mg_ptr= NULL;
    }
    return 0; // ignored anyway
}

template <typename T>
static T *get_magic_ext(SV *obj, MGVTBL *tab)
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

#ifdef MGf_LOCAL
#define TAB_TAIL ,0
#else
#define TAB_TAIL
#endif

// magic table for PTX::Parse
static const char *s_pp = "PTX::Parse";
static HV *s_pp_pkg = nullptr;
static MGVTBL pp_magic_vt = {
        0, /* get */
        0, /* write */
        0, /* length */
        0, /* clear */
        magic_del<PTXParser_str>,
        0, /* copy */
        0 /* dup */
        TAB_TAIL
};

// magic table for PTX::Parse::Res
static const char *s_ppres = "PTX::Parse::Res";
static HV *s_ppres_pkg = nullptr;
static MGVTBL ppres_magic_vt = {
        0, /* get */
        0, /* write */
        0, /* length */
        0, /* clear */
        magic_del<ParseRes>,
        0, /* copy */
        0 /* dup */
        TAB_TAIL
};

// magic table for PTX::Parse::Instr
static const char *s_ppins = "PTX::Parse::Instr";
static HV *s_ppins_pkg = nullptr;
static MGVTBL ppins_magic_vt = {
        0, /* get */
        0, /* write */
        0, /* length */
        0, /* clear */
        dummy_free, // not destructors for PTXIns - they are static const objects
        0, /* copy */
        0 /* dup */
        TAB_TAIL
};


MODULE = PTX::Parse		PACKAGE = PTX::Parse

void
new(obj_or_pkg)
  SV *obj_or_pkg
 INIT:
  HV *pkg = NULL;
  SV *objref= NULL;
  int ok = 1;
 PPCODE:
  if (SvPOK(obj_or_pkg) && (pkg= gv_stashsv(obj_or_pkg, 0))) {
    if (!sv_derived_from(obj_or_pkg, s_pp)) {
      ok = 0;
      croak("Package %s does not derive from %s", SvPV_nolen(obj_or_pkg), s_pp);
    }
  } else {
    ok = 0;
    croak("new: first arg must be package name or blessed object");
  }
  if ( !ok ) {
    ST(0) = &PL_sv_undef;
  } else {
    PTXParser_str *res = new PTXParser_str(nullptr);
    res->m_elog = &s_log;
    SV *msv = newSViv(0);
    objref= sv_2mortal(newRV_noinc(msv));
    sv_bless(objref, pkg);
    ST(0)= objref;
    // attach magic
    sv_magicext(msv, NULL, PERL_MAGIC_ext, &pp_magic_vt, (const char*)res, 0);
  }
  XSRETURN(1);

void
parse(SV *obj, const char *str, int track_ops = 0)
 INIT:
   SV *objref= NULL;
   PTXParser_str *p= get_magic_ext<PTXParser_str>(obj, &pp_magic_vt);
 PPCODE:
   auto res = p->parse(str, track_ops);
   if ( !res ) {
    ST(0) = &PL_sv_undef;
   } else {
     SV *msv = newSViv(0);
     objref= sv_2mortal(newRV_noinc(msv));
     sv_bless(objref, s_ppres_pkg);
     ST(0) = objref;
     // attach magic
     sv_magicext(msv, NULL, PERL_MAGIC_ext, &ppres_magic_vt, (const char*)res, 0);
   }
   XSRETURN(1);

SV *
rems(SV *obj)
 INIT:
   PTXParser_str *p= get_magic_ext<PTXParser_str>(obj, &pp_magic_vt);
 CODE:
   auto rem = p->rem_attrs();
   if ( rem.empty() )
     RETVAL = &PL_sv_undef;
   else {
     // make hash where key is index and value is string
     HV *hv = newHV();
     for ( auto &r: rem ) {
       hv_store_ent(hv, newSVuv(r.first), newSVpv( r.second.second.data(), r.second.second.size() ), 0);
     }
     RETVAL = newRV_noinc((SV*)hv);
   }
 OUTPUT:
  RETVAL

SV *
pred(SV *obj)
ALIAS:
  PTX::Parse::tail = 1
 INIT:
   PTXParser_str *p= get_magic_ext<PTXParser_str>(obj, &pp_magic_vt);
 CODE:
   auto pred = 1 == ix ? p->tail() : p->pred();
   if ( pred.empty() )
     RETVAL = &PL_sv_undef;
   else
     RETVAL = newSVpv( pred.data(), pred.size() );
 OUTPUT:
  RETVAL

SV *
body(SV *obj)
 INIT:
   PTXParser_str *p= get_magic_ext<PTXParser_str>(obj, &pp_magic_vt);
 CODE:
   auto body = p->body();
   if ( body.empty() )
     RETVAL = &PL_sv_undef;
   else
     RETVAL = newSVpv( body.data(), body.size() );
 OUTPUT:
  RETVAL

MODULE = PTX::Parse		PACKAGE = PTX::Parse::Res

SV *
types(SV *obj)
 INIT:
   ParseRes *p= get_magic_ext<ParseRes>(obj, &ppres_magic_vt);
 CODE:
   if ( p->types.empty() )
     RETVAL = &PL_sv_undef;
   else {
     AV *av = newAV();
     for ( auto &r: p->types ) {
       av_push(av, newSVpv( r.data(), r.size() ));
     }
     RETVAL = newRV_noinc((SV*)av);
   }
 OUTPUT:
  RETVAL

SV *
attrs(SV *obj)
 INIT:
   ParseRes *p= get_magic_ext<ParseRes>(obj, &ppres_magic_vt);
 CODE:
  if ( p->attrs.empty() )
     RETVAL = &PL_sv_undef;
  else { // there is not multimap in perl (well, I know about https://metacpan.org/pod/Hash::MultiValue)
    // so just return av with [ key, name ]
    AV *av = newAV();
    for ( auto a: p->attrs ) {
      AV *curr = newAV();
      av_push(curr, newSViv(a.first));
      av_push(curr, newSVpv( a.second.data(), a.second.size() ));
      av_push(av, newRV_noinc((SV*)curr));
    }
    RETVAL = newRV_noinc((SV*)av);
  }
 OUTPUT:
  RETVAL

SV *
instrs(SV *obj)
 INIT:
   ParseRes *p= get_magic_ext<ParseRes>(obj, &ppres_magic_vt);
 CODE:
  if ( p->forms.empty() )
     RETVAL = &PL_sv_undef;
   else {
     AV *av = newAV();
     for ( auto &r: p->forms ) {
       // instrs are const object so it's safe to separeate them from ParseRes
       SV *msv = newSViv(0);
       SV *objref= newRV_noinc(msv);
       sv_bless(objref, s_ppins_pkg);
       // attach magic
       sv_magicext(msv, NULL, PERL_MAGIC_ext, &ppins_magic_vt, (const char*)r, 0);
       av_push(av, (SV*)objref);
     }
     RETVAL = newRV_noinc((SV*)av);
   }
 OUTPUT:
  RETVAL

MODULE = PTX::Parse		PACKAGE = PTX::Parse::Instr

SV *
name(SV *obj)
 INIT:
   PTXIns *i = get_magic_ext<PTXIns>(obj, &ppins_magic_vt);
 CODE:
   RETVAL = newSVpv(i->name, strlen(i->name));
 OUTPUT:
  RETVAL

int
ln(SV *obj)
 INIT:
   PTXIns *i = get_magic_ext<PTXIns>(obj, &ppins_magic_vt);
 CODE:
   RETVAL = i->ln;
 OUTPUT:
  RETVAL

SV *
fmt(SV *obj)
ALIAS:
  PTX::Parse::Instr::types = 1
 INIT:
   PTXIns *i = get_magic_ext<PTXIns>(obj, &ppins_magic_vt);
 CODE:
   auto what = ix == 1 ? i->ops : i->fmt;
   if ( !what )
     RETVAL = &PL_sv_undef;
   else
     RETVAL = newSVpv(what, strlen(what));
 OUTPUT:
  RETVAL

int
mask(SV *obj, int mj, int mn)
 INIT:
   PTXIns *i = get_magic_ext<PTXIns>(obj, &ppins_magic_vt);
 CODE:
   RETVAL = i->has_bit(mj, mn);
 OUTPUT:
  RETVAL


BOOT:
{
 s_pp_pkg = gv_stashpv(s_pp, 0);
 if ( !s_pp_pkg )
    croak("Package %s does not exists", s_pp);
 s_ppres_pkg = gv_stashpv(s_ppres, 0);
 if ( !s_ppres_pkg )
    croak("Package %s does not exists", s_ppres);
 s_ppins_pkg = gv_stashpv(s_ppins, 0);
 if ( !s_ppins_pkg )
    croak("Package %s does not exists", s_ppins);
}

