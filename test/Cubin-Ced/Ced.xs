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

typedef std::pair<const render_base*, std::list<const render_named *> > RItem;
typedef std::vector<RItem> RItems;

static SV *make_etail(const std::list<const render_named *> &et) {
  if ( et.empty() ) return &PL_sv_undef;
  AV *av = newAV();
  for ( auto ea: et ) {
    av_push(av, newSVpv( ea->name, strlen(ea->name) ));
  }
  return newRV_noinc((SV*)av);
}

static SV *make_vbase(const ve_base *vb) {
  AV *av = newAV();
  // [0] - type
  // [1] - pfx
  // [2] - arg if presents
  av_push(av, newSViv(vb->type));
  if ( vb->pfx )
    av_push(av, newSVpv(&vb->pfx, 1));
  else
    av_push(av, &PL_sv_undef);
  if ( vb->arg )
    av_push(av, newSVpv( vb->arg, strlen(vb->arg)) );
  return newRV_noinc((SV*)av);
}

static SV *make_vblist(const std::list<ve_base> &vl) {
  AV *av = newAV();
  for ( auto &v: vl ) av_push(av, make_vbase(&v));
  return newRV_noinc((SV*)av);
}

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
    for ( auto &iter: m_cached_hvs ) SvREFCNT_dec(iter.second);
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
  inline unsigned long get_flush() const {
    return flush_cnt;
  }
  inline unsigned long get_rdr() const {
    return rdr_cnt;
  }
  inline bool is_dirty() const {
    return block_dirty;
  }
  int sm_num() const {
    return m_sm;
  }
  const char *sm_name() const {
    return m_sm_name;
  }
  SV *extract_instrs() const;
  int sef_func(const char *fname) {
    if ( has_ins() && block_dirty ) flush_buf();
    reset_ins();
    Ced_named::const_iterator fiter = m_named.find({ fname, strlen(fname) });
    if ( fiter == m_named.end() ) {
      Err("unknown fn: %s\n", fname);
      return 0;
    }
    return setup_f(fiter, fname);
  }
  int set_section(int idx) {
    if ( has_ins() && block_dirty ) flush_buf();
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
    if ( !flush_buf() ) return 0;
    int res = _verify_off(off);
    if ( !res ) reset_ins();
    return res;
  }
  SV *get_start() {
    if ( m_state < WantOff ) return &PL_sv_undef;
    return newSVuv(m_obj_off);
  }
  SV *get_end() {
    if ( m_state < WantOff ) return &PL_sv_undef;
    return newSVuv(m_obj_off + m_obj_size);
  }
  int next() {
    if ( m_state < WantOff ) return 0;
    int res = _next_off();
    if ( !res ) reset_ins();
    return res;
  }
  SV *get_off() {
    if ( !ins() ) return &PL_sv_undef;
    return newSVuv(m_dis->offset());
  }
  SV *get_ctrl() {
    if ( !ins() ) return &PL_sv_undef;
    if ( m_width == 128 ) return &PL_sv_no;
    uint8_t c = 0, o = 0;
    m_dis->get_ctrl(c, o);
    return newSVuv(c);
  }
  SV *get_opcode() {
    if ( !ins() ) return &PL_sv_undef;
    if ( m_width == 128 ) return &PL_sv_no;
    uint8_t c = 0, o = 0;
    m_dis->get_ctrl(c, o);
    return newSVuv(o);
  }
  // patch methods
  SV *nop();
  int replace(const char *s);
  int patch_pred(int is_not, int v) {
    if ( !has_ins() ) return 0;
    return _patch_pred(v, is_not, false);
  }
  int patch_field(const char *fname, SV *v);
  int patch_tab(int t_idx, int v);
  int patch_cb(unsigned long v1, unsigned long v2);
  // instruction properties
  SV *ins_line() const {
    if ( !ins() ) return &PL_sv_undef;
    return newSViv(ins()->line);
  }
  SV *ins_alt() const {
    if ( !ins() ) return &PL_sv_undef;
    return newSViv(ins()->alt);
  }
  SV *ins_setp() const {
    if ( !ins() ) return &PL_sv_undef;
    return newSViv(ins()->setp);
  }
  SV *ins_brt() const {
    if ( !ins() ) return &PL_sv_undef;
    return newSViv(ins()->brt);
  }
  SV *ins_scbd() const {
    if ( !ins() ) return &PL_sv_undef;
    return newSViv(ins()->scbd);
  }
  SV *ins_scbd_type() const {
    if ( !ins() ) return &PL_sv_undef;
    return newSViv(ins()->scbd_type);
  }
  SV *ins_name() const {
    if ( !ins() ) return &PL_sv_undef;
    return newSVpv(ins()->name, strlen(ins()->name));
  }
  SV *ins_class() const {
    if ( !ins() ) return &PL_sv_undef;
    return newSVpv(ins()->cname, strlen(ins()->cname));
  }
  SV *ins_target() const {
    if ( !ins() || !ins()->target_index ) return &PL_sv_undef;
    return newSVpv(ins()->target_index, strlen(ins()->target_index));
  }
  SV *ins_cc() const {
    if ( !ins() || !ins()->cc_index ) return &PL_sv_undef;
    return newSVpv(ins()->cc_index, strlen(ins()->cc_index));
  }
  SV *ins_dual() const {
    if ( !has_ins() ) return &PL_sv_undef;
    return check_dual(cex()) ? &PL_sv_yes : &PL_sv_no;
  }
  SV *check_false() const {
    if ( !has_ins() ) return &PL_sv_undef;
    return always_false(ins(), m_rend, cex()) ? &PL_sv_yes : &PL_sv_no;
  }
  SV *check_lut() const {
    if ( !has_ins() ) return &PL_sv_undef;
    int idx = 0;
    if ( NV_renderer::check_lut(ins(), m_rend, cex(), idx) ) return newSViv(idx);
    return &PL_sv_undef;
  }
  SV *lut_name(int idx) const {
    auto s = get_lut(idx);
    if ( !s ) return &PL_sv_undef;
    return newSVpv(s, strlen(s));
  }
  SV *ins_sidl() const {
    if ( !ins() || !ins()->sidl_name ) return &PL_sv_undef;
    return newSVpv(ins()->sidl_name, strlen(ins()->sidl_name));
  }
  bool has_pending_tabs() {
    return !m_inc_tabs.empty();
  }
  bool has_ins() const {
    return (m_rend != nullptr) && (curr_dis.first != nullptr);
  }
  bool ins_mask(std::string &res) {
    if ( !has_ins() ) return false;
    res = ins()->mask;
    return true;
  }
  bool gen_mask(std::string &res) {
    if ( !has_ins() ) return false;
    return m_dis->gen_mask(res);
  }
  bool ins_text(std::string &res) {
    if ( !has_ins() ) return false;
    render(m_rend, res, ins(), cex(), nullptr, 1);
    return !res.empty();
  }
  HV *make_kv();
  // return ref to hash where key is predicate name
  SV *ins_pred() {
    if ( !has_ins() ) return &PL_sv_undef;
    if ( !ins()->predicated ) return &PL_sv_undef;
    HV *hv = newHV();
    for ( auto &pred: *ins()->predicated ) {
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
  // return ref to hash where key is enum NVP_ops and value is ref to array where
  // [0] - type - enum NVP_type
  // [1..n] - field names
  SV *ins_prop() {
    if ( !has_ins() ) return &PL_sv_undef;
    if ( !ins()->props ) return &PL_sv_undef;
    HV *hv = newHV();
    for ( size_t i = 0; i < ins()->props->size(); ++i ) {
      auto prop = get_it(*ins()->props, i);
      hv_store_ent(hv, newSViv(prop->op), make_prop(prop), 0);
    }
    return newRV_noinc((SV*)hv);
  }
  bool make_render(RItems &);
  SV *extract_cb();
  SV *extract_efield(const char *);
  SV *extract_efields();
  SV *extract_vfield(const char *);
  SV *extract_vfields();
  SV *make_enum(const char *);
  // tabs
  SV *tab_count() {
    if ( !has_ins() ) return &PL_sv_undef;
    return newSViv(ins()->tab_fields.size());
  }
  bool get_tab(IV, SV **n, SV **d);
 protected:
  SV *make_prop(const NV_Prop *prop);
  SV *make_enum_arr(const nv_eattr *ea);
  HV *make_enum(const std::unordered_map<int, const char *> *);
  SV *make_vfield(const nv_vattr &);
  SV *fill_simple_tab(const std::unordered_map<int, const unsigned short *> *);
  SV *fill_tab(const std::unordered_map<int, const unsigned short *> *, size_t);
  IElf *m_e;
  // cached enums (HV *), key is nv_eattr->ename
  std::unordered_map<std::string_view, HV *> m_cached_hvs;
};

int Ced_perl::replace(const char *s)
{
  int add_res = add(s, 0);
  if ( !add_res || m_forms.empty() ) {
    Err("cannot parse %s\n", s);
    return 0;
  }
  const one_form *of = &m_forms.at(0);
  if ( of->label_op ) {
    Err("instructions with labels not supported\n");
    return 0;
  }
  NV_extracted kv;
  if ( !_extract_full(kv, of) ) {
    Err("cannot extract values for %s\n", s);
    return 0;
  }
  copy_tail_values(of->instr, of->rend, cex(), kv);
  if ( !generic_ins(of->instr, kv) ) return 0;
  if ( !flush_buf() ) {
    Err("instr %s flush failed\n", s);
    return 0;
  }
  reset_ins();
  return 1;
}

SV *Ced_perl::nop()
{
  if ( !has_ins() ) return &PL_sv_undef;
  if ( !m_nop ) {
    Err("warning: cannot patch nop\n");
    return &PL_sv_no;
  }
  NV_extracted out_res;
  copy_tail_values(ins(), m_nop_rend, cex(), out_res);
  if ( !generic_ins(m_nop, out_res) ) return &PL_sv_no;
  reset_ins();
  if ( !flush_buf() ) {
    Err("nop flush failed\n");
    return &PL_sv_no;
  }
  return &PL_sv_yes;
}

bool Ced_perl::make_render(RItems &res) {
  if ( !has_ins() || !m_rend ) return false;
  for ( auto r: *m_rend ) {
    if ( r->type == R_enum ) {
      const render_named *rn = (const render_named *)r;
      auto ea = find(ins()->eas, rn->name);
      if ( ea && ea->ea->ignore ) {
        res.back().second.push_back(rn);
        continue;
      }
    }
    res.push_back( { r, {} });
  }
  return !res.empty();
}

// return array of insns mnemonic names
SV *Ced_perl::extract_instrs() const {
  if ( !m_sorted ) return &PL_sv_undef;
  AV *av = newAV();
  for ( auto it = m_sorted->begin(); it != m_sorted->end(); ++it )
    av_push(av, newSVpv( it->first.data(), it->first.size() ));
  return newRV_noinc((SV*)av);
}

// patched CEd::process_p, too many changes to extract parts in CEd_base
int Ced_perl::patch_field(const char *fname, SV *v)
{
  std::string p = fname;
  const NV_tab_fields *tab = nullptr;
  const NV_field *field = nullptr;
  const nv_eattr *ea = nullptr;
  const nv_vattr *va = nullptr;
  int cb_idx = 0, tab_idx = 0;
  bool ctr = p == "Ctrl";
  if ( ctr && m_width == 128 ) {
    Err("Ctrl not supported for 128bit\n");
    return 0;
  }
  const NV_cbank *cb = is_cb_field(ins(), p, cb_idx);
  if ( !ctr && !cb ) {
    tab = is_tab_field(ins(), p, tab_idx);
    if ( !tab ) {
      field = std::lower_bound(ins()->fields.begin(), ins()->fields.end(), p,
       [](const NV_field &f, const std::string &w) {
         return f.name < w;
      });
      if ( field == ins()->fields.end() ) {
        Err("unknown field %s\n", fname);
        return 0;
      }
      // cool, some real field
      ea = find_ea(ins(), p);
      if ( !ea && ins()->vas )
        va = find(ins()->vas, p);
    }
  }
  m_v = 0;
  // check what we have and what kind of SV
  if ( va || ctr ) { // some imm value
    if ( SvPOK(v) ) { // string
      STRLEN len;
      auto pv = SvPVbyte(v, len);
      std::string_view sv{ pv, len };
      if ( !parse_num(va->kind, sv) ) {
        Err("cannot parse num %.*s\n", len, sv.data());
        return 0;
      }
    } else if ( SvUOK(v) && (va->kind == NV_BITSET || va->kind == NV_UImm) )
     m_v = SvUV(v);
    else if ( SvIOK(v) )
     m_v = SvIV(v);
    else {
      int skip = 1;
      if ( !ctr && SvNOK(v) && (va->kind == NV_F64Imm || va->kind == NV_F32Imm || va->kind == NV_F16Imm) ) {
        double d = SvNV(v);
        if ( va->kind == NV_F64Imm ) m_v = *(uint64_t *)&d;
        else if ( va->kind == NV_F32Imm ) {
          float fl = (float)d;
          *(float *)&m_v = fl;
        } else if ( va->kind == NV_F16Imm ) {
          *(float *)&m_v = fp16_ieee_from_fp32_value(float(d));
        }
        skip = 0;
      }
      if ( skip ) {
        Err("Unknown SV type %d in patch", SvTYPE(v));
        return 0;
      }
    }
  } else if ( ea ) {
    if ( SvPOK(v) ) { // string
      STRLEN len;
      auto pv = SvPVbyte(v, len);
      std::string_view sv{ pv, len };
      if ( !m_renums ) {
        Err("no renums for field %s, enum %s\n", p.c_str(), ea->ename);
        return 0;
      }
      auto ed = m_renums->find(ea->ename);
      if ( ed == m_renums->end() ) {
        Err("cannot find renum %s for field %s\n", ea->ename, p.c_str());
        return 0;
      }
      auto edi = ed->second->find(sv);
      if ( edi == ed->second->end() ) {
        Err("cannot find %.*s in enum %s for field %s\n", len, sv.data(), ea->ename, p.c_str());
        return 0;
      }
      m_v = edi->second;
    } else if ( SvIOK(v) ) {
      m_v = SvIV(v);
      auto ei = ea->em->find(m_v);
      if ( ei == ea->em->end() ) {
        Err("value %lX for field %s not in enum %s\n", m_v, p.c_str(), ea->ename);
        return 0;
      }
    } else {
      Err("Unknown SV type %d for enum %s in patch", SvTYPE(v), ea->ename);
      return 0;
    }
  } else {
    Err("unknown field %s, ignoring\n", p.c_str());
    return 0;
  }
  // check how this field should be patched
  if ( ctr ) {
    int res = m_dis->put_ctrl(m_v);
    if ( res ) block_dirty = true;
    return res;
  }
  if ( field ) {
    int res = patch(field, field->scale ? m_v / field->scale : m_v, p.c_str());
    ex()[field->name] = m_v;
    return res;
  }
  if ( cb ) {
    unsigned long c1 = 0, c2 = 0;
    auto kv = ex();
    if ( !cb_idx ) {
      c1 = m_v;
      // store into current kv bcs next p can patch second cbank value
      kv[cb->f1] = m_v;
      c2 = value_or_def(ins(), cb->f2, kv);
    } else {
      c2 = m_v;
      // store into current kv bcs next p can patch second cbank value
      kv[cb->f2] = m_v;
      c1 = value_or_def(ins(), cb->f1, kv);
    }
    return generic_cb(ins(), c1, c2, true);
  }
  if ( tab ) {
    // check if provided value is valid for table
    std::vector<unsigned short> tab_row;
    if ( make_tab_row(opt_v, ins(), tab, cex(), tab_row, tab_idx) ) return 0;
    tab_row[tab_idx] = (unsigned short)m_v;
    int tab_value = 0;
    if ( !ins()->check_tab(tab->tab, tab_row, tab_value) ) {
      NV_extracted &kv = ex();
      kv[p] = m_v;
      m_inc_tabs.insert(tab);
      if ( opt_v ) {
        Err("Warning: value %ld for %s invalid in table\n", m_v, p.c_str());
        dump_tab_fields(tab);
      }
      return 1;
    } else
     return patch(tab, tab_value, p.c_str());
  } else {
    Err("dont know how to patch %s\n", p.c_str());
    return 0;
  }
  return 1;
}

int Ced_perl::patch_tab(int t_idx, int v)
{
  // first check if we have tab with index t_idx
  if ( !ins() ) return 0;
  if ( t_idx < 0 || t_idx >= ins()->tab_fields.size() ) return 0;
  auto &f = get_it(ins()->tab_fields, t_idx);
  // check if v is valid value for this tab
  auto ti = f->tab->find(v);
  if ( ti == f->tab->end() ) {
    Err("invalid value %d for tab %d", v, t_idx);
    return 0;
  }
  std::string tmp = "tab ";
  tmp += std::to_string(t_idx);
  if ( !patch(f, v, tmp.c_str()) ) return 0;
  // first remove pending tabs
  m_inc_tabs.erase(f);
  // then sync kv with array in ti
  auto a = ti->second;
  for ( size_t ai = 0; ai < f->fields.size(); ai++ )
  {
    auto &sv = get_it(f->fields, ai);
    ex()[sv] = a[1 + ai];
  }
  return 1;
}

int Ced_perl::patch_cb(unsigned long v1, unsigned long v2)
{
  if ( !ins() || !ins()->cb_field ) return 0;
  return generic_cb(ins(), v1, v2, true);
}

// extract tab with index idx
// put ref to array with names into n
// put ref to hash into d
bool Ced_perl::get_tab(IV idx, SV **n, SV **d) {
  if ( !has_ins() ) return false;
  if ( idx < 0 || idx >= ins()->tab_fields.size() ) return false;
  auto t = get_it(ins()->tab_fields, idx);
  // fill names
  AV *av = newAV();
  for ( size_t ni = 0; ni < t->fields.size(); ++ni ) {
    auto &nf = get_it(t->fields, ni);
    av_push(av, newSVpv( nf.data(), nf.size() ));
  }
  *n = newRV_noinc((SV*)av);
  // dict can be just int key -> int value
  // or int key -> ref to array when fields > 1
  if ( 1 == t->fields.size() )
    *d = fill_simple_tab(t->tab);
  else
    *d = fill_tab(t->tab, t->fields.size());
  return true;
}

SV *Ced_perl::fill_simple_tab(const std::unordered_map<int, const unsigned short *> *t)
{
  HV *hv = newHV();
  for ( auto ei: *t ) {
    hv_store_ent(hv, newSViv(ei.first), newSVuv(ei.second[1]), 0);
  }
  return newRV_noinc((SV*)hv);
}

SV *Ced_perl::fill_tab(const std::unordered_map<int, const unsigned short *> *t, size_t ts)
{
  HV *hv = newHV();
  for ( auto ei: *t ) {
    auto row = ei.second;
    AV *av = newAV();
    for ( size_t i = 1; i < ts; ++i ) av_push(av, newSVuv(row[i]));
    hv_store_ent(hv, newSViv(ei.first), newRV_noinc((SV*)av), 0);
  }
  return newRV_noinc((SV*)hv);
}

// return ref to array of property fields, item at index 0 is type
SV *Ced_perl::make_prop(const NV_Prop *prop) {
  AV *av = newAV();
  av_push(av, newSViv(prop->t));
  for ( size_t i = 0; i < prop->fields.size(); ++i ) {
    auto &field = get_it(prop->fields, 0);
    av_push(av, newSVpv(field.data(), field.size()));
  }
  return newRV_noinc((SV*)av);
}

// return array with couple of fields names, 3rd is scale when non-zero
SV *Ced_perl::extract_cb()
{
  if ( !has_ins() || !ins()->cb_field ) return &PL_sv_undef;
  AV *av = newAV();
  av_push(av, newSVpv(ins()->cb_field->f1.data(), ins()->cb_field->f1.size()) );
  av_push(av, newSVpv(ins()->cb_field->f2.data(), ins()->cb_field->f2.size()) );
  if ( ins()->cb_field->scale )
    av_push(av, newSViv(ins()->cb_field->scale));
  return newRV_noinc((SV*)av);
}

// return hash with nv_eattr, key is field name, value is ref to array:
// 0 - ignore
// 1 - print
// 2 - has_def_value
// 3 - def value if presents
SV *Ced_perl::make_enum_arr(const nv_eattr *ea)
{
  AV *av = newAV();
  av_push(av, ea->ignore ? &PL_sv_yes : &PL_sv_no);
  av_push(av, ea->print ? &PL_sv_yes : &PL_sv_no);
  av_push(av, ea->has_def_value ? &PL_sv_yes : &PL_sv_no);
  if ( ea->has_def_value )
    av_push(av, newSViv(ea->def_value));
  return newRV_noinc((SV*)av);
}

SV *Ced_perl::extract_efields()
{
  if ( !has_ins() || !ins()->eas.size() ) return &PL_sv_undef;
  HV *hv = newHV();
  for ( size_t i = 0; i < ins()->eas.size(); ++i ) {
    auto &ea = get_it(ins()->eas, i);
    hv_store(hv, ea.name.data(), ea.name.size(), make_enum_arr(ea.ea), 0);
  }
  return newRV_noinc((SV*)hv);
}

// if we have width for some field - return [ type, width ]
// else just return type in SViv
SV *Ced_perl::make_vfield(const nv_vattr &v)
{
  SV *t = newSViv(v.kind);
  if ( !ins()->vwidth ) return t;
  auto vw = find(ins()->vwidth, v.name);
  if ( !vw ) return t;
  AV *av = newAV();
  av_push(av, t);
  av_push(av, newSViv(vw->w));
  return newRV_noinc((SV*)av);
}

SV *Ced_perl::extract_vfields()
{
  if ( !has_ins() || !ins()->vas ) return &PL_sv_undef;
  HV *hv = newHV();
  for ( size_t i = 0; i < ins()->vas->size(); ++i ) {
    auto &va = get_it(*ins()->vas, i);
    hv_store(hv, va.name.data(), va.name.size(), make_vfield(va), 0);
  }
  return newRV_noinc((SV*)hv);
}

SV *Ced_perl::extract_vfield(const char *name)
{
  if ( !has_ins() || !ins()->vas ) return &PL_sv_undef;
  std::string_view tmp{ name, strlen(name) };
  auto va = find(ins()->vas, tmp);
  if ( !va ) return &PL_sv_undef;
  return make_vfield(*va);
}

HV *Ced_perl::make_enum(const std::unordered_map<int, const char *> *em)
{
  HV *hv = newHV();
  for ( auto ei: *em ) {
    hv_store_ent(hv, newSViv(ei.first), newSVpv(ei.second, strlen(ei.second)), 0);
  }
  return hv;
}

SV *Ced_perl::extract_efield(const char *name)
{
  if ( !has_ins() ) return &PL_sv_undef;
  std::string_view tmp{ name, strlen(name) };
  auto ea = find(ins()->eas, tmp);
  if ( !ea ) return &PL_sv_undef;
  return make_enum_arr(ea->ea);
}

SV *Ced_perl::make_enum(const char *name)
{
  if ( !has_ins() ) return &PL_sv_undef;
  std::string_view tmp{ name, strlen(name) };
  auto ea = find(ins()->eas, tmp);
  if ( !ea ) return &PL_sv_undef;
  std::string_view key{ ea->ea->ename, strlen( ea->ea->ename ) };
  auto cached_iter = m_cached_hvs.find(key);
  if ( cached_iter != m_cached_hvs.end() ) return newRV_inc((SV *)cached_iter->second);
  HV *curr = make_enum(ea->ea->em);
  if ( !curr ) return &PL_sv_undef;
  // store this hv in cache
  m_cached_hvs[key] = curr;
  return newRV_inc((SV *)curr);
}

HV *Ced_perl::make_kv()
{
  HV *hv = newHV();
  for ( NV_extracted::const_iterator ei = cex().cbegin(); ei != cex().cend(); ++ei ) {
    if ( ins()->vas ) {
      auto va = find(ins()->vas, ei->first);
      if ( va ) {
        // fill value with according format
        if ( va->kind == NV_F64Imm ) {
          auto v = ei->second;
          hv_store(hv, ei->first.data(), ei->first.size(), newSVnv(*(double *)&v), 0);
          continue;
        }
        if ( va->kind == NV_F32Imm ) {
          auto v = ei->second;
          hv_store(hv, ei->first.data(), ei->first.size(), newSVnv(*(float *)&v), 0);
          continue;
        }
        if ( va->kind == NV_F16Imm ) {
          float f32 = fp16_ieee_to_fp32_bits((uint16_t)ei->second);
          hv_store(hv, ei->first.data(), ei->first.size(), newSVnv(f32), 0);
          continue;
        }
        if ( va->kind == NV_SImm || va->kind == NV_SSImm || va->kind == NV_RSImm ) {
          // lets convert signed value to SViv
          long conv = 0;
          if ( check_branch(ins(), ei, conv) || conv_simm(ins(), ei, conv) ) {
            hv_store(hv, ei->first.data(), ei->first.size(), newSViv(conv), 0);
            continue;
          }
        }
      }
    }
    hv_store(hv, ei->first.data(), ei->first.size(), newSVuv(ei->second), 0);
  }
  return hv;
}

#ifdef MGf_LOCAL
#define TAB_TAIL ,0
#else
#define TAB_TAIL
#endif

// magic table for Cubin::Ced
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

// magic table for Cubin::Ced::Render
static const char *s_ca_render = "Cubin::Ced::Render";
static HV *s_ca_render_pkg = nullptr;

template<typename T>
static U32 ced_magic_size(pTHX_ SV* sv, MAGIC* mg) {
    if (mg->mg_ptr) {
        auto *m = (T *)mg->mg_ptr;
        return m->size()-1;
    }
    return 0; // ignored anyway
}

static MGVTBL ca_rend_magic_vt = {
        0, /* get */
        0, /* write */
        ced_magic_size<RItems>, /* length */
        0, /* clear */
        magic_del<RItems>,
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
     auto p = SvPVbyte(sv, len);
     res = e->set_section(p, len);
   } else if ( SvIOK(sv) ) res = e->set_section(SvIV(sv));
   else my_warn("set_s: unknown arg type");
   RETVAL = res ? &PL_sv_yes : &PL_sv_no;
 OUTPUT:
  RETVAL

int
optd(SV *obj, int v)
 CODE:
   RETVAL = opt_d;
   opt_d = v;
 OUTPUT:
  RETVAL

int
optv(SV *obj, int v)
 CODE:
   RETVAL = opt_v;
   opt_v = v;
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

SV *
next(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->next() ? &PL_sv_yes : &PL_sv_no;
 OUTPUT:
  RETVAL

SV *
get_off(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->get_off();
 OUTPUT:
  RETVAL

SV *
start(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->get_start();
 OUTPUT:
  RETVAL

SV *
end(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->get_end();
 OUTPUT:
  RETVAL

SV *
ctrl(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   if ( !e->has_ins() )
     RETVAL = &PL_sv_undef;
   else
     RETVAL = e->get_ctrl();
 OUTPUT:
  RETVAL

SV *
opcode(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   if ( !e->has_ins() )
     RETVAL = &PL_sv_undef;
   else
     RETVAL = e->get_opcode();
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
instrs(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->extract_instrs();
 OUTPUT:
  RETVAL

int
sm_num(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->sm_num();
 OUTPUT:
  RETVAL

const char *
sm_name(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->sm_name();
 OUTPUT:
  RETVAL

SV *
ins_dual(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->ins_dual();
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
ins_false(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->check_false();
 OUTPUT:
  RETVAL

SV *
ins_target(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->ins_target();
 OUTPUT:
  RETVAL

SV *
ins_brt(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->ins_brt();
 OUTPUT:
  RETVAL

SV *
ins_scbd(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->ins_scbd();
 OUTPUT:
  RETVAL

SV *
ins_scbd_type(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->ins_scbd_type();
 OUTPUT:
  RETVAL

SV *
ins_cc(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->ins_cc();
 OUTPUT:
  RETVAL

SV *
ins_sidl(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->ins_sidl();
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
mask(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
   std::string mask;
 CODE:
   bool res = e->gen_mask(mask);
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

SV *
ins_prop(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->ins_prop();
 OUTPUT:
  RETVAL

SV *
efield(SV *obj, const char *fname)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->extract_efield(fname);
 OUTPUT:
  RETVAL

SV *
efields(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->extract_efields();
 OUTPUT:
  RETVAL

SV *vfield(SV *obj, const char *fname)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->extract_vfield(fname);
 OUTPUT:
  RETVAL

SV *vfields(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->extract_vfields();
 OUTPUT:
  RETVAL

SV *tab_count(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->tab_count();
 OUTPUT:
  RETVAL

void
tab(SV *obj, IV key)
 PREINIT:
  U8 gimme = GIMME_V;
 INIT:
  bool res;
  SV *names = nullptr, *dict = nullptr;
  Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 PPCODE:
  res = e->get_tab(key, &names, &dict);
  if ( !res ) {
    ST(0) = &PL_sv_undef;
    XSRETURN(1);
  } else {
    if ( gimme == G_ARRAY) {
      EXTEND(SP, 2);
      mXPUSHs(names);
      mXPUSHs(dict);
      XSRETURN(2);
    } else {
      AV *av = newAV();
      av_push(av, names);
      av_push(av, dict);
      mXPUSHs(newRV_noinc((SV*)av));
      XSRETURN(1);
    }
  }

void
stat(SV *obj)
 PREINIT:
  U8 gimme = GIMME_V;
 INIT:
  Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 PPCODE:
  if ( gimme == G_ARRAY) {
      EXTEND(SP, 3);
      mXPUSHi(e->get_flush());
      mXPUSHi(e->get_rdr());
      mXPUSHi(e->is_dirty());
      XSRETURN(3);
  } else {
      AV *av = newAV();
      av_push(av, newSViv(e->get_flush()));
      av_push(av, newSViv(e->get_rdr()));
      av_push(av, e->is_dirty() ? &PL_sv_yes : &PL_sv_no);
      mXPUSHs(newRV_noinc((SV*)av));
      XSRETURN(1);
  }

SV *
get_enum(SV *obj, const char *ename)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->make_enum(ename);
 OUTPUT:
  RETVAL

SV *
ins_cb(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->extract_cb();
 OUTPUT:
  RETVAL

SV *
has_lut(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->check_lut();
 OUTPUT:
  RETVAL

SV *
lut(SV *obj, int idx)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->lut_name(idx);
 OUTPUT:
  RETVAL

SV *
kv(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   if ( !e->has_ins() )
     RETVAL = &PL_sv_undef;
   else
     RETVAL = newRV_noinc((SV*)e->make_kv());
 OUTPUT:
  RETVAL

SV *
nop(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->nop();
 OUTPUT:
  RETVAL

SV *
replace(SV *obj, const char *s)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   if ( !e->has_ins() )
     RETVAL = &PL_sv_undef;
   else
     RETVAL = e->replace(s) ? &PL_sv_yes : &PL_sv_no;
 OUTPUT:
  RETVAL

SV *
patch_pred(SV *obj, int is_not, int pred)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   if ( !e->has_ins() )
     RETVAL = &PL_sv_undef;
   else
     RETVAL = e->patch_pred(is_not, pred) ? &PL_sv_yes : &PL_sv_no;
 OUTPUT:
  RETVAL

SV *
patch_cb(SV *obj, unsigned long l1, unsigned long l2)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   if ( !e->has_ins() )
     RETVAL = &PL_sv_undef;
   else
     RETVAL = e->patch_cb(l1, l2) ? &PL_sv_yes : &PL_sv_no;
 OUTPUT:
  RETVAL

SV *
patch_tab(SV *obj, int idx, int v)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   if ( !e->has_ins() )
     RETVAL = &PL_sv_undef;
   else
     RETVAL = e->patch_tab(idx, v) ? &PL_sv_yes : &PL_sv_no;
 OUTPUT:
  RETVAL

SV *
patch(SV *obj, const char *fname, SV *v)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   if ( !e->has_ins() )
     RETVAL = &PL_sv_undef;
   else
     RETVAL = e->patch_field(fname, v) ? &PL_sv_yes : &PL_sv_no;
 OUTPUT:
  RETVAL

SV *
ptabs(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   if ( !e->has_ins() )
     RETVAL = &PL_sv_undef;
   else
     RETVAL = e->has_pending_tabs() ? &PL_sv_yes : &PL_sv_no;
 OUTPUT:
  RETVAL

SV *
render(SV *obj)
 INIT:
  AV *fake;
  SV *objref= NULL;
  MAGIC* magic;
  RItems *res;
  Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
PPCODE:
  res = new RItems;
  if ( e->make_render(*res) ) {
    fake = newAV();
    objref = newRV_noinc((SV*)fake);
    sv_bless(objref, s_ca_render_pkg);
    magic = sv_magicext((SV*)fake, NULL, PERL_MAGIC_tied, &ca_rend_magic_vt, (const char *)res, 0);
    SvREADONLY_on((SV*)fake);
    ST(0) = objref;
  } else {
    delete res;
    ST(0) = &PL_sv_undef;
  }
  XSRETURN(1);

MODULE = Cubin::Ced		PACKAGE = Cubin::Ced::Render

void
FETCH(self, key)
  SV *self;
  IV key;
 INIT:
  AV *res;
  auto *d = magic_tied<RItems>(self, 1, &ca_rend_magic_vt);
 PPCODE:
  if ( key >= d->size() || key < 0 ) {
    ST(0) = &PL_sv_undef;
  } else {
    res = newAV();
    // fill output array res
    // [0] - type
    // [1] - pfx
    // [2] - sfx
    // [3] - mod
    // [4] - abs
    auto &ri = d->at(key);
    av_push(res, newSViv(ri.first->type));
    if ( ri.first->pfx )
      av_push(res, newSVpv(&ri.first->pfx, 1));
    else
      av_push(res, &PL_sv_undef);
    if ( ri.first->sfx )
      av_push(res, newSVpv(&ri.first->sfx, 1));
    else
      av_push(res, &PL_sv_undef);
    if ( ri.first->mod )
      av_push(res, newSVpv(&ri.first->mod, 1));
    else
      av_push(res, &PL_sv_undef);
    av_push(res, newSViv(ri.first->abs));
    // [5] - tail of enums
    av_push(res, make_etail(ri.second));
    ST(0) = newRV_noinc((SV*)res);
    // rest is
    // [6] - name
    // [7] - left
    // [8] - right
    switch(ri.first->type) {
      case R_value:
      case R_predicate:
      case R_enum: {
        const render_named *rn = (const render_named *)ri.first;
        av_push(res, newSVpv(rn->name, strlen(rn->name)));
       }
       break;
      case R_C:
      case R_CX: {
         const render_C *rn = (const render_C *)ri.first;
         if ( rn->name )
           av_push(res, newSVpv(rn->name, strlen(rn->name)));
         else
           av_push(res, &PL_sv_undef);
         av_push(res, make_vbase(&rn->left));
         if ( !rn->right.empty() ) av_push(res, make_vblist(rn->right));
       }
       break;
      case R_TTU: {
         const render_TTU *rt = (const render_TTU *)ri.first;
         // no name
         av_push(res, &PL_sv_undef);
         av_push(res, make_vbase(&rt->left));
       }
       break;
      case R_M1: {
         const render_M1 *m1 = (const render_M1 *)ri.first;
         if ( m1->name )
           av_push(res, newSVpv(m1->name, strlen(m1->name)));
         else
           av_push(res, &PL_sv_undef);
         av_push(res, make_vbase(&m1->left));
       }
       break;
      case R_desc: {
         const render_desc *rd = (const render_desc *)ri.first;
         // no name
         av_push(res, &PL_sv_undef);
         av_push(res, make_vbase(&rd->left));
         if ( !rd->right.empty() ) av_push(res, make_vblist(rd->right));
       }
       break;
      case R_mem: {
         const render_mem *rm = (const render_mem *)ri.first;
         if ( rm->name )
           av_push(res, newSVpv(rm->name, strlen(rm->name)));
         else
           av_push(res, &PL_sv_undef);
         // no left
         av_push(res, &PL_sv_undef);
         av_push(res, make_vblist(rm->right));
       }
       break;
    }
    ST(0) = newRV_noinc((SV*)res);
  }
  XSRETURN(1);

BOOT:
 s_ca_pkg = gv_stashpv(s_ca, 0);
 if ( !s_ca_pkg )
    croak("Package %s does not exists", s_ca);
 s_ca_render_pkg = gv_stashpv(s_ca_render, 0);
 if ( !s_ca_render_pkg )
    croak("Package %s does not exists", s_ca_render);
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
 EXPORT_ENUM(NV_Scbd, SOURCE_RD)
 EXPORT_ENUM(NV_Scbd, SOURCE_WR)
 EXPORT_ENUM(NV_Scbd, SINK)
 EXPORT_ENUM(NV_Scbd, SOURCE_SINK_RD)
 EXPORT_ENUM(NV_Scbd, SOURCE_SINK_WR)
 EXPORT_ENUM(NV_Scbd, NON_BARRIER_INT_INST)
 EXPORT_ENUM(NV_Scbd_Type, BARRIER_INST)
 EXPORT_ENUM(NV_Scbd_Type, MEM_INST)
 EXPORT_ENUM(NV_Scbd_Type, BB_ENDING_INST)
 // render types
 EXPORT_ENUM(NV_rend, R_value)
 EXPORT_ENUM(NV_rend, R_enum)
 EXPORT_ENUM(NV_rend, R_predicate)
 EXPORT_ENUM(NV_rend, R_opcode)
 EXPORT_ENUM(NV_rend, R_C)
 EXPORT_ENUM(NV_rend, R_CX)
 EXPORT_ENUM(NV_rend, R_TTU)
 EXPORT_ENUM(NV_rend, R_M1)
 EXPORT_ENUM(NV_rend, R_desc)
 EXPORT_ENUM(NV_rend, R_mem)
