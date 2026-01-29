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
#include "bf16.h"

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

static SV *get_ritem(const RItem &ri)
{
    AV *res = newAV();
    // fill output array res
    // [0] - type
    // [1] - pfx
    // [2] - sfx
    // [3] - mod
    // [4] - abs
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
    } // end of switch
    return newRV_noinc((SV*)res);
}

static SV *
form_float_conv(const NV_conv *fc) {
  if ( !fc ) return &PL_sv_undef;
  HV *hv = newHV();
  for ( auto &oc: *fc ) {
    // form array
    AV *curr = newAV();
    // 0 - fmt_var
    av_push(curr, newSVpv( oc.fmt_var.data(), oc.fmt_var.size() ));
    // 1 - f_t
    av_push(curr, newSVuv(oc.f_t));
    // 2 - f_f
    av_push(curr, newSVuv(oc.f_f));
    // 3 - v1
    av_push(curr, newSVuv(oc.v1));
    // 4 - v2
    if ( oc.v2 != -1 ) av_push(curr, newSVuv(oc.v2));
    // insert to hash, key oc.name
    hv_store(hv, oc.name.data(), oc.name.size(), newRV_noinc((SV*)curr), 0);
  }
  return newRV_noinc((SV*)hv);
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
 private:
   int ref_cnt = 1;
 public:
  reg_reuse reus;
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
  void add_ref() { ref_cnt++; }
  void release() { if ( !--ref_cnt ) delete this; }
  virtual int check_rel(unsigned long off) override {
     m_cur_rsym = nullptr;
     m_cur_rel = nullptr;
     has_relocs = false;
     if ( !m_cur_srels ) return 0;
     auto si = m_cur_srels->find(off);
     if ( si == m_cur_srels->end() ) return 0;
     has_relocs = true;
     m_next_roff = off;
     // ups, this offset contains reloc - make warning
     // fprintf(stderr, "Warning: offset %lX has reloc %d\n", off, si->second.first);
     m_cur_rel = &si->second;
     m_cur_rsym = &m_syms[si->second.second];
     return 1;
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
  int block_mask() const {
    return m_block_mask;
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
  int get_rz() const {
    return m_dis->rz;
  }
  int sm_num() const {
    return m_sm;
  }
  const char *sm_name() const {
    return m_sm_name;
  }
  const NV_field *field_at(IV off) const {
    if ( !has_ins() ) return nullptr;
    return ins()->field_atoff((short)off);
  }
  SV *extract_instrs() const;
  SV *extract_instrs(REGEXP *) const;
  bool extract_insn(const char *, std::vector<SV *> &);
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
  int try_swap(UV off);
  int set_off(UV off) {
    if ( m_state < WantOff ) {
      my_warn("m_state %d for off %lX", m_state, off);
      return 0;
    }
    if ( !flush_buf() ) return 0;
    int res = _verify_off(off);
    if ( !res ) {
      my_warn("verify_off %lX failed", off);
      reset_ins();
    }
    else reus.apply(ins(), cex());
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
    else reus.apply(ins(), cex());
    return res;
  }
  SV *get_off() {
    if ( !ins() ) return &PL_sv_undef;
    return newSVuv(m_dis->offset());
  }
  SV *next_off() {
    if ( !ins() ) return &PL_sv_undef;
    return newSVuv(m_dis->off_next());
  }
  SV *block_off(UV off) {
    if ( !m_block_mask ) return newSVuv(off);
    return newSVuv(off & ~m_block_mask);
  }
  SV *prev_off(UV off) {
    if ( !m_block_mask ) {
      if ( off < 16 ) return &PL_sv_undef;
      return newSVuv(off - 16);
    }
    // ok, we have SM with blocks - check if instruction at off is not first
    auto masked = off & m_block_mask;
    if ( masked >= 16 ) return newSVuv(off - 8);
    // get address of this block
    auto bstart = off & ~m_block_mask;
    if ( bstart < m_block_mask ) return &PL_sv_undef;
    return newSVuv(bstart - 8);
  }
  SV *ins_conv() {
    if ( !ins() ) return &PL_sv_undef;
    return form_float_conv( ins()->vf_conv );
  }
  SV *get_ctrl() {
    if ( !ins() ) return &PL_sv_undef;
    if ( m_width != 64 ) return &PL_sv_no;
    uint8_t c = 0, o = 0;
    m_dis->get_ctrl(c, o);
    return newSVuv(c);
  }
  SV *get_opcode() {
    if ( !ins() ) return &PL_sv_undef;
    if ( m_width != 64 ) return &PL_sv_no;
    uint8_t c = 0, o = 0;
    m_dis->get_ctrl(c, o);
    return newSVuv(o);
  }
  SV *get_cword() {
    if ( !ins() ) return &PL_sv_undef;
    if ( m_width == 64 ) return &PL_sv_no;
    uint64_t cword = m_dis->get_cword();
    return newSVuv(cword);
  }
  SV *rend_cword(UV v) const {
    char buf[128];
    render_cword(v, buf, sizeof(buf) - 1);
    buf[127] = 0;
    return newSVpv(buf, strlen(buf));
  }
  SV *has_comp(const NV_rlist *rl) const {
    if ( !rl ) return &PL_sv_undef;
    for ( auto r: *rl )
      if ( is_compound(r->type) ) return newSViv(r->type);
    return &PL_sv_no;
  }
  SV *has_comp() const {
    if ( !ins() ) return &PL_sv_undef;
    return has_comp(m_rend);
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
  template <auto nv_instr::*fptr>
  SV *ins_intxxx() const {
    if ( !ins() ) return &PL_sv_undef;
    return newSViv(ins()->*fptr);
  }
  template <auto nv_instr::*fptr>
  SV *ins_pvxxx() const {
    if ( !ins() ) return &PL_sv_undef;
    auto v = ins()->*fptr;
    if ( !v ) return &PL_sv_undef;
    return newSVpv(v, strlen(v));
  }
  SV *ins_line() const {
    return ins_intxxx<&nv_instr::line>();
  }
  SV *ins_alt() const {
    return ins_intxxx<&nv_instr::alt>();
  }
  SV *ins_setp() const {
    return ins_intxxx<&nv_instr::setp>();
  }
  SV *ins_brt() const {
    return ins_intxxx<&nv_instr::brt>();
  }
  SV *ins_scbd() const {
    return ins_intxxx<&nv_instr::scbd>();
  }
  SV *ins_scbd_type() const {
    return ins_intxxx<&nv_instr::scbd_type>();
  }
  SV *ins_itype() const {
    return ins_intxxx<&nv_instr::itype>();
  }
  SV *ins_min_wait() const {
    return ins_intxxx<&nv_instr::min_wait>();
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
    return ins_pvxxx<&nv_instr::target_index>();
  }
  SV *ins_cc() const {
    return ins_pvxxx<&nv_instr::cc_index>();
  }
  SV *ins_dual() const {
    if ( !has_ins() ) return &PL_sv_undef;
    return check_dual(cex()) ? &PL_sv_yes : &PL_sv_no;
  }
  SV *check_false() const {
    if ( !has_ins() ) return &PL_sv_undef;
    return always_false(ins(), m_rend, cex()) ? &PL_sv_yes : &PL_sv_no;
  }
  SV *has_pred() const {
    if ( !has_ins() ) return &PL_sv_undef;
    return has_predicate(m_rend, cex()) ? &PL_sv_yes : &PL_sv_no;
  }
  SV *check_pred(const NV_rlist *rl) const {
    auto pred_name = has_predicate(rl);
    if ( !pred_name ) return &PL_sv_no;
    return newSVpv(pred_name, strlen(pred_name));
  }
  SV *check_pred() const {
    if ( !has_ins() ) return &PL_sv_undef;
    return check_pred(m_rend);
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
    return ins_pvxxx<&nv_instr::sidl_name>();
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
  bool is_branch(long &off) {
    return 0 != NV_renderer::collect_labels(m_rend, ins(), cex(), nullptr, &off);
  }
  std::optional<long> ins_cb(unsigned short *cb_idx, bool is_pure) {
    if ( !has_ins() ) return std::nullopt;
    return is_pure ? check_cbank_pure(m_rend, cex(), cb_idx):
                     check_cbank(m_rend, cex(), cb_idx);
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
  bool grep_kv(REGEXP *, std::vector<std::string_view> &) const;
  SV *special_kv(NV_extracted::const_iterator &);
  SV *get_kv(const std::string_view &fname) {
    if ( !has_ins() ) return &PL_sv_undef;
    auto &kv = cex();
    NV_extracted::const_iterator ei = kv.find(fname);
    if ( ei == kv.cend() ) return &PL_sv_undef;
    auto *sv = special_kv(ei);
    if ( sv ) return sv;
    return newSVuv(ei->second);
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
  SV *ins_pred(const char *key) {
    if ( !has_ins() ) return &PL_sv_undef;
    const NV_Preds *preds = ins()->predicated;
    if ( !preds ) return &PL_sv_undef;
    auto pri = preds->find(key);
    if ( pri != preds->end() ) {
      int res = pri->second(cex());
      if ( m_vq && cmp(pri->first, "VQ") ) {
        auto name = m_vq(res);
        if ( name ) return newSVpv(name, strlen(name));
      }
      return newSViv(res);
    }
    // not found
    return &PL_sv_undef;
  }
  bool get_lxx(std::vector<SV *> &, int is_col) const;
  // return ref to hash where key is enum NVP_ops and value is ref to array where
  // [0] - type - enum NVP_type
  // [1..n] - field names
  SV *ins_prop(const nv_instr *inst) const {
    if ( !inst->props ) return &PL_sv_undef;
    HV *hv = newHV();
    for ( size_t i = 0; i < inst->props->size(); ++i ) {
      auto prop = get_it(*inst->props, i);
      hv_store_ent(hv, newSViv(prop->op), make_prop(prop), 0);
    }
    return newRV_noinc((SV*)hv);
  }
  SV *ins_prop() const {
    if ( !has_ins() ) return &PL_sv_undef;
    return ins_prop(ins());
  }
  // almost as ins_prop
  SV *grep_prop(IV key, const nv_instr *inst) const {
    if ( !inst->props ) return &PL_sv_undef;
    for ( size_t i = 0; i < inst->props->size(); ++i ) {
      auto prop = get_it(*inst->props, i);
      if ( prop->op != key ) continue;
      return make_prop(prop);
    }
    // not found
    return &PL_sv_undef;
  }
  SV *grep_prop(IV key) const {
    if ( !has_ins() ) return &PL_sv_undef;
    return grep_prop(key, ins());
  }
  bool collect_labels(long *);
  bool make_render(RItems &);
  bool make_render(RItems &, const NV_rlist *);
  SV *extract_cb() const;
  SV *extract_cb(const nv_instr *) const;
  SV *extract_efield(const char *) const;
  SV *extract_efield(const nv_instr *, const char *) const;
  SV *extract_efields() const;
  SV *extract_efields(const nv_instr *) const;
  SV *extract_vfield(const char *) const;
  SV *extract_vfield(const nv_instr *, const char *) const;
  SV *extract_vfields() const;
  SV *extract_vfields(const nv_instr *) const;
  SV *make_enum(const char *);
  // tabs
  SV *check_tab(const char *fname, int do_filter) const;
  SV *tab_count(const nv_instr *inst) const {
    return newSViv(inst->tab_fields.size());
  }
  SV *tab_count() const {
    if ( !has_ins() ) return &PL_sv_undef;
    return newSViv(ins()->tab_fields.size());
  }
  SV *has_tfield(const nv_instr *inst, std::string_view &) const;
  SV *has_tfield(std::string_view &sv) const {
    if ( !has_ins() ) return &PL_sv_undef;
    return has_tfield(ins(), sv);
  }
  bool get_tab(IV, SV **n, SV **d) const;
  bool get_tab(const nv_instr *,IV, SV **n, SV **d) const;
  inline int apply(reg_pad *rt) {
    return has_ins() ? track_regs(rt, m_rend, curr_dis, m_dis->offset()) : 0;
  }
  SV *tab_map(const nv_instr *inst, IV) const;
  SV *tab_map(IV key) const {
    if ( !has_ins() ) return &PL_sv_undef;
    return tab_map(ins(), key);
  }
  void tab_fields(const nv_instr *inst, IV, std::vector<std::string_view> &) const;
  void tab_fields(IV key, std::vector<std::string_view> &res) const {
    if ( !has_ins() ) return;
    tab_fields(ins(), key, res);
  }
 protected:
  template <typename T>
  int patch_int(const nv_vattr *va, T value) {
    int fmt = va ? va->kind : (std::is_signed_v<T> ? NV_SImm : NV_UImm);
    if ( fmt < NV_F64Imm ) {
      m_v = value;
      return 1;
    }
    if ( !va ) return 0;
    check_fconv(ins(), cex(), *va, fmt);
    if ( fmt == NV_F64Imm ) {
      *(double *)m_v = double(value);
      return 1;
    }
    if ( fmt == NV_F32Imm ) {
      float fl = (float)value;
      *(float *)&m_v = fl;
      return 1;
    }
    if ( fmt == NV_E8M7Imm ) {
      m_v = e8m7_f(float(value));
      return 1;
    }
    if ( fmt == NV_F16Imm ) {
      *(float *)&m_v = fp16_ieee_from_fp32_value(float(value));
      return 1;
    }
    return 0;
  }
  SV *make_prop(const NV_Prop *prop) const;
  SV *make_enum_arr(const nv_eattr *ea) const;
  HV *make_enum(const std::unordered_map<int, const char *> *);
  SV *make_vfield(const nv_instr *, const nv_vattr &) const;
  SV *fill_tab(const NV_tab_fields *t) const;
  SV *fill_simple_tab(const std::unordered_map<int, const unsigned short *> *) const;
  SV *fill_tab(const std::unordered_map<int, const unsigned short *> *, size_t) const;
  IElf *m_e;
  // cached enums (HV *), key is nv_eattr->ename
  std::unordered_map<std::string_view, HV *> m_cached_hvs;
};

int Ced_perl::try_swap(UV off) {
  if ( !has_ins() ) return 0;
  return swap_with(off);
}

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

bool Ced_perl::make_render(RItems &res, const NV_rlist *rend) {
  for ( auto r: *rend ) {
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

bool Ced_perl::make_render(RItems &res) {
  if ( !has_ins() || !m_rend ) return false;
  return make_render(res, m_rend);
}

// return array of insns mnemonic names
SV *Ced_perl::extract_instrs() const {
  if ( !m_sorted ) return &PL_sv_undef;
  AV *av = newAV();
  for ( auto it = m_sorted->begin(); it != m_sorted->end(); ++it )
    av_push(av, newSVpv( it->first.data(), it->first.size() ));
  return newRV_noinc((SV*)av);
}

SV *Ced_perl::extract_instrs(REGEXP *rx) const {
  if ( !m_sorted ) return &PL_sv_undef;
  AV *av = newAV();
  for ( auto it = m_sorted->begin(); it != m_sorted->end(); ++it )
  {
    auto s = it->first.data();
    SV *scream = newSVpv(s, it->first.size());
    STRLEN retlen;
    char *input = SvPVutf8(scream, retlen);
    I32 nmatch = pregexec(rx, input, input + retlen, input, 0, scream, 0);
    SvREFCNT_dec(scream);
    if ( nmatch > 0 ) av_push(av, newSVpv( it->first.data(), it->first.size() ));
  }
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
  if ( ctr && m_width != 64 ) {
    Err("Ctrl not supported for 88/128 bits\n");
    return 0;
  }
  const NV_cbank *cb = is_cb_field(ins(), p, cb_idx);
  if ( !ctr && !cb ) {
    tab = is_tab_field(ins(), fname, tab_idx);
    if ( !tab ) {
      field = std::lower_bound(ins()->fields.begin(), ins()->fields.end(), p,
       [](const NV_field &f, const std::string &w) {
         return f.name < w;
      });
      if ( field == ins()->fields.end() ) {
        Err("unknown field %s, line %d, offset %lX\n", fname, ins()->line, m_dis->offset());
        return 0;
      }
      // cool, some real field
      ea = find_ea(ins(), p);
      if ( !ea && ins()->vas )
        va = find(ins()->vas, p);
    }
  }
  m_v = 0;
  bool has_v = false;
  // check what we have and what kind of SV
  if ( va || ctr ) { // some imm value
    if ( va && SvPOK(v) ) { // string
      STRLEN len;
      auto pv = SvPVbyte(v, len);
      std::string_view sv{ pv, len };
      if ( !parse_num(va, sv) ) {
        Err("cannot parse num %.*s, offset %lX\n", len, sv.data(), m_dis->offset());
        return 0;
      }
    } else if ( SvUOK(v) ) {
     if ( !patch_int(va, SvUV(v)) ) {
       Err("Cannot patch field %s from unsigned int value", fname);
       return 0;
     }
    } else if ( SvIOK(v) ) {
     if ( !patch_int(va, SvIV(v)) ) {
       Err("Cannot patch field %s from int value", fname);
       return 0;
     }
    } else {
      int skip = 1;
      if ( SvNOK(v) ) {
        // we have some floating point value - we able to assign it to ctrl or NV_SImm only
        double d = SvNV(v);
        if ( !va ) {
          Err("round float value for %s", fname);
          m_v = round(d);
          skip = 0;
        } else { // this is value field
          int fmt = va->kind;
          if ( fmt >= NV_F64Imm )
            check_fconv(ins(), cex(), *va, fmt);
          if ( (fmt == NV_F64Imm || fmt == NV_F32Imm || fmt == NV_F16Imm || fmt == NV_E8M7Imm) )
          {
            if ( fmt == NV_F64Imm ) m_v = *(uint64_t *)&d;
            else if ( fmt == NV_F32Imm ) {
              float fl = (float)d;
              *(float *)&m_v = fl;
            } else if ( fmt == NV_E8M7Imm ) {
              m_v = e8m7_f(float(d));
            } else if ( fmt == NV_F16Imm ) {
              *(float *)&m_v = fp16_ieee_from_fp32_value(float(d));
            }
            skip = 0;
          } else if ( fmt == NV_SImm ) {
            Err("round float value for SImm %s", fname);
            m_v = round(d);
            skip = 0;
          }
        }
      }
      if ( skip ) {
        Err("Unknown SV type %d in patch %s, offset %lX", SvTYPE(v), fname, m_dis->offset());
        return 0;
      }
    }
  } else if ( ea ) {
    if ( SvPOK(v) ) { // string
      STRLEN len;
      auto pv = SvPVbyte(v, len);
      std::string_view sv{ pv, len };
      if ( !m_renums ) {
        Err("no renums for field %s, enum %s, offset %lX\n", fname, ea->ename, m_dis->offset());
        return 0;
      }
      auto ed = m_renums->find(ea->ename);
      if ( ed == m_renums->end() ) {
        Err("cannot find renum %s for field %s, offset %lX\n", ea->ename, fname, m_dis->offset());
        return 0;
      }
      auto edi = ed->second->find(sv);
      if ( edi == ed->second->end() ) {
        Err("cannot find %.*s in enum %s for field %s, offset %lX\n", len, sv.data(), ea->ename, fname, m_dis->offset());
        return 0;
      }
      m_v = edi->second;
      has_v = true;
    } else if ( SvIOK(v) ) {
      m_v = SvIV(v);
      auto ei = ea->em->find(m_v);
      if ( ei == ea->em->end() ) {
        Err("value %lX for field %s not in enum %s, offset %lX\n", m_v, fname, ea->ename, m_dis->offset());
        return 0;
      }
      has_v = true;
    } else {
      Err("Unknown SV type %d for enum %s in patch, offset %lX", SvTYPE(v), ea->ename, m_dis->offset());
      return 0;
    }
  }
  // check how this field should be patched
  if ( ctr ) {
    int res = m_dis->put_ctrl(m_v);
    if ( res ) block_dirty = true;
    return res;
  }
  if ( field ) {
    int res = patch(field, field->scale ? m_v / field->scale : m_v, fname);
    ex()[field->name] = m_v;
    return res;
  }
  if ( !has_v && SvIOK(v) )
    m_v = SvIV(v);
  else {
    Err("Unknown SV type %d for %s in patch(%s), offset %lX", SvTYPE(v), tab ? "tab" : "cb", fname, m_dis->offset());
    return 0;
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
        Err("Warning: value %ld for %s invalid in table, offset %lX\n", m_v, fname, m_dis->offset());
      }
      if ( opt_d )
        dump_tab_fields(tab);
      return 1;
    } else
     return patch(tab, tab_value, fname);
  }
  Err("dont know how to patch %s, offset %lX\n", fname, m_dis->offset());
  return 0;
}

SV *Ced_perl::check_tab(const char *fname, int do_filter) const
{
  if ( !has_ins() ) return &PL_sv_undef;
  std::unordered_set<unsigned short> res;
  if ( !filter_tab_rows(fname, do_filter, &res) ) return &PL_sv_undef;
  HV *hv = newHV();
  for ( auto ei: res ) {
    hv_store_ent(hv, newSViv(ei), &PL_sv_yes, 0);
  }
  return newRV_noinc((SV*)hv);
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
    Err("invalid value %d for tab %d, offset %lX", v, t_idx, m_dis->offset());
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

// check if some table contains field with name tfname
// returns index of table or undef
SV *Ced_perl::has_tfield(const nv_instr *inst, std::string_view &tfname) const
{
  for ( size_t idx = 0; idx < inst->tab_fields.size(); ++idx ) {
    auto t = get_it(inst->tab_fields, idx);
    for ( size_t ni = 0; ni < t->fields.size(); ++ni ) {
      auto &nf = get_it(t->fields, ni);
      if ( !nf.compare(tfname) ) return newSViv(idx);
    }
  }
  return &PL_sv_undef;
}

SV *Ced_perl::fill_tab(const NV_tab_fields *t) const {
  // dict can be just int key -> int value
  // or int key -> ref to array when fields > 1
  if ( 1 == t->fields.size() )
    return fill_simple_tab(t->tab);
  else
    return fill_tab(t->tab, t->fields.size());
}

SV *Ced_perl::tab_map(const nv_instr *inst, IV idx) const {
  if ( idx < 0 || idx >= inst->tab_fields.size() ) return &PL_sv_undef;
  auto t = get_it(inst->tab_fields, idx);
  return fill_tab(t);
}

// extract tab with index idx
// put ref to array with names into n
// put ref to hash into d
bool Ced_perl::get_tab(const nv_instr *inst, IV idx, SV **n, SV **d) const {
  if ( idx < 0 || idx >= inst->tab_fields.size() ) return false;
  auto t = get_it(inst->tab_fields, idx);
  // fill names
  AV *av = newAV();
  for ( size_t ni = 0; ni < t->fields.size(); ++ni ) {
    auto &nf = get_it(t->fields, ni);
    av_push(av, newSVpv( nf.data(), nf.size() ));
  }
  *n = newRV_noinc((SV*)av);
  if ( d ) *d = fill_tab(t);
  return true;
}

void Ced_perl::tab_fields(const nv_instr *inst, IV idx, std::vector<std::string_view> &res) const {
  if ( idx < 0 || idx >= inst->tab_fields.size() ) return;
  auto t = get_it(inst->tab_fields, idx);
  for ( size_t ni = 0; ni < t->fields.size(); ++ni ) {
    res.push_back(get_it(t->fields, ni));
  }
}

bool Ced_perl::get_tab(IV idx, SV **n, SV **d) const {
  if ( !has_ins() ) return false;
  return get_tab(ins(), idx, n, d);
}

SV *Ced_perl::fill_simple_tab(const std::unordered_map<int, const unsigned short *> *t) const
{
  HV *hv = newHV();
  for ( auto ei: *t ) {
    hv_store_ent(hv, newSViv(ei.first), newSVuv(ei.second[1]), 0);
  }
  return newRV_noinc((SV*)hv);
}

SV *Ced_perl::fill_tab(const std::unordered_map<int, const unsigned short *> *t, size_t ts) const
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
SV *Ced_perl::make_prop(const NV_Prop *prop) const {
  AV *av = newAV();
  av_push(av, newSViv(prop->t));
  for ( size_t i = 0; i < prop->fields.size(); ++i ) {
    auto &field = get_it(prop->fields, i);
    av_push(av, newSVpv(field.data(), field.size()));
  }
  return newRV_noinc((SV*)av);
}

// return array with couple of fields names, 3rd is scale when non-zero
SV *Ced_perl::extract_cb(const nv_instr *inst) const
{
  if ( !inst->cb_field ) return &PL_sv_undef;
  AV *av = newAV();
  av_push(av, newSVpv(inst->cb_field->f1.data(), inst->cb_field->f1.size()) );
  av_push(av, newSVpv(inst->cb_field->f2.data(), inst->cb_field->f2.size()) );
  if ( inst->cb_field->scale )
    av_push(av, newSViv(inst->cb_field->scale));
  return newRV_noinc((SV*)av);
}

SV *Ced_perl::extract_cb() const {
  if ( !has_ins() ) return &PL_sv_undef;
  return extract_cb(ins());
}

// return hash with nv_eattr, key is field name, value is ref to array:
// 0 - ignore
// 1 - print
// 2 - has_def_value
// 3 - def value if presents
SV *Ced_perl::make_enum_arr(const nv_eattr *ea) const
{
  AV *av = newAV();
  av_push(av, ea->ignore ? &PL_sv_yes : &PL_sv_no);
  av_push(av, ea->print ? &PL_sv_yes : &PL_sv_no);
  av_push(av, ea->has_def_value ? &PL_sv_yes : &PL_sv_no);
  if ( ea->has_def_value )
    av_push(av, newSViv(ea->def_value));
  return newRV_noinc((SV*)av);
}

SV *Ced_perl::extract_efields(const nv_instr *inst) const
{
  if ( !inst->eas.size() ) return &PL_sv_undef;
  HV *hv = newHV();
  for ( size_t i = 0; i < inst->eas.size(); ++i ) {
    auto &ea = get_it(inst->eas, i);
    hv_store(hv, ea.name.data(), ea.name.size(), make_enum_arr(ea.ea), 0);
  }
  return newRV_noinc((SV*)hv);
}

SV *Ced_perl::extract_efields() const
{
  if ( !has_ins() ) return &PL_sv_undef;
  return extract_efields(ins());
}

// if we have width for some field - return [ type, width ]
// else just return type in SViv
SV *Ced_perl::make_vfield(const nv_instr *inst, const nv_vattr &v) const
{
  SV *t = newSViv(v.kind);
  if ( !inst->vwidth ) return t;
  auto vw = find(inst->vwidth, v.name);
  if ( !vw ) return t;
  AV *av = newAV();
  av_push(av, t);
  av_push(av, newSViv(vw->w));
  return newRV_noinc((SV*)av);
}

SV *Ced_perl::extract_vfields(const nv_instr *inst) const
{
  if ( !inst->vas ) return &PL_sv_undef;
  HV *hv = newHV();
  for ( size_t i = 0; i < inst->vas->size(); ++i ) {
    auto &va = get_it(*inst->vas, i);
    hv_store(hv, va.name.data(), va.name.size(), make_vfield(inst, va), 0);
  }
  return newRV_noinc((SV*)hv);
}

SV *Ced_perl::extract_vfields() const
{
  if ( !has_ins() ) return &PL_sv_undef;
  return extract_vfields(ins());
}

SV *Ced_perl::extract_vfield(const nv_instr *inst, const char *name) const
{
  if ( !inst->vas ) return &PL_sv_undef;
  std::string_view tmp{ name, strlen(name) };
  auto va = find(inst->vas, tmp);
  if ( !va ) return &PL_sv_undef;
  return make_vfield(inst, *va);
}

SV *Ced_perl::extract_vfield(const char *name) const
{
  if ( !has_ins() ) return &PL_sv_undef;
  return extract_vfield(ins(), name);
}

HV *Ced_perl::make_enum(const std::unordered_map<int, const char *> *em)
{
  HV *hv = newHV();
  for ( auto ei: *em ) {
    hv_store_ent(hv, newSViv(ei.first), newSVpv(ei.second, strlen(ei.second)), 0);
  }
  return hv;
}

SV *Ced_perl::extract_efield(const nv_instr *inst, const char *name) const
{
  std::string_view tmp{ name, strlen(name) };
  auto ea = find(inst->eas, tmp);
  if ( !ea ) return &PL_sv_undef;
  return make_enum_arr(ea->ea);
}

SV *Ced_perl::extract_efield(const char *name) const
{
  if ( !has_ins() ) return &PL_sv_undef;
  return extract_efield(ins(), name);
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

bool Ced_perl::grep_kv(REGEXP *rx, std::vector<std::string_view> &res) const
{
  if ( !has_ins() ) return false;
  for ( auto &kvi: cex() ) {
    auto s = kvi.first.data();
    SV *scream = newSVpv(s, kvi.first.size());
    STRLEN retlen;
    char *input = SvPVutf8(scream, retlen);
    I32 nmatch = pregexec(rx, input, input + retlen, input, 0, scream, 0);
    SvREFCNT_dec(scream);
    if (nmatch > 0 ) res.push_back(kvi.first);
  }
  return !res.empty();
}

SV *Ced_perl::special_kv(NV_extracted::const_iterator &ei)
{
  if ( !ins()->vas ) return nullptr;
  auto va = find(ins()->vas, ei->first);
  if ( !va ) return nullptr;
  int fmt = va->kind;
  if ( fmt >= NV_F64Imm )
    check_fconv(ins(), cex(), *va, fmt);
  // fill value with according format
  if ( fmt == NV_F64Imm ) {
    auto v = ei->second;
    return newSVnv(*(double *)&v);
  }
  if ( fmt == NV_F32Imm ) {
    auto v = ei->second;
    return newSVnv(*(float *)&v);
  }
  if ( fmt == NV_E8M7Imm ) {
    float f32 = e8m7_f((uint16_t)ei->second);
    return newSVnv(f32);
  }
  if ( fmt == NV_F16Imm ) {
    float f32 = fp16_ieee_to_fp32_bits((uint16_t)ei->second);
    return newSVnv(f32);
  }
  if ( va->kind == NV_SImm || va->kind == NV_SSImm || va->kind == NV_RSImm ) {
    // lets convert signed value to SViv
    long conv = 0;
    if ( check_branch(ins(), ei, conv) || conv_simm(ins(), ei, conv) ) return newSViv(conv);
  }
  return nullptr;
}

HV *Ced_perl::make_kv()
{
  HV *hv = newHV();
  for ( NV_extracted::const_iterator ei = cex().cbegin(); ei != cex().cend(); ++ei ) {
    auto *sv = special_kv(ei);
    if ( sv ) {
      hv_store(hv, ei->first.data(), ei->first.size(), sv, 0);
      continue;
    }
    hv_store(hv, ei->first.data(), ei->first.size(), newSVuv(ei->second), 0);
  }
  return hv;
}

bool Ced_perl::collect_labels(long *res) {
  if ( !has_ins() ) return false;
  return NV_renderer::collect_labels(m_rend, ins(), cex(), nullptr, res);
}

// return merged map of (u)preds from track_snap
static SV *merge_preds(const track_snap *snap) {
  bool e_pr = snap->empty_pr(), e_upr = snap->empty_upr();
  if ( e_pr && e_upr ) return &PL_sv_undef;
  HV *hv = newHV();
  if ( !e_pr ) {
   for ( int i = 0; i < snap->pr_size; i++ ) {
     if ( snap->pr[i] ) hv_store_ent(hv, newSViv(i), newSViv(snap->pr[i]), 0);
   }
  }
  if ( !e_upr ) {
   for ( int i = 0; i < snap->pr_size; i++ ) {
     if ( snap->upr[i] ) hv_store_ent(hv, newSViv(i | 0x8000), newSViv(snap->upr[i]), 0);
   }
  }
  return newRV_inc((SV *)hv);
}

static SV *gprs(const track_snap *snap) {
  if ( snap->gpr.empty() ) return &PL_sv_undef;
  HV *hv = newHV();
  for ( auto r: snap->gpr ) {
    hv_store_ent(hv, newSViv(r.first), newSViv(r.second), 0);
  }
  return newRV_inc((SV *)hv);
}

// from SM dis - don't have KV, only render
struct one_instr {
  const nv_instr *ins;
  const NV_rlist *rend;
  Ced_perl *base;
  ~one_instr() {
    if ( base ) base->release();
  }
  template <auto nv_instr::*fptr>
  SV *iv() const {
    return newSViv(ins->*fptr);
  }
  template <auto nv_instr::*fptr>
  SV *strv() const {
    auto v = ins->*fptr;
    if ( !v ) return &PL_sv_undef;
    return newSVpv(v, strlen(v));
  }
};

// latency table indexes
struct lat_idx {
  const NV_tabref *tr = nullptr;
  char is_col = 0; // 0 - row, 1 - column
};

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
        magic_release<Ced_perl>,
        0, /* copy */
        0 /* dup */
        TAB_TAIL
};

// magic table for Cubin::Ced::LatIndex
static const char *s_ca_latindex = "Cubin::Ced::LatIndex";
static HV *s_ca_latindex_pkg = nullptr;
static MGVTBL ca_latindex_magic_vt = {
        0, /* get */
        0, /* write */
        0, /* length */
        0, /* clear */
        magic_del<lat_idx>,
        0, /* copy */
        0 /* dup */
        TAB_TAIL
};

bool Ced_perl::get_lxx(std::vector<SV *> &res, int is_col) const {
  if ( !has_ins() ) return false;
  auto *what = is_col ? ins()->cols : ins()->rows;
  if ( !what ) return false;
  // filter out and fill res
  for ( auto &tr: *what ) {
    // check table
    if ( tr.filter ) {
      if ( !tr.filter(ins(), cex()) ) continue;
    }
    auto &tr_what = is_col ? tr.tab->cols : tr.tab->rows;
    // check cond list
    auto &cl = get_it(tr_what, tr.idx);
    if ( !check_sched_cond(ins(), cex(), cl) ) continue;
    // store tr into res
    lat_idx *li = new lat_idx;
    li->is_col = is_col;
    li->tr = &tr;
    SV *msv = newSViv(0);
    SV *objref= newRV_noinc(msv);
    sv_bless(objref, s_ca_latindex_pkg);
    // attach magic
    sv_magicext(msv, NULL, PERL_MAGIC_ext, &ca_latindex_magic_vt, (const char*)li, 0);
    res.push_back(objref);
  }
  return !res.empty();
}

// magic table for Cubin::Ced::Instr
static const char *s_ca_instr = "Cubin::Ced::Instr";
static HV *s_ca_instr_pkg = nullptr;
static MGVTBL ca_instr_magic_vt = {
        0, /* get */
        0, /* write */
        0, /* length */
        0, /* clear */
        magic_del<one_instr>,
        0, /* copy */
        0 /* dup */
        TAB_TAIL
};

bool Ced_perl::extract_insn(const char *name, std::vector<SV *> &sres) {
  if ( !m_sorted ) return false;
  std::string_view key{name};
  NV_sorted::const_iterator si = std::lower_bound(m_sorted->cbegin(), m_sorted->cend(), key,
    [](const auto &a, const std::string_view &b) { return a.first < b; });
  if ( si == m_sorted->cend() ) return false;
  if ( !is_sv(&si->first, name) ) return false;
  for ( auto i: si->second ) {
    auto *res = new one_instr;
    res->ins = i;
    res->rend = m_dis->get_rend(i->n);
    res->base = this; add_ref();
    SV *msv = newSViv(0);
    SV *objref = newRV_noinc(msv);
    sv_bless(objref, s_ca_instr_pkg);
    // attach magic
    sv_magicext(msv, NULL, PERL_MAGIC_ext, &ca_instr_magic_vt, (const char*)res, 0);
    SvREADONLY_on((SV*)msv);
    sres.push_back(objref);
  }
  return !sres.empty();
}

// magic table for Cubin::Ced::RegTrack
static const char *s_ca_regtrack = "Cubin::Ced::RegTrack";
static HV *s_ca_regtrack_pkg = nullptr;
static MGVTBL ca_regtrack_magic_vt = {
        0, /* get */
        0, /* write */
        0, /* length */
        0, /* clear */
        magic_del<reg_pad>,
        0, /* copy */
        0 /* dup */
        TAB_TAIL
};

template <typename T>
SV *fill_rhash(const std::unordered_map<int, std::vector<T> > &rs) {
  HV *hv = newHV();
  for ( auto ei: rs ) {
    hv_store_ent(hv, newSViv(ei.first), newSVuv(ei.second.size()), 0);
  }
  return newRV_inc((SV *)hv);
}

template <typename T>
SV *fill_reg(const std::vector<T> &vt, unsigned long from) {
  // return sorted array of array refs where
  // [0] - offset
  // [1] - mask
  // [2] - is write
  // [3] - has predicate (or undef)
  // [4] - for typed_reg_history - type if presents
  constexpr bool has_type = requires(T &t) { t.type; };
  auto start = vt.cbegin();
  if ( from ) start = std::lower_bound(start, vt.cend(), from, [](const T& rh, unsigned long v) -> bool { return rh.off < v; });
  if ( start == vt.end() ) return &PL_sv_undef;
  AV *av = newAV();
  for ( ; start != vt.cend(); ++start ) {
    AV *curr = newAV();
    av_push(curr, newSVuv(start->off));
    av_push(curr, newSViv(start->kind));
    av_push(curr, start->kind & 0x8000 ? &PL_sv_yes : &PL_sv_no);
    // 3) check predicate
    int pred = 0;
    if ( start->has_pred(pred) ) av_push(curr, newSViv(pred));
    else av_push(curr, &PL_sv_undef);
    // 4) type
    if constexpr ( has_type ) {
      if ( start->type != GENERIC ) av_push(curr, newSViv(start->type));
    }
    // finally add ref to av
    av_push(av, newRV_noinc((SV*)curr));
  }
  return newRV_noinc((SV*)av);
}


static SV *
make_one_cb(const cbank_history &cbh) {
  AV *curr = newAV();
  av_push(curr, newSVuv(cbh.off));
  av_push(curr, newSViv(cbh.cb_num));
  av_push(curr, newSVuv(cbh.cb_off));
  av_push(curr, newSViv(cbh.kind));
  return newRV_noinc((SV*)curr);
}

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

IV swap(SV *obj, UV off)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->try_swap(off);
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
next_off(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->next_off();
 OUTPUT:
  RETVAL

SV *
prev_off(SV *obj, UV off)
 ALIAS:
  Cubin::Ced::block_off = 1
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = (ix == 1) ? e->block_off(off) : e->prev_off(off);
 OUTPUT:
  RETVAL

SV *
start(SV *obj)
 ALIAS:
  Cubin::Ced::end = 1
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = (ix == 1) ? e->get_end() : e->get_start();
 OUTPUT:
  RETVAL

SV *
ctrl(SV *obj)
 ALIAS:
  Cubin::Ced::opcode = 1
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   if ( !e->has_ins() )
     RETVAL = &PL_sv_undef;
   else
     RETVAL = 1 == ix ? e->get_opcode() : e->get_ctrl();
 OUTPUT:
  RETVAL

SV *
cword(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   if ( !e->has_ins() )
     RETVAL = &PL_sv_undef;
   else
     RETVAL = e->get_cword();
 OUTPUT:
  RETVAL

SV *
render_cword(SV *obj, UV v)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
  RETVAL = e->rend_cword(v);
 OUTPUT:
  RETVAL

SV *
print_cword(SV *obj, unsigned long cword)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
   char buffer[128];
 CODE:
  e->render_cword(cword, buffer, 127);
  buffer[127] = 0;
  RETVAL = newSVpv(buffer, strlen(buffer));
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

UV
block_mask(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->block_mask();
 OUTPUT:
  RETVAL

SV *
reloc_name(SV *obj, IV rt)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   auto rt_name = get_cuda_reloc_name(rt);
   if ( !rt_name )
     RETVAL = &PL_sv_undef;
   else
     RETVAL = newSVpv(rt_name, strlen(rt_name));
 OUTPUT:
  RETVAL

SV *
instrs(SV *obj, SV *re = nullptr)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
   REGEXP *rx = nullptr;
 CODE:
   // check if we have re
   if ( re ) {
     if ( !SvROK(re) || SvTYPE(SvRV(re)) != SVt_REGEXP ) {
      if ( SvROK(re) )
       croak("instrs: arg must be regexp, ref type %d", SvTYPE(SvRV(re)));
      else
       croak("instrs: arg must be regexp, type %d", SvTYPE(re));
     }
     rx = (REGEXP *)SvRV(re);
   }
   RETVAL = (rx != nullptr) ? e->extract_instrs(rx) : e->extract_instrs();
 OUTPUT:
  RETVAL

SV *
by_name(SV *obj, const char *name)
 PREINIT:
  U8 gimme = GIMME_V;
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
   std::vector<SV *> res;
 PPCODE:
  if ( !e->extract_insn(name, res) ) {
    ST(0) = &PL_sv_undef;
    XSRETURN(1);
  } else {
    if ( gimme == G_ARRAY) {
      EXTEND(SP, res.size());
      for ( auto si: res )
       mPUSHs(si);
    } else {
      AV *av = newAV();
      for ( auto si: res )
       av_push(av, si);
      mXPUSHs(newRV_noinc((SV*)av));
      XSRETURN(1);
    }
  }

void
field_at(SV *obj, IV off)
 PREINIT:
  U8 gimme = GIMME_V;
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
   auto *f = e->field_at(off);
 PPCODE:
   if ( !f ) {
    ST(0) = &PL_sv_undef;
    XSRETURN(1);
   } else {
    SV *name = newSVpv(f->name.data(), f->name.size());
    if ( gimme == G_ARRAY) {
      EXTEND(SP, 2);
      mPUSHs(name);
      mPUSHi(f->mask[0].second); // size
    } else {
      ST(0) = name;
      XSRETURN(1);
    }
   }

int rz(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->get_rz();
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
 ALIAS:
  Cubin::Ced::ins_class = 1
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = (ix == 1) ? e->ins_class() : e->ins_name();
 OUTPUT:
  RETVAL

SV *
ins_clabs(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
  if ( !e->has_ins() )
   RETVAL = &PL_sv_undef;
  else {
   long res = 0;
   if ( e->collect_labels(&res) )
    RETVAL = newSViv(res);
   else
    RETVAL = &PL_sv_undef;
  }
 OUTPUT:
  RETVAL

SV *
ins_conv(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
  if ( !e->has_ins() )
   RETVAL = &PL_sv_undef;
  else
   RETVAL = e->ins_conv();
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
 ALIAS:
  Cubin::Ced::ins_scbd_type = 1
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = (ix == 1) ? e->ins_scbd_type() : e->ins_scbd();
 OUTPUT:
  RETVAL

SV *
ins_min_wait(SV *obj)
 ALIAS:
  Cubin::Ced::ins_itype = 1
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = (ix == 1) ? e->ins_itype() : e->ins_min_wait();
 OUTPUT:
  RETVAL

SV *
ins_cc(SV *obj)
 ALIAS:
  Cubin::Ced::ins_sidl = 1
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = (ix == 1) ? e->ins_sidl() : e->ins_cc();
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
 ALIAS:
  Cubin::Ced::mask = 1
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
   std::string mask;
 CODE:
   bool res = 1 == ix ? e->gen_mask(mask) : e->ins_mask(mask);
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
grep_pred(SV *obj, const char *key)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->ins_pred(key);
 OUTPUT:
  RETVAL

SV *
ins_reuse(SV *obj)
 ALIAS:
  Cubin::Ced::ins_reuse2 = 1
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
  if ( !e->has_ins() )
   RETVAL = &PL_sv_undef;
  else
   RETVAL = newSVuv((ix == 1) ? e->reus.mask2 : e->reus.mask);
 OUTPUT:
  RETVAL

SV *
ins_keep(SV *obj)
 ALIAS:
  Cubin::Ced::ins_keep2 = 1
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
  if ( !e->has_ins() )
   RETVAL = &PL_sv_undef;
  else
   RETVAL = newSVuv((ix == 1) ? e->reus.keep2 : e->reus.keep);
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
grep_prop(SV *obj, IV key)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->grep_prop(key);
 OUTPUT:
  RETVAL

SV *
efield(SV *obj, const char *fname)
 ALIAS:
  Cubin::Ced::vfield = 1
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = (ix == 1) ? e->extract_vfield(fname) : e->extract_efield(fname);
 OUTPUT:
  RETVAL

SV *
efields(SV *obj)
 ALIAS:
  Cubin::Ced::vfields = 1
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = (ix == 1) ? e->extract_vfields() : e->extract_efields();
 OUTPUT:
  RETVAL

SV *
has_tfield(SV *obj, const char *tfname)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
   std::string_view tmp{tfname};
 CODE:
   RETVAL = e->has_tfield(tmp);
 OUTPUT:
  RETVAL

SV *
check_tab(SV *obj, const char *tfname, int do_filter = 0)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->check_tab(tfname, do_filter);
 OUTPUT:
  RETVAL

void
ins_branch(SV *obj)
 PREINIT:
  U8 gimme = GIMME_V;
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 PPCODE:
  if ( !e->has_ins() ) {
    ST(0) = &PL_sv_undef;
    XSRETURN(1);
  } else {
    long off = 0;
    bool res = e->is_branch(off);
    if ( !res || gimme != G_ARRAY ) {
      ST(0) = res ? &PL_sv_yes : &PL_sv_no;
      XSRETURN(1);
    } else {
      EXTEND(SP, 2);
      mXPUSHs(res ? &PL_sv_yes : &PL_sv_no);
      mXPUSHi(off);
    }
  }

SV *tab_count(SV *obj)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->tab_count();
 OUTPUT:
  RETVAL

void
ins_cbank(SV *obj)
 ALIAS:
  Cubin::Ced::ins_cbank_pure = 1
 PREINIT:
  U8 gimme = GIMME_V;
 INIT:
  int res_size = 1;
  unsigned short cb_idx = 0xffff;
  Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 PPCODE:
  if ( !e->has_ins() ) {
    ST(0) = &PL_sv_undef;
    XSRETURN(1);
  } else {
    auto cb_off = e->ins_cb(&cb_idx, 1 == ix);
    if ( 0xffff == cb_idx ) {
      ST(0) = &PL_sv_undef;
      XSRETURN(1);
    } else {
      if ( cb_off.has_value() ) res_size++;
      if ( gimme == G_ARRAY ) {
        EXTEND(SP, res_size);
        mXPUSHi(cb_idx);
        if ( res_size > 1 )
          mPUSHi(cb_off.value());
      } else {
        AV *av = newAV();
        av_push(av, newSViv(cb_idx));
        if ( res_size > 1 )
          av_push(av, newSVuv(cb_off.value()));
        mXPUSHs(newRV_noinc((SV*)av));
        XSRETURN(1);
      }
    }
  }

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
      mPUSHs(names);
      mPUSHs(dict);
    } else {
      AV *av = newAV();
      av_push(av, names);
      av_push(av, dict);
      mXPUSHs(newRV_noinc((SV*)av));
      XSRETURN(1);
    }
  }

SV *
tab_map(SV *obj, IV key)
 INIT:
  Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = e->tab_map(key);
 OUTPUT:
  RETVAL

void
tab_fields(SV *obj, IV key)
 PREINIT:
  U8 gimme = GIMME_V;
 INIT:
  std::vector<std::string_view> names;
  Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 PPCODE:
  e->tab_fields(key, names);
  if ( names.empty() ) {
    ST(0) = &PL_sv_undef;
    XSRETURN(1);
  } else {
    if ( gimme == G_ARRAY) {
      EXTEND(SP, names.size());
      for ( auto &s: names ) mPUSHs(newSVpv(s.data(), s.size()));
    } else {
      AV *av = newAV();
      for ( auto &s: names ) av_push(av, newSVpv(s.data(), s.size()));
      mXPUSHs(newRV_noinc((SV*)av));
      XSRETURN(1);
    }
  }

void
lcols(SV *obj)
 ALIAS:
  Cubin::Ced::lrows = 1
 PREINIT:
  U8 gimme = GIMME_V;
 INIT:
  std::vector<SV *> indexes;
  Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 PPCODE:
  if ( !e->get_lxx(indexes, !ix) ) {
    ST(0) = &PL_sv_undef;
    XSRETURN(1);
  } else {
    if ( gimme == G_ARRAY) {
      EXTEND(SP, indexes.size());
      for ( auto s: indexes ) mPUSHs(s);
    } else {
      AV *av = newAV();
      for ( auto s: indexes ) av_push(av, s);
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
      mPUSHi(e->get_flush());
      mPUSHi(e->get_rdr());
      mPUSHi(e->is_dirty());
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
has_pred(SV *obj)
 ALIAS:
  Cubin::Ced::pred_name = 1
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   RETVAL = 1 == ix ? e->check_pred() : e->has_pred();
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
get(SV *obj, const char *field_name)
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
   if ( !e->has_ins() )
     RETVAL = &PL_sv_undef;
   else
     RETVAL = e->get_kv(field_name);
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

void
grep(SV *obj, SV *re)
 PREINIT:
  U8 gimme = GIMME_V;
 INIT:
   Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
   std::vector<std::string_view> res;
   REGEXP *rx;
 PPCODE:
   // from https://blogs.perl.org/users/robert_acock/2025/06/learning-xs---regular-expressions.html
   if ( !SvROK(re) || SvTYPE(SvRV(re)) != SVt_REGEXP ) {
     if ( SvROK(re) )
      croak("grep: arg must be regexp, ref type %d", SvTYPE(SvRV(re)));
     else
      croak("grep: arg must be regexp, type %d", SvTYPE(re));
   }
   rx = (REGEXP *)SvRV(re);
   e->grep_kv(rx, res);
   if ( res.empty() ) {
    ST(0) = &PL_sv_undef;
    XSRETURN(1);
  } else {
    if ( gimme == G_ARRAY) {
      EXTEND(SP, res.size());
      for ( auto &s: res ) mPUSHs(newSVpv(s.data(), s.size()));
    } else {
      AV *av = newAV();
      for ( auto &s: res ) av_push(av, newSVpv(s.data(), s.size()));
      mXPUSHs(newRV_noinc((SV*)av));
      XSRETURN(1);
    }
  }

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

SV *
track(SV *obj, SV *rt)
 INIT:
  Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
  reg_pad *r= get_magic_ext<reg_pad>(rt, &ca_regtrack_magic_vt);
 CODE:
  if ( !e->has_ins() || !r )
    RETVAL = &PL_sv_undef;
  else
    RETVAL = newSViv(e->apply(r));
 OUTPUT:
  RETVAL

SV *
has_comp(SV *obj)
 INIT:
  Ced_perl *e= get_magic_ext<Ced_perl>(obj, &ca_magic_vt);
 CODE:
    RETVAL = e->has_comp();
 OUTPUT:
  RETVAL

MODULE = Cubin::Ced		PACKAGE = Cubin::Ced::Instr

SV *
line(SV *obj)
 ALIAS:
  Cubin::Ced::Instr::alt = 1
  Cubin::Ced::Instr::mask = 2
  Cubin::Ced::Instr::name = 3
  Cubin::Ced::Instr::class = 4
  Cubin::Ced::Instr::min_wait = 5
  Cubin::Ced::Instr::setp = 6
  Cubin::Ced::Instr::brt = 7
  Cubin::Ced::Instr::scbd = 8
  Cubin::Ced::Instr::scbd_type = 9
  Cubin::Ced::Instr::target = 10
  Cubin::Ced::Instr::cc = 11
  Cubin::Ced::Instr::tab_count = 12
  Cubin::Ced::Instr::has_comp = 13
  Cubin::Ced::Instr::pred_name = 14
  Cubin::Ced::Instr::itype = 15
 INIT:
  one_instr *e= get_magic_ext<one_instr>(obj, &ca_instr_magic_vt);
 CODE:
    switch(ix) {
     case 0: RETVAL = e->iv<&nv_instr::line>();
      break;
     case 1: RETVAL = e->iv<&nv_instr::alt>();
      break;
     case 2: RETVAL = e->strv<&nv_instr::mask>();
      break;
     case 3: RETVAL = e->strv<&nv_instr::name>();
      break;
     case 4: RETVAL = e->strv<&nv_instr::cname>();
      break;
     case 5: RETVAL = e->iv<&nv_instr::min_wait>();
      break;
     case 6: RETVAL = e->iv<&nv_instr::setp>();
      break;
     case 7: RETVAL = e->iv<&nv_instr::brt>();
      break;
     case 8: RETVAL = e->iv<&nv_instr::scbd>();
      break;
     case 9: RETVAL = e->iv<&nv_instr::scbd_type>();
      break;
     case 10: RETVAL = e->strv<&nv_instr::target_index>();
      break;
     case 11: RETVAL = e->strv<&nv_instr::cc_index>();
      break;
     case 12: RETVAL = e->base->tab_count(e->ins);
      break;
     case 13: RETVAL = e->base->has_comp(e->rend);
      break;
     case 14: RETVAL = e->base->check_pred(e->rend);
      break;
     case 15: RETVAL = e->iv<&nv_instr::itype>();
      break;
     default: croak("unknown ix %d in Cubin::Ced::Instr", ix);
    }
 OUTPUT:
  RETVAL

SV *
ins_conv(SV *obj)
 INIT:
  one_instr *e= get_magic_ext<one_instr>(obj, &ca_instr_magic_vt);
 CODE:
   RETVAL = form_float_conv(e->ins->vf_conv);
 OUTPUT:
  RETVAL

SV *
ins_cb(SV *obj)
 INIT:
  one_instr *e= get_magic_ext<one_instr>(obj, &ca_instr_magic_vt);
 CODE:
   RETVAL = e->base->extract_cb(e->ins);
 OUTPUT:
  RETVAL

SV *
efield(SV *obj, const char *fname)
 ALIAS:
  Cubin::Ced::Instr::vfield = 1
 INIT:
  one_instr *e= get_magic_ext<one_instr>(obj, &ca_instr_magic_vt);
 CODE:
   RETVAL = (ix == 1) ? e->base->extract_vfield(e->ins, fname) : e->base->extract_efield(e->ins, fname);
 OUTPUT:
  RETVAL

SV *
efields(SV *obj)
 ALIAS:
  Cubin::Ced::Instr::vfields = 1
 INIT:
  one_instr *e= get_magic_ext<one_instr>(obj, &ca_instr_magic_vt);
 CODE:
   RETVAL = (ix == 1) ? e->base->extract_vfields(e->ins) : e->base->extract_efields(e->ins);
 OUTPUT:
  RETVAL

SV *
has_tfield(SV *obj, const char *tfname)
 INIT:
   one_instr *e= get_magic_ext<one_instr>(obj, &ca_instr_magic_vt);
   std::string_view tmp{tfname};
 CODE:
   RETVAL = e->base->has_tfield(e->ins, tmp);
 OUTPUT:
  RETVAL

void
tab_fields(SV *obj, IV key)
 PREINIT:
  U8 gimme = GIMME_V;
 INIT:
  std::vector<std::string_view> names;
  one_instr *e= get_magic_ext<one_instr>(obj, &ca_instr_magic_vt);
 PPCODE:
  e->base->tab_fields(e->ins, key, names);
  if ( names.empty() ) {
    ST(0) = &PL_sv_undef;
    XSRETURN(1);
  } else {
    if ( gimme == G_ARRAY) {
      EXTEND(SP, names.size());
      for ( auto &s: names ) mPUSHs(newSVpv(s.data(), s.size()));
    } else {
      AV *av = newAV();
      for ( auto &s: names ) av_push(av, newSVpv(s.data(), s.size()));
      mXPUSHs(newRV_noinc((SV*)av));
      XSRETURN(1);
    }
  }

SV *
tab_map(SV *obj, IV key)
 INIT:
  one_instr *e= get_magic_ext<one_instr>(obj, &ca_instr_magic_vt);
 CODE:
   RETVAL = e->base->tab_map(e->ins, key);
 OUTPUT:
  RETVAL

void
tab(SV *obj, IV key)
 PREINIT:
  U8 gimme = GIMME_V;
 INIT:
  bool res;
  SV *names = nullptr, *dict = nullptr;
  one_instr *e= get_magic_ext<one_instr>(obj, &ca_instr_magic_vt);
 PPCODE:
  res = e->base->get_tab(e->ins, key, &names, &dict);
  if ( !res ) {
    ST(0) = &PL_sv_undef;
    XSRETURN(1);
  } else {
    if ( gimme == G_ARRAY) {
      EXTEND(SP, 2);
      mPUSHs(names);
      mPUSHs(dict);
    } else {
      AV *av = newAV();
      av_push(av, names);
      av_push(av, dict);
      mXPUSHs(newRV_noinc((SV*)av));
      XSRETURN(1);
    }
  }

SV *
prop(SV *obj)
 INIT:
  one_instr *e= get_magic_ext<one_instr>(obj, &ca_instr_magic_vt);
 CODE:
   RETVAL = e->base->ins_prop(e->ins);
 OUTPUT:
  RETVAL

SV *
grep_prop(SV *obj, IV key)
 INIT:
  one_instr *e= get_magic_ext<one_instr>(obj, &ca_instr_magic_vt);
 CODE:
   RETVAL = e->base->grep_prop(key, e->ins);
 OUTPUT:
  RETVAL

SV *
render(SV *obj)
 INIT:
  AV *fake;
  SV *objref= NULL;
  MAGIC* magic;
  RItems *res;
  one_instr *e= get_magic_ext<one_instr>(obj, &ca_instr_magic_vt);
PPCODE:
  res = new RItems;
  if ( e->base->make_render(*res, e->rend) ) {
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
  auto *d = magic_tied<RItems>(self, 1, &ca_rend_magic_vt);
 PPCODE:
  if ( key >= d->size() || key < 0 ) {
    ST(0) = &PL_sv_undef;
  } else {
    ST(0) = get_ritem(d->at(key));
  }
  XSRETURN(1);

SV *
has_mem(self)
  SV *self;
INIT:
  SV *res = &PL_sv_no;
  auto *d = magic_tied<RItems>(self, 1, &ca_rend_magic_vt);
 CODE:
   for ( auto &r: *d ) {
     if ( r.first->type > R_opcode ) {
       res = &PL_sv_yes;
       break;
     }
   }
   RETVAL = res;
 OUTPUT:
  RETVAL

void
grep(self, key)
  SV *self;
  IV key;
 PREINIT:
  U8 gimme = GIMME_V;
INIT:
  auto *d = magic_tied<RItems>(self, 1, &ca_rend_magic_vt);
  std::vector<RItem *> tmp;
 PPCODE:
  // collect
  for ( auto &r: *d ) {
    if ( r.first->type == key ) tmp.push_back(&r);
  }
  if ( tmp.empty() ) {
    ST(0) = &PL_sv_undef;
    XSRETURN(1);
  } else {
    if ( gimme == G_ARRAY) {
      EXTEND(SP, tmp.size());
      for ( auto r: tmp ) {
        mPUSHs(get_ritem(*r));
      }
    } else {
      AV *av = newAV();
      for ( auto r: tmp ) {
        auto item = get_ritem(*r);
        av_push(av, item);
      }
      mXPUSHs(newRV_noinc((SV*)av));
      XSRETURN(1);
    }
  }

MODULE = Cubin::Ced		PACKAGE = Cubin::Ced::LatIndex

IV
idx(SV *obj)
 INIT:
    lat_idx *li = get_magic_ext<lat_idx>(obj, &ca_latindex_magic_vt);
 CODE:
   RETVAL = li->tr->idx;
 OUTPUT:
  RETVAL

int
is_col(SV *obj)
 ALIAS:
  Cubin::Ced::LatIndex::is_row = 1
 INIT:
   lat_idx *li = get_magic_ext<lat_idx>(obj, &ca_latindex_magic_vt);
 CODE:
   if ( ix == 1 ) RETVAL = !li->is_col;
   else RETVAL = li->is_col;
 OUTPUT:
  RETVAL

SV *
name(SV *obj)
 INIT:
   lat_idx *li = get_magic_ext<lat_idx>(obj, &ca_latindex_magic_vt);
   const NV_gnames &what = li->is_col ? li->tr->tab->cols : li->tr->tab->rows;
 CODE:
   if ( li->tr->idx >= what.size() ) RETVAL = &PL_sv_undef;
   else {
     auto &sv = *(what.begin() + li->tr->idx);
     RETVAL = newSVpv(sv.first, strlen(sv.first));
   }
 OUTPUT:
  RETVAL

const char *
tab_name(SV *obj)
 ALIAS:
  Cubin::Ced::LatIndex::conn_name = 1
 INIT:
   lat_idx *li = get_magic_ext<lat_idx>(obj, &ca_latindex_magic_vt);
 CODE:
   RETVAL = (ix == 1) ? li->tr->tab->connection : li->tr->tab->name;
 OUTPUT:
  RETVAL

IV
line(SV *obj)
 INIT:
   lat_idx *li = get_magic_ext<lat_idx>(obj, &ca_latindex_magic_vt);
 CODE:
   RETVAL = li->tr->tab->line;
 OUTPUT:
  RETVAL

UV
tab(SV *obj)
 INIT:
   lat_idx *li = get_magic_ext<lat_idx>(obj, &ca_latindex_magic_vt);
 CODE:
   // extemly bad idea but I need key to group results by real table and their names can be non-uniq
   RETVAL = (UV)li->tr->tab;
 OUTPUT:
  RETVAL

SV *
at(SV *obj, SV *other)
 INIT:
   lat_idx *idx1 = get_magic_ext<lat_idx>(obj, &ca_latindex_magic_vt),
           *idx2 = get_magic_ext<lat_idx>(other, &ca_latindex_magic_vt);
 CODE:
   // check if both indexes refers to the same table
   if ( idx1->tr->tab != idx2->tr->tab ) RETVAL = &PL_sv_undef;
   else {
     // check if one is column and other is row
     if ( idx1->is_col == idx2->is_col ) RETVAL = &PL_sv_undef;
     else {
       std::optional<short> res;
       // tab_get(col, row)
       if ( idx1->is_col ) // this is column, then idx2 is row
         res = idx1->tr->tab->get(idx1->tr->idx, idx2->tr->idx);
       else // this is row, then idx2 is column
         res = idx2->tr->tab->get(idx2->tr->idx, idx1->tr->idx);
       if ( !res.has_value() ) RETVAL = &PL_sv_undef;
       else RETVAL = newSViv(res.value());
     }
   }
 OUTPUT:
  RETVAL

MODULE = Cubin::Ced		PACKAGE = Cubin::Ced::RegTrack

void
new(obj_or_pkg)
  SV *obj_or_pkg
 INIT:
  HV *pkg = NULL;
  SV *msv;
  SV *objref= NULL;
  reg_pad *res = NULL;
  int ok = 1;
 PPCODE:
  if (SvPOK(obj_or_pkg) && (pkg= gv_stashsv(obj_or_pkg, 0))) {
    if (!sv_derived_from(obj_or_pkg, s_ca_regtrack)) {
      ok = 0;
      croak("Package %s does not derive from %s", SvPV_nolen(obj_or_pkg), s_ca_regtrack);
    }
  } else {
    ok = 0;
    croak("new: first arg must be package name or blessed object");
  }
  if ( !ok ) {
    ST(0) = &PL_sv_undef;
  } else {
    res = new reg_pad;
    res->snap = new track_snap();
    msv = newSViv(0);
    objref= sv_2mortal(newRV_noinc(msv));
    sv_bless(objref, pkg);
    ST(0)= objref;
    // attach magic
    sv_magicext(msv, NULL, PERL_MAGIC_ext, &ca_regtrack_magic_vt, (const char*)res, 0);
  }
  XSRETURN(1);

void
finalize(SV *obj)
 INIT:
   reg_pad *r= get_magic_ext<reg_pad>(obj, &ca_regtrack_magic_vt);
 CODE:
   NV_renderer::finalize_rt(r);

SV *
empty(SV *obj)
 INIT:
   reg_pad *r= get_magic_ext<reg_pad>(obj, &ca_regtrack_magic_vt);
 CODE:
  RETVAL = r->empty() ? &PL_sv_yes : &PL_sv_no;
 OUTPUT:
  RETVAL

SV *
snap_empty(SV *obj)
 INIT:
   reg_pad *r= get_magic_ext<reg_pad>(obj, &ca_regtrack_magic_vt);
 CODE:
  RETVAL = r->snap->empty() ? &PL_sv_yes : &PL_sv_no;
 OUTPUT:
  RETVAL

void
clear(SV *obj)
 INIT:
   reg_pad *r= get_magic_ext<reg_pad>(obj, &ca_regtrack_magic_vt);
 CODE:
   r->clear();

UV
mask(SV *obj)
 ALIAS:
  Cubin::Ced::RegTrack::mask2 = 1
 INIT:
   reg_pad *r= get_magic_ext<reg_pad>(obj, &ca_regtrack_magic_vt);
 CODE:
   RETVAL = (ix == 1) ? r->m_reuse.mask2 : r->m_reuse.mask;
 OUTPUT:
  RETVAL

IV keep(SV *obj)
 ALIAS:
  Cubin::Ced::RegTrack::keep2 = 1
 INIT:
   reg_pad *r= get_magic_ext<reg_pad>(obj, &ca_regtrack_magic_vt);
 CODE:
   RETVAL = (ix == 1) ? r->m_reuse.keep2 : r->m_reuse.keep;
 OUTPUT:
  RETVAL

void
snap_clear(SV *obj)
 INIT:
   reg_pad *r= get_magic_ext<reg_pad>(obj, &ca_regtrack_magic_vt);
 CODE:
   r->snap->reset();

void
snap(SV *obj)
 PREINIT:
  U8 gimme = GIMME_V;
 INIT:
   reg_pad *r= get_magic_ext<reg_pad>(obj, &ca_regtrack_magic_vt);
 PPCODE:
  if ( r->snap->empty() ) {
    ST(0) = &PL_sv_undef;
    XSRETURN(1);
  } else {
    auto regs = gprs(r->snap);
    auto prs = merge_preds(r->snap);
    if ( gimme == G_ARRAY) {
      int esize = 1;
      if ( prs != &PL_sv_undef ) esize++;
      EXTEND(SP, esize);
      mPUSHs(regs);
      if( esize > 1 ) mPUSHs(prs);
    } else {
      AV *av = newAV();
      av_push(av, regs);
      if ( prs != &PL_sv_undef ) av_push(av, prs);
      mXPUSHs(newRV_noinc((SV*)av));
      XSRETURN(1);
    }
  }

SV *
cbs(SV *obj)
 PREINIT:
  U8 gimme = GIMME_V;
 INIT:
   reg_pad *r= get_magic_ext<reg_pad>(obj, &ca_regtrack_magic_vt);
 PPCODE:
  if ( r->cbs.empty() ) {
    ST(0) = &PL_sv_undef;
    XSRETURN(1);
  } else {
    if ( gimme == G_ARRAY) {
      EXTEND(SP, r->cbs.size());
      for ( auto &cbh: r->cbs ) { mPUSHs(make_one_cb(cbh)); }
    } else {
      AV *av = newAV();
      for ( auto &cbh: r->cbs ) av_push(av, make_one_cb(cbh));
      mXPUSHs(newRV_noinc((SV*)av));
      XSRETURN(1);
    }
  }

SV *
rs(SV *obj)
 ALIAS:
  Cubin::Ced::RegTrack::urs = 1
 INIT:
   reg_pad *r= get_magic_ext<reg_pad>(obj, &ca_regtrack_magic_vt);
   auto &rs = (ix == 1) ? r->ugpr: r->gpr;
 CODE:
  RETVAL = fill_rhash(rs);
 OUTPUT:
  RETVAL

SV *
ps(SV *obj)
 ALIAS:
  Cubin::Ced::RegTrack::ups = 1
 INIT:
   reg_pad *r= get_magic_ext<reg_pad>(obj, &ca_regtrack_magic_vt);
   auto &rs = (ix == 1) ? r->upred: r->pred;
 CODE:
  RETVAL = fill_rhash(rs);
 OUTPUT:
  RETVAL

SV *
r(SV *obj, IV key, unsigned long from = 0)
 ALIAS:
  Cubin::Ced::RegTrack::ur = 1
 INIT:
   reg_pad *r= get_magic_ext<reg_pad>(obj, &ca_regtrack_magic_vt);
   auto &rs = (ix == 1) ? r->ugpr: r->gpr;
   auto rs_iter = rs.find(key);
 CODE:
  if ( rs_iter == rs.end() ) RETVAL = &PL_sv_undef;
  else RETVAL = fill_reg(rs_iter->second, from);
 OUTPUT:
  RETVAL

SV *
p(SV *obj, IV key, unsigned long from = 0)
 ALIAS:
  Cubin::Ced::RegTrack::up = 1
 INIT:
   reg_pad *r= get_magic_ext<reg_pad>(obj, &ca_regtrack_magic_vt);
   auto &rs = (ix == 1) ? r->upred: r->pred;
   auto rs_iter = rs.find(key);
 CODE:
  if ( rs_iter == rs.end() ) RETVAL = &PL_sv_undef;
  else RETVAL = fill_reg(rs_iter->second, from);
 OUTPUT:
  RETVAL

BOOT:
 s_ca_pkg = gv_stashpv(s_ca, 0);
 if ( !s_ca_pkg )
    croak("Package %s does not exists", s_ca);
 s_ca_render_pkg = gv_stashpv(s_ca_render, 0);
 if ( !s_ca_render_pkg )
    croak("Package %s does not exists", s_ca_render);
 s_ca_regtrack_pkg = gv_stashpv(s_ca_regtrack, 0);
 if ( !s_ca_regtrack_pkg )
    croak("Package %s does not exists", s_ca_regtrack);
 s_ca_instr_pkg = gv_stashpv(s_ca_instr, 0);
 if ( !s_ca_instr_pkg )
    croak("Package %s does not exists", s_ca_instr);
 s_ca_latindex_pkg = gv_stashpv(s_ca_latindex, 0);
 if ( !s_ca_latindex_pkg )
    croak("Package %s does not exists", s_ca_latindex);
 // add enums from nv_types.h
 HV *stash = gv_stashpvn(s_ca, 10, 1);
 EXPORT_ENUM(NVP_ops, IDEST)
 EXPORT_ENUM(NVP_ops, IDEST2)
 EXPORT_ENUM(NVP_ops, ISRC_A)
 EXPORT_ENUM(NVP_ops, ISRC_B)
 EXPORT_ENUM(NVP_ops, ISRC_C)
 EXPORT_ENUM(NVP_ops, ISRC_E)
 EXPORT_ENUM(NVP_ops, ISRC_H)
 EXPORT_ENUM(NVP_ops, ISRC_I)
 EXPORT_ENUM(NV_IType, ITYPE_ABC_REG)
 EXPORT_ENUM(NV_IType, ITYPE_ABC_BCST)
 EXPORT_ENUM(NV_IType, ITYPE_ABC_CCST)
 EXPORT_ENUM(NV_IType, ITYPE_ABC_B20I)
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
 EXPORT_ENUM(NV_Format, NV_E8M7Imm)
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
 // export cuda relocs, 0 - R_NONE so skip it
 for ( unsigned int rt = 1; ; rt++ ) {
   auto name = get_cuda_reloc_name(rt);
   if ( !name ) break;
   SV *rel_v = newSViv(rt);
   SvREADONLY_on(rel_v);
   newCONSTSUB(stash, name, rel_v);
 }