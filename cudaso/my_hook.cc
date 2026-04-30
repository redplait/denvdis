#include <dlfcn.h>
#include <pthread.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include "de_bg_data.h"
#include "decuda_data.h"
#include "simple_api.h"
#include "cereal/archives/json.hpp"

// dummy pthread_rwlock_t RAII classes
struct rd_raii {
  explicit rd_raii(pthread_rwlock_t *r) {
    pthread_rwlock_rdlock(r);
    m_lock = r;
  }
  ~rd_raii() {
    pthread_rwlock_unlock(m_lock);
  }
 protected:
  pthread_rwlock_t *m_lock;
};

struct wr_raii {
  explicit wr_raii(pthread_rwlock_t *r) {
    pthread_rwlock_wrlock(r);
    m_lock = r;
  }
  ~wr_raii() {
    pthread_rwlock_unlock(m_lock);
  }
 protected:
  pthread_rwlock_t *m_lock;
};

typedef void *(*Tdlsym)(void*, const char*);
typedef long (*Texp_tab)(const void **, unsigned char *);
static Tdlsym real_sym = NULL;
static Texp_tab real_exp_tab = NULL; // original cuGetExportTable

#ifdef HOOK_nvPTX
typedef int (*Tcompile)(void *, int, const char *const *);
typedef int (*Tcompiler_create)(void **, size_t, const char *);
typedef int (*Tcompiler_destroy)(void *);
typedef int (*Tcompiler_get)(void *, void *);
typedef int (*Tcompiler_getlen)(void *, size_t *);

static Tcompile real_compile = NULL;
static Tcompiler_create real_compile_create = NULL;
static Tcompiler_destroy real_compile_destroy = NULL;
static Tcompiler_get real_compile_get = NULL;
static Tcompiler_getlen real_compile_getlen = NULL;

// hash for holding pairs ctx -> size_t from nvPTXCompilerGetCompiledProgram
static std::unordered_map<void *, size_t> s_cpsizes;
static pthread_rwlock_t cpsizes_lock = PTHREAD_RWLOCK_INITIALIZER;
#endif

static pthread_once_t once_dlsym = PTHREAD_ONCE_INIT,
 once_de_bg = PTHREAD_ONCE_INIT;
static FILE *s_log = NULL;
static de_bg_data s_de_data;
static decuda_data s_cudata;
static bool has_cudata = false;
static void *sh_deb = NULL,
 *sh_cuda = NULL;

/* resolve real dlsym, open log and read json from home dir */
static void init_dlsym(void) {
  // stolen from https://github.com/kentstone84/APEX-GPU/blob/main/apex_dlsym_intercept.c
  real_sym = (Tdlsym)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
  if (!real_sym) real_sym = (Tdlsym)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.0");
  if (!real_sym) {
    vlog("cannot resolve real dlsym\n");
    exit(1);
  }
  // make log filename
  auto s = getenv("TMPDIR");
  std::string t_path;
  if ( s ) t_path = s;
  else {
    s = getenv("TMP");
    if ( s ) t_path = s;
    else {
      s = getenv("TEMP");
      if ( s ) t_path = s;
      else t_path = "/tmp";
    }
  }
  auto pid = getpid();
  if ( !t_path.ends_with("/") ) t_path += "/";
  t_path += "cuda.";
  t_path += std::to_string(pid);
  t_path += ".log";
  s_log = fopen(t_path.c_str(), "a");
  if ( !s_log ) {
    vlog("cannot create log %s\n", t_path.c_str());
    exit(1);
  }
  set_logger_fp(s_log);
  // read json from home dir
  s = getenv("HOME");
  if ( !s ) {
    vlog("cannot get HOME\n");
    exit(1);
  }
  t_path = s;
  if ( !t_path.ends_with("/") ) t_path += "/";
  std::string js_name = t_path;
  js_name += json_deb;
  {
    std::ifstream i_de(js_name);
    if ( !i_de.is_open() ) {
      vlog("cannot open deb.json at %s\n", js_name.c_str());
      exit(1);
    }
    cereal::JSONInputArchive archive( i_de );
    s_de_data.load(archive);
  }
  /* and second for cuda */
  js_name = t_path;
  js_name += json_cuda;
  {
    std::ifstream i_de(js_name);
    if ( i_de.is_open() ) {
      cereal::JSONInputArchive archive( i_de );
      archive( s_cudata );
      has_cudata = true;
    } else {
     vlog("cannot open deb.json at %s\n", js_name.c_str());
    }
  }
}

// implemented in de_bg.cc
extern int simple_hook_debg(void *, FILE *, de_bg_data *);

/* hook debugger - called after init_dlsym */
void hook_de_bg() {
  if ( is_debg_patched() ) return; // in case if was called check_cudbg
  auto api = real_sym(sh_deb, de_api);
  if ( !api ) {
    vlog("cannot find %s\n", de_api);
    exit(3);
  }
  simple_hook_debg(api, s_log, &s_de_data);
}

// cuGetExportTable hook
long my_exp_tab(const void **tab, unsigned char *uuid) {
  if ( !tab || !uuid ) return real_exp_tab(tab, uuid);
  auto res = real_exp_tab(tab, uuid);
  if ( res ) // error
    vlog("%8.8X-%4.4hX-%4.4hX-%2.2X%2.2X-%2.2X%2.2X%2.2X%2.2X%2.2X%2.2X %d\n",
     *(uint32_t *)(uuid), *(unsigned short *)(uuid + 4), *(unsigned short *)(uuid + 6),
     uuid[8], uuid[9], uuid[10], uuid[11], uuid[12], uuid[13], uuid[14], uuid[15], res
    );
  else
    vlog("%8.8X-%4.4hX-%4.4hX-%2.2X%2.2X-%2.2X%2.2X%2.2X%2.2X%2.2X%2.2X %d %p\n",
     *(uint32_t *)(uuid), *(unsigned short *)(uuid + 4), *(unsigned short *)(uuid + 6),
     uuid[8], uuid[9], uuid[10], uuid[11], uuid[12], uuid[13], uuid[14], uuid[15],
     res, *tab
    );
  return res;
}

#ifdef HOOK_nvPTX
int my_compile(void *ctx, int len, const char *const *opts) {
  auto res = real_compile(ctx, len, opts);
  vlog_slist("compile %p %d\n", len, opts, ctx, res);
  return res;
}

int my_compiler_destroy(void *ctx) {
  auto res = real_compile_destroy(ctx);
  vlog("compiler_destroy %p %d\n", ctx, res);
  {
    wr_raii tmp(&cpsizes_lock);
    auto it = s_cpsizes.find(ctx);
    if ( it != s_cpsizes.end() ) s_cpsizes.erase(it);
  }
  return res;
}

int my_compile_get(void *ctx, void *out_buf) {
  auto res = real_compile_get(ctx, out_buf);
  std::optional<size_t> len;
  if ( res )
    vlog("compiler_get %p %d\n", ctx, res);
  else {
    rd_raii tmp(&cpsizes_lock);
    auto it = s_cpsizes.find(ctx);
    if ( it != s_cpsizes.end() ) len.emplace(it->second);
  }
  // TODO: do here something with out_buf, len in len.value()
  if ( len.has_value() ) {
    vlog("can dump len %d\n", len.value());
  }
  return res;
}

int my_compile_getlen(void *ctx, size_t *out_len) {
  auto res = real_compile_getlen(ctx, out_len);
  if ( res )
    vlog("compiler_getlen %p %d\n", ctx, res);
  else {
    vlog("compiler_getlen %p %d %d\n", ctx, res, *out_len);
    if ( *out_len ) {
      wr_raii tmp(&cpsizes_lock);
      s_cpsizes[ctx] = *out_len;
    }
  }
  return res;
}
#endif

void* dlsym(void* handle, const char* symbol) {
  pthread_once(&once_dlsym, init_dlsym);
  if ( symbol ) {
    // for debugging
    vlog("dlsym %s\n", symbol);
    if ( !strcmp(symbol, "DebugAgentMain") ) {
      sh_deb = handle;
      pthread_once(&once_de_bg, hook_de_bg);
    } else if ( !strcmp(symbol, de_api) ) {
      sh_deb = handle;
      pthread_once(&once_de_bg, hook_de_bg);
    } else if ( !strcmp(symbol, "cuGetExportTable") ) {
      sh_cuda = handle;
      if ( !real_exp_tab )
        real_exp_tab = (Texp_tab)real_sym(handle, symbol);
      if ( real_exp_tab ) return (void *)&my_exp_tab;
    } else if ( symbol[0] == 'n' ) {
#ifdef HOOK_nvPTX
      if ( !strcmp(symbol, "nvPTXCompilerCompile") ) {
        if ( !real_compile ) real_compile = (Tcompile)real_sym(handle, symbol);
        if ( real_compile ) return (void *)&my_compile;
      } else if ( !strcmp(symbol, "nvPTXCompilerDestroy") ) {
        if ( !real_compile_destroy ) real_compile_destroy = (Tcompiler_destroy)real_sym(handle, symbol);
        if ( real_compile_destroy ) return (void *)&my_compiler_destroy;
      } else if ( !strcmp(symbol, "nvPTXCompilerGetCompiledProgramSize") ) {
        if ( !real_compile_getlen ) real_compile_getlen = (Tcompiler_getlen)real_sym(handle, symbol);
        if ( real_compile_getlen ) return (void *)&my_compile_getlen;
      } else if ( !strcmp(symbol, "nvPTXCompilerGetCompiledProgram") ) {
        if ( !real_compile_get ) real_compile_get = (Tcompiler_get)real_sym(handle, symbol);
        if ( real_compile_get ) return (void *)&my_compile_get;
      }
#endif
    }
  }
  return real_sym(handle, symbol);
}
