#include <dlfcn.h>
#include <pthread.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include "de_bg_data.h"
#include "decuda_data.h"
#include "simple_api.h"
#include "cereal/archives/json.hpp"

typedef void *(*Tdlsym)(void*, const char*);
typedef long (*Texp_tab)(const void **, unsigned char *);
static Tdlsym real_sym = NULL;
static Texp_tab real_exp_tab = NULL; // original cuGetExportTable

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
  vlog("%8.8X-%4.4hX-%4.4hX-%2.2X%2.2X-%2.2X%2.2X%2.2X%2.2X%2.2X%2.2X %d %p",
   *(uint32_t *)(uuid), *(unsigned short *)(uuid + 4), *(unsigned short *)(uuid + 6),
   uuid[8], uuid[9], uuid[10], uuid[11], uuid[12], uuid[13], uuid[14], uuid[15],
   res, *tab
  );
  return res;
}

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
    }
  }
  return real_sym(handle, symbol);
}
