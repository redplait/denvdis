#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>
#include <mutex>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include "trace_fmt.h"

static std::mutex patch_mtx;
// patch data - protected with patch_mtx
static dbg_trace old_handler = nullptr;
static uint64_t logger_addr = 0;
static void **logger_data = nullptr;

#define STORED_MASKS   0x60

static unsigned char hex_masks[STORED_MASKS];
static std::mutex s_mtx;
static FILE *s_fp = nullptr;

static const char hexes[] = "0123456789ABCDEF";

static void HexDump(const unsigned char *From, int Len)
{
 int i;
 int j,k;
 char buffer[256];
 char *ptr;

 for(i=0;i<Len;)
     {
          ptr = buffer;
          sprintf(ptr, "%08X ",i);
          ptr += 9;
          for(j=0;j<16 && i<Len;j++,i++)
          {
             *ptr++ = j && !(j%4)?(!(j%8)?'|':'-'):' ';
             *ptr++ = hexes[From[i] >> 4];
             *ptr++ = hexes[From[i] & 0xF];
          }
          for(k=16-j;k!=0;k--)
          {
            ptr[0] = ptr[1] = ptr[2] = ' ';
            ptr += 3;

          }
          ptr[0] = ptr[1] = ' ';
          ptr += 2;
          for(;j!=0;j--)
          {
               if(From[i-j]>=0x20 && From[i-j]<0x80)
                   *ptr = From[i-j];

               else
                    *ptr = '.';
               ptr++;
          }
          *ptr = 0;
          fprintf(s_fp, "%s\n", buffer);
     }
     fprintf(s_fp, "\n");
}

#define MAKE_TS    time_t t = time(NULL); \
  struct tm ltm; \
  localtime_r(&t, &ltm); \
  char stime[200]; \
  strftime(stime, sizeof(stime), "%d/%m/%Y %H:%M:%S", &ltm);


// cuda logger
static void my_logger(void *user_data, int packet_type, int func_num, void *packet, void *ud2) {
  MAKE_TS
  const unsigned char *body = (const unsigned char *)packet;
  int need_hex = hex_masks[packet_type];
  {
    std::lock_guard tmp(s_mtx);
    if ( s_fp ) {
      fprintf(s_fp, "%s PID %d packet %d func %d\n", stime, getpid(), packet_type, func_num);
      // type 6 at 0x30 is function name
      if ( 6 == packet_type && packet )
        fprintf(s_fp, "%s\n", *(char **)(body + 0x30));
      // type 2 can pass null as packet
      if ( need_hex && packet ) HexDump(body, *(uint32_t *)packet);
    }
  }
  if ( old_handler )
    old_handler(user_data, packet_type, func_num, packet, ud2);
}

// cuda patcher
int patch_dbg(uint64_t addr, uint64_t data_addr, FILE *fp, const unsigned char *mask, size_t mask_size) {
  std::lock_guard tmp(patch_mtx);
  memset(hex_masks, 0, sizeof(hex_masks));
  for ( int i = 0; i < std::min(mask_size, sizeof(hex_masks)); i++ ) {
    hex_masks[i] = ( mask[i] & 2 ) ? 1 : 0;
  }
  logger_addr = addr;
  logger_data = (void **)data_addr;
  old_handler = *(dbg_trace *)addr;
  s_fp = fp;
  *(dbg_trace *)addr = my_logger;
  return 1;
};

extern "C" bool is_cuda_patched() {
  std::lock_guard tmp(patch_mtx);
  if ( !logger_addr ) return false;
  return *(dbg_trace *)logger_addr == my_logger;
}

// debugger logging - data protected by patch_mtx
typedef void (*debugger_trace)(const char *);
// old handler
static debugger_trace s_dbg_68 = nullptr;
// it's address
static uint64_t dbg_logger_addr = 0;

static void my_dbg_trace(const char *packet) {
  MAKE_TS
  const char **name = (const char **)(packet + 0x28);
  {
    std::lock_guard tmp(s_mtx);
    if ( s_fp ) {
      fprintf(s_fp, "%s %s\n", stime, *name);
      // hexdump
      HexDump((const unsigned char *)packet, 0x30);
    }
  }
  if ( s_dbg_68 ) s_dbg_68(packet);
}

int patch_dbg_trace(FILE *fp, uint64_t addr) {
  std::lock_guard tmp(patch_mtx);
  dbg_logger_addr = addr;
  s_dbg_68 = *(debugger_trace *)addr;
  s_fp = fp;
  *(debugger_trace *)addr = my_dbg_trace;
  return 1;
}

extern "C" bool is_debg_patched() {
  std::lock_guard tmp(patch_mtx);
  if ( !dbg_logger_addr ) return false;
  return *(debugger_trace *)dbg_logger_addr == my_dbg_trace;
}

extern "C" int reset_logger() {
  auto f_copy = s_fp;
  {
    std::lock_guard tmp(s_mtx);
    s_fp = nullptr;
    if ( dbg_logger_addr ) {
      *(debugger_trace *)dbg_logger_addr = s_dbg_68;
    }
    if ( logger_addr ) {
      *(dbg_trace *)logger_addr = old_handler;
    }
  }
  if ( f_copy && (f_copy != stdout && f_copy != stderr) ) fclose(f_copy);
  return 1;
}

extern "C" void set_logger_fp(FILE *fp) {
  std::lock_guard tmp(s_mtx);
  if ( s_fp && s_fp != fp && (s_fp != stdout && s_fp != stderr) ) fclose(s_fp);
  s_fp = fp;
}

// var arg logger
extern "C" int vlog(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  MAKE_TS
  int res = 0;
  if ( !s_fp ) {
    fprintf(stderr, "%s ", stime);
    res = vfprintf(stderr, fmt, args);
  } else {
    std::lock_guard tmp(s_mtx);
    fprintf(s_fp, "%s ", stime);
    res = vfprintf(s_fp, fmt, args);
  }
  va_end(args);
  return res;
}

extern "C" int vlog_slist(const char *fmt, size_t n, const char *const *slist, ...) {
  va_list args;
  va_start(args, slist);
  MAKE_TS
  int res = 0;
  if ( !s_fp ) {
    fprintf(stderr, "%s ", stime);
    res = vfprintf(stderr, fmt, args);
    for ( size_t i = 0; i < n; i++ ) fprintf(stderr, " [%d] %s\n", i, slist[i]);
  } else {
    std::lock_guard tmp(s_mtx);
    fprintf(s_fp, "%s ", stime);
    res = vfprintf(s_fp, fmt, args);
    for ( size_t i = 0; i < n; i++ ) fprintf(s_fp, " [%d] %s\n", i, slist[i]);
  }
  va_end(args);
  return res;
}
