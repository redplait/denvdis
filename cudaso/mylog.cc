#include <stdint.h>
#include <stdio.h>
#include <mutex>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include "trace_fmt.h"

static dbg_trace old_handler = nullptr;
static uint64_t logger_addr;
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


// my logger
static void my_logger(void *user_data, int packet_type, int func_num, void *packet, void *ud2) {
  time_t t = time(NULL);
  struct tm ltm;
  localtime_r(&t, &ltm);
  char stime[200];
  strftime(stime, sizeof(stime), "%d/%m/%Y %H:%M:%S", &ltm);
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


// patcher
int patch_dbg(uint64_t addr, uint64_t data_addr, FILE *fp, const unsigned char *mask, size_t mask_size) {
  memset(hex_masks, 0, sizeof(hex_masks));
  for ( int i = 0; i < std::min(mask_size, sizeof(hex_masks)); i++ ) {
    hex_masks[i] = ( mask[i] & 2 ) ? 1 : 0;
  }
  logger_addr = addr;
  logger_data = (void **)data_addr;
  old_handler = *(dbg_trace *)addr;
  *(dbg_trace *)addr = my_logger;
  s_fp = fp;
  return 1;
};

extern "C" int reset_logger() {
  if ( !logger_addr ) return 0;
  auto f_copy = s_fp;
  {
    std::lock_guard tmp(s_mtx);
    *(dbg_trace *)logger_addr = old_handler;
    s_fp = nullptr;
  }
  if ( f_copy != stdout ) fclose(f_copy);
  return 1;
}

// debugger logging
typedef void (*debugger_trace)(const char *);
static debugger_trace s_dbg_68 = nullptr;

void my_dbg_trace(const char *packet) {
  time_t t = time(NULL);
  struct tm ltm;
  localtime_r(&t, &ltm);
  char stime[200];
  strftime(stime, sizeof(stime), "%d/%m/%Y %H:%M:%S", &ltm);
  const char **name = (const char **)(packet + 0x28);
  fprintf(s_fp, "%s %s\n", stime, *name);
  // hexdump
  HexDump((const unsigned char *)packet, 0x30);
  if ( s_dbg_68 ) s_dbg_68(packet);
}