#pragma once

/* trace packet formats
  packet_type 2 - init/finit
  packet_type 6 - API call, function name at 0x30
  packet_type 7 - called from A094798C-2E74-2E74-93F2-0800200C0A66 callback hooks interface
  packet_type 0xC - cudart
  packet_type 0x14 - ctx manipulation
*/

//                              rdi            rsi              edx             rcx           r8
typedef void (*dbg_trace)(void *user_data, int packet_type, int func_num, void *packer, void *ud2);