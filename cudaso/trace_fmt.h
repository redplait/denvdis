#pragma once

/* trace packet formats
  packet_type 2 - init/finit
  packet_type 6 - API call, function name at 0x30
*/

//                              rdi            rsi              edx             rcx           r8
typedef void (*dbg_trace)(void *user_data, int packet_type, int func_num, void *packer, void *ud2);