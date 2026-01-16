#pragma once

/* trace packet formats */
struct trace_v6 {
/* 0 */  uint64_t size;
/* 8 */  const char *name;
};

//                              rdi            rsi              edx             rcx           r8
typedef void (*dbg_trace)(void *user_data, int packet_type, int func_num, void *packer, void *ud2);