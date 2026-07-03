#include "nv_rend.h"

// seems that second part (for Cnp functions) has only 2 variants
// one for maxler & pascal
static NvCBParamNames s_maxler_dbg = {
// unknown what is at offset 0 - 64bit, base of kernel?
 { 0x8, "entry_addr" }, // 32bit !!!
 { 0xC, "bar_cnt" },
 { 0x10, "trap_table" },
 { 0x18, "pause_table" },
 { 0x20, "smem_size" }, // I don't know if this is the same as %total_smem_size
};

// and this since volta
static NvCBParamNames s_volta_dbg = {
 { 0x0, "entry_addr" }, // 64 bit
 { 0x8, "trap_table" },
 { 0x10, "pause_table" },
 { 0x18, "bar_cnt" },
 { 0x1c, "smem_size" }, // I don't know if this is the same as %total_smem_size
};

// maxler - sm50, sm52, sm53
static NvCBParamNames s_maxler = {
{ 0x0, "shared_base" }, // 32bit
{ 0x4, "local_base" }, // 32bit
{ 0x8, "%ntid.x" },
{ 0xC, "%ntid.y" },
{ 0x10, "%ntid.z" },
{ 0x14, "%nctaid.x" },
{ 0x18, "%nctaid.y" },
{ 0x1C, "%nctaid.z" },
{ 0x20, "stack_ptr" },
{ 0x24, "%total_smem_size" },
{ 0x28, "%gridid" }, // 64 bit
{ 0x30, "%envreg0" },
{ 0x34, "%envreg1" },
{ 0x38, "%envreg2" },
{ 0x3C, "%envreg3" },
{ 0x40, "%envreg4" },
{ 0x44, "%envreg5" },
{ 0x48, "%envreg6" },
{ 0x4C, "%envreg7" },
{ 0x50, "%envreg8" },
{ 0x54, "%envreg9" },
{ 0x58, "%envreg10" },
{ 0x5C, "%envreg11" },
{ 0x60, "%envreg12" },
{ 0x64, "%envreg13" },
{ 0x68, "%envreg14" },
{ 0x6C, "%envreg15" },
{ 0x70, "%envreg16" },
{ 0x74, "%envreg17" },
{ 0x78, "%envreg18" },
{ 0x7C, "%envreg19" },
{ 0x80, "%envreg20" },
{ 0x84, "%envreg21" },
{ 0x88, "%envreg22" },
{ 0x8C, "%envreg23" },
{ 0x90, "%envreg24" },
{ 0x94, "%envreg25" },
{ 0x98, "%envreg26" },
{ 0x9C, "%envreg27" },
{ 0xa0, "%envreg28" },
{ 0xa4, "%envreg29" },
{ 0xa8, "%envreg30" },
{ 0xaC, "%envreg31" },
{ 0xB0, "txq_desc_table" },
{ 0xB8, "samp_desc_table" },
{ 0xC0, "surf_desc_table" },
{ 0xC8, "cb0_ptr" }, // ptr to const bank 0, don't ask me where they hide cb2, cb1 below
{ 0xD0, "cb3_ptr" }, // ptr to const bank 3 - user_consts
{ 0xD8, "cb4_ptr" }, // ptr to const bank 4
{ 0xE0, "cb5_ptr" }, // ptr to const bank 5
{ 0xE8, "cb5_ptr" }, // ptr to const bank 6
{ 0xF0, "cb1_ptr" }, // ptr to const bank 1
{ 0xF8, "%nsmid" },
{ 0xFC, "%dynamic_smem_size" },
};

// pascal - sm60, sm61, sm62 - till 0x100 the same as maxler
static NvCBParamNames s_pascal = {
{ 0x0, "shared_base" }, // 32bit
{ 0x4, "local_base" }, // 32bit
{ 0x8, "%ntid.x" },
{ 0xC, "%ntid.y" },
{ 0x10, "%ntid.z" },
{ 0x14, "%nctaid.x" },
{ 0x18, "%nctaid.y" },
{ 0x1C, "%nctaid.z" },
{ 0x20, "stack_ptr" },
{ 0x24, "%total_smem_size" },
{ 0x28, "%gridid" }, // 64 bit
{ 0x30, "%envreg0" },
{ 0x34, "%envreg1" },
{ 0x38, "%envreg2" },
{ 0x3C, "%envreg3" },
{ 0x40, "%envreg4" },
{ 0x44, "%envreg5" },
{ 0x48, "%envreg6" },
{ 0x4C, "%envreg7" },
{ 0x50, "%envreg8" },
{ 0x54, "%envreg9" },
{ 0x58, "%envreg10" },
{ 0x5C, "%envreg11" },
{ 0x60, "%envreg12" },
{ 0x64, "%envreg13" },
{ 0x68, "%envreg14" },
{ 0x6C, "%envreg15" },
{ 0x70, "%envreg16" },
{ 0x74, "%envreg17" },
{ 0x78, "%envreg18" },
{ 0x7C, "%envreg19" },
{ 0x80, "%envreg20" },
{ 0x84, "%envreg21" },
{ 0x88, "%envreg22" },
{ 0x8C, "%envreg23" },
{ 0x90, "%envreg24" },
{ 0x94, "%envreg25" },
{ 0x98, "%envreg26" },
{ 0x9C, "%envreg27" },
{ 0xa0, "%envreg28" },
{ 0xa4, "%envreg29" },
{ 0xa8, "%envreg30" },
{ 0xaC, "%envreg31" },
{ 0xB0, "txq_desc_table" },
{ 0xB8, "samp_desc_table" },
{ 0xC0, "surf_desc_table" },
{ 0xC8, "cb0_ptr" }, // ptr to const bank 0, don't ask me where they hide cb2, cb1 below
{ 0xD0, "cb3_ptr" }, // ptr to const bank 3 - user_consts
{ 0xD8, "cb4_ptr" }, // ptr to const bank 4
{ 0xE0, "cb5_ptr" }, // ptr to const bank 5
{ 0xE8, "cb5_ptr" }, // ptr to const bank 6
{ 0xF0, "cb1_ptr" }, // ptr to const bank 1
{ 0xF8, "%nsmid" },
{ 0xFC, "%dynamic_smem_size" },
{ 0x100, "shared_base_hi" }, // 32bit
{ 0x104, "local_base_hi" }, // 32bit
{ 0x108, "num_sm" }, // ???
// 0x10c - unknown
{ 0x110, "is_coop_mode" },
};

// volta - sm70, sm72
static NvCBParamNames s_volta = {
{ 0x0, "%ntid.x" },
{ 0x4, "%ntid.y" },
{ 0x8, "%ntid.z" },
{ 0xc, "%nctaid.x" },
{ 0x10, "%nctaid.y" },
{ 0x14, "%nctaid.z" },
{ 0x18, "shared_base" }, // 32bit
{ 0x1c, "shared_base_hi" }, // 32bit
{ 0x20, "local_base" }, // 32bit
{ 0x24, "local_base_hi" }, // 32bit
{ 0x28, "stack_ptr" }, // 32bit
{ 0x2c, "%dynamic_smem_size" },
{ 0x30, "%gridid" }, // 64 bit
// 0x38 - unknown ptr 64bit
{ 0x40, "cb0_ptr" }, // ptr to const bank 0
{ 0x48, "cb1_ptr" }, // ptr to const bank 1
{ 0x50, "cb3_ptr" }, // ptr to const bank 3 - user_consts
{ 0x58, "cb4_ptr" }, // ptr to const bank 4
{ 0x60, "cb5_ptr" }, // ptr to const bank 5
{ 0x68, "cb5_ptr" }, // ptr to const bank 6
{ 0x70, "txq_desc_table" },
{ 0x78, "samp_desc_table" },
{ 0x80, "surf_desc_table" },
{ 0x88, "%envreg0" },
{ 0x8c, "%envreg1" },
{ 0x90, "%envreg2" },
{ 0x94, "%envreg3" },
{ 0x98, "%envreg4" },
{ 0x9c, "%envreg5" },
{ 0xa0, "%envreg6" },
{ 0xa4, "%envreg7" },
{ 0xa8, "%envreg8" },
{ 0xac, "%envreg9" },
{ 0xb0, "%envreg10" },
{ 0xb4, "%envreg11" },
{ 0xb8, "%envreg12" },
{ 0xbc, "%envreg13" },
{ 0xc0, "%envreg14" },
{ 0xc4, "%envreg15" },
{ 0xc8, "%envreg16" },
{ 0xcc, "%envreg17" },
{ 0xd0, "%envreg18" },
{ 0xd4, "%envreg19" },
{ 0xd8, "%envreg20" },
{ 0xdc, "%envreg21" },
{ 0xe0, "%envreg22" },
{ 0xe4, "%envreg23" },
{ 0xe8, "%envreg24" },
{ 0xec, "%envreg25" },
{ 0xf0, "%envreg26" },
{ 0xf4, "%envreg27" },
{ 0xf8, "%envreg28" },
{ 0xfc, "%envreg29" },
{ 0x100, "%envreg30" },
{ 0x104, "%envreg31" },
{ 0x108, "%nsmid" }, // 264
{ 0x10C, "num_sm" },
{ 0x110, "is_coop_mode" },
{ 0x114, "kparams_end" }, // ptr 32 bit
{ 0x118, "cb2_ptr" }, // ptr to const bank 2 - suddenly
{ 0x120, "%current_graph_exec" },
};

// turing sm75 - like volta but they removed cb2
static NvCBParamNames s_turing = {
{ 0x0, "%ntid.x" },
{ 0x4, "%ntid.y" },
{ 0x8, "%ntid.z" },
{ 0xc, "%nctaid.x" },
{ 0x10, "%nctaid.y" },
{ 0x14, "%nctaid.z" },
{ 0x18, "shared_base" }, // 32bit
{ 0x1c, "shared_base_hi" }, // 32bit
{ 0x20, "local_base" }, // 32bit
{ 0x24, "local_base_hi" }, // 32bit
{ 0x28, "stack_ptr" }, // 32bit
{ 0x2c, "%dynamic_smem_size" },
{ 0x30, "%gridid" }, // 64 bit
// 0x38 - unknown ptr 64bit
{ 0x40, "cb0_ptr" }, // ptr to const bank 0
{ 0x48, "cb1_ptr" }, // ptr to const bank 1
{ 0x50, "cb3_ptr" }, // ptr to const bank 3 - user_consts
{ 0x58, "cb4_ptr" }, // ptr to const bank 4
{ 0x60, "cb5_ptr" }, // ptr to const bank 5
{ 0x68, "cb5_ptr" }, // ptr to const bank 6
{ 0x70, "txq_desc_table" },
{ 0x78, "samp_desc_table" },
{ 0x80, "surf_desc_table" },
{ 0x88, "%envreg0" },
{ 0x8c, "%envreg1" },
{ 0x90, "%envreg2" },
{ 0x94, "%envreg3" },
{ 0x98, "%envreg4" },
{ 0x9c, "%envreg5" },
{ 0xa0, "%envreg6" },
{ 0xa4, "%envreg7" },
{ 0xa8, "%envreg8" },
{ 0xac, "%envreg9" },
{ 0xb0, "%envreg10" },
{ 0xb4, "%envreg11" },
{ 0xb8, "%envreg12" },
{ 0xbc, "%envreg13" },
{ 0xc0, "%envreg14" },
{ 0xc4, "%envreg15" },
{ 0xc8, "%envreg16" },
{ 0xcc, "%envreg17" },
{ 0xd0, "%envreg18" },
{ 0xd4, "%envreg19" },
{ 0xd8, "%envreg20" },
{ 0xdc, "%envreg21" },
{ 0xe0, "%envreg22" },
{ 0xe4, "%envreg23" },
{ 0xe8, "%envreg24" },
{ 0xec, "%envreg25" },
{ 0xf0, "%envreg26" },
{ 0xf4, "%envreg27" },
{ 0xf8, "%envreg28" },
{ 0xfc, "%envreg29" },
{ 0x100, "%envreg30" },
{ 0x104, "%envreg31" },
{ 0x108, "%nsmid" },
{ 0x10C, "num_sm" },
{ 0x110, "is_coop_mode" },
{ 0x114, "kparams_end" }, // ptr 32 bit
{ 0x120, "%current_graph_exec" },
};

// ampere - sm80, sm86 - like turing + some additional fields
static NvCBParamNames s_ampere = {
{ 0x0, "%ntid.x" },
{ 0x4, "%ntid.y" },
{ 0x8, "%ntid.z" },
{ 0xc, "%nctaid.x" },
{ 0x10, "%nctaid.y" },
{ 0x14, "%nctaid.z" },
{ 0x18, "shared_base" }, // 32bit
{ 0x1c, "shared_base_hi" }, // 32bit
{ 0x20, "local_base" }, // 32bit
{ 0x24, "local_base_hi" }, // 32bit
{ 0x28, "stack_ptr" }, // 32bit
{ 0x2c, "%dynamic_smem_size" },
{ 0x30, "%gridid" }, // 64 bit
// 0x38 - unknown ptr 64bit
{ 0x40, "cb0_ptr" }, // ptr to const bank 0
{ 0x48, "cb1_ptr" }, // ptr to const bank 1
{ 0x50, "cb3_ptr" }, // ptr to const bank 3 - user_consts
{ 0x58, "cb4_ptr" }, // ptr to const bank 4
{ 0x60, "cb5_ptr" }, // ptr to const bank 5
{ 0x68, "cb5_ptr" }, // ptr to const bank 6
{ 0x70, "txq_desc_table" },
{ 0x78, "samp_desc_table" },
{ 0x80, "surf_desc_table" },
{ 0x88, "%envreg0" },
{ 0x8c, "%envreg1" },
{ 0x90, "%envreg2" },
{ 0x94, "%envreg3" },
{ 0x98, "%envreg4" },
{ 0x9c, "%envreg5" },
{ 0xa0, "%envreg6" },
{ 0xa4, "%envreg7" },
{ 0xa8, "%envreg8" },
{ 0xac, "%envreg9" },
{ 0xb0, "%envreg10" },
{ 0xb4, "%envreg11" },
{ 0xb8, "%envreg12" },
{ 0xbc, "%envreg13" },
{ 0xc0, "%envreg14" },
{ 0xc4, "%envreg15" },
{ 0xc8, "%envreg16" },
{ 0xcc, "%envreg17" },
{ 0xd0, "%envreg18" },
{ 0xd4, "%envreg19" },
{ 0xd8, "%envreg20" },
{ 0xdc, "%envreg21" },
{ 0xe0, "%envreg22" },
{ 0xe4, "%envreg23" },
{ 0xe8, "%envreg24" },
{ 0xec, "%envreg25" },
{ 0xf0, "%envreg26" },
{ 0xf4, "%envreg27" },
{ 0xf8, "%envreg28" },
{ 0xfc, "%envreg29" },
{ 0x100, "%envreg30" },
{ 0x104, "%envreg31" },
{ 0x108, "%nsmid" },
{ 0x10C, "num_sm" },
{ 0x110, "is_coop_mode" },
{ 0x114, "%total_smem_size" },
{ 0x118, "policy_default" },
{ 0x120, "%reserved_smem_offset_end" },
{ 0x124, "%reserved_smem_offset_1" },
// 0x128 is unknown - we could assume this is kparams_start but IT IS'NT
{ 0x12c, "kparams_end" },
{ 0x130, "%current_graph_exec" },
};

// ada - sm89 - like ampere but last field kparams_end was removed
static NvCBParamNames s_ada = {
{ 0x0, "%ntid.x" },
{ 0x4, "%ntid.y" },
{ 0x8, "%ntid.z" },
{ 0xc, "%nctaid.x" },
{ 0x10, "%nctaid.y" },
{ 0x14, "%nctaid.z" },
{ 0x18, "shared_base" }, // 32bit
{ 0x1c, "shared_base_hi" }, // 32bit
{ 0x20, "local_base" }, // 32bit
{ 0x24, "local_base_hi" }, // 32bit
{ 0x28, "stack_ptr" }, // 32bit
{ 0x2c, "%dynamic_smem_size" },
{ 0x30, "%gridid" }, // 64 bit
// 0x38 - unknown ptr 64bit
{ 0x40, "cb0_ptr" }, // ptr to const bank 0
{ 0x48, "cb1_ptr" }, // ptr to const bank 1
{ 0x50, "cb3_ptr" }, // ptr to const bank 3 - user_consts
{ 0x58, "cb4_ptr" }, // ptr to const bank 4
{ 0x60, "cb5_ptr" }, // ptr to const bank 5
{ 0x68, "cb5_ptr" }, // ptr to const bank 6
{ 0x70, "txq_desc_table" },
{ 0x78, "samp_desc_table" },
{ 0x80, "surf_desc_table" },
{ 0x88, "%envreg0" },
{ 0x8c, "%envreg1" },
{ 0x90, "%envreg2" },
{ 0x94, "%envreg3" },
{ 0x98, "%envreg4" },
{ 0x9c, "%envreg5" },
{ 0xa0, "%envreg6" },
{ 0xa4, "%envreg7" },
{ 0xa8, "%envreg8" },
{ 0xac, "%envreg9" },
{ 0xb0, "%envreg10" },
{ 0xb4, "%envreg11" },
{ 0xb8, "%envreg12" },
{ 0xbc, "%envreg13" },
{ 0xc0, "%envreg14" },
{ 0xc4, "%envreg15" },
{ 0xc8, "%envreg16" },
{ 0xcc, "%envreg17" },
{ 0xd0, "%envreg18" },
{ 0xd4, "%envreg19" },
{ 0xd8, "%envreg20" },
{ 0xdc, "%envreg21" },
{ 0xe0, "%envreg22" },
{ 0xe4, "%envreg23" },
{ 0xe8, "%envreg24" },
{ 0xec, "%envreg25" },
{ 0xf0, "%envreg26" },
{ 0xf4, "%envreg27" },
{ 0xf8, "%envreg28" },
{ 0xfc, "%envreg29" },
{ 0x100, "%envreg30" },
{ 0x104, "%envreg31" },
{ 0x108, "%nsmid" },
{ 0x10C, "num_sm" },
{ 0x110, "is_coop_mode" },
{ 0x114, "%total_smem_size" },
{ 0x118, "policy_default" },
{ 0x120, "%reserved_smem_offset_end" },
{ 0x124, "%reserved_smem_offset_1" },
// 0x128 is unknown
{ 0x130, "%current_graph_exec" },
};

// hopper - sm90
static NvCBParamNames s_hopper = {
{ 0x0, "%ntid.x" },
{ 0x4, "%ntid.y" },
{ 0x8, "%ntid.z" },
{ 0xC, "%nctaid.x" },
{ 0x10, "%nctaid.y" },
{ 0x14, "%nctaid.z" },
{ 0x28, "stack_ptr" },
{ 0x2C, "%dynamic_smem_size" },
{ 0x30, "%gridid" },
{ 0x40, "%envreg0" },
{ 0x44, "%envreg1" },
{ 0x48, "%envreg2" },
{ 0x4C, "%envreg3" },
{ 0x50, "%envreg4" },
{ 0x54, "%envreg5" },
{ 0x58, "%envreg6" },
{ 0x5C, "%envreg7" },
{ 0x60, "%envreg8" },
{ 0x64, "%envreg9" },
{ 0x68, "%envreg10" },
{ 0x6C, "%envreg11" },
{ 0x70, "%envreg12" },
{ 0x74, "%envreg13" },
{ 0x78, "%envreg14" },
{ 0x7C, "%envreg15" },
{ 0x80, "%envreg16" },
{ 0x84, "%envreg17" },
{ 0x88, "%envreg18" },
{ 0x8C, "%envreg19" },
{ 0x90, "%envreg20" },
{ 0x94, "%envreg21" },
{ 0x98, "%envreg22" },
{ 0x9C, "%envreg23" },
{ 0xA0, "%envreg24" },
{ 0xA4, "%envreg25" },
{ 0xA8, "%envreg26" },
{ 0xAC, "%envreg27" },
{ 0xB0, "%envreg28" },
{ 0xB4, "%envreg29" },
{ 0xB8, "%envreg30" },
{ 0xBC, "%envreg31" },
{ 0xC0, "cb0_ptr" }, // ptr to const bank 0
{ 0xC8, "cb1_ptr" }, // ptr to const bank 1, don't ask me where they hide cb2
{ 0xD0, "cb3_ptr" }, // ptr to const bank 3 - user_consts
{ 0xD8, "cb4_ptr" }, // ptr to const bank 4
{ 0xE0, "cb5_ptr" }, // ptr to const bank 5
{ 0xE8, "cb5_ptr" }, // ptr to const bank 6
{ 0xF0, "txq_desc_table" },
{ 0xF8, "samp_desc_table" },
{ 0x100, "surf_desc_table" },
{ 0x108, "%nsmid" },
{ 0x10C, "num_sm" },
{ 0x110, "is_coop_mode" }, // from CGS_get_intrinsic_handle
{ 0x114, "%total_smem_size" },
{ 0x120, "tools_table" }, // 8 32bit dwords
{ 0x13C, "%aggr_smem_size" },
{ 0x140, "%is_explicit_cluster" },
{ 0x144, "%cluster_nctaid.x" },
{ 0x148, "%cluster_nctaid.y" },
{ 0x14C, "%cluster_nctaid.z" },
{ 0x15C, "%nclusterid.x" },
{ 0x160, "%nclusterid.y" },
{ 0x164, "%nclusterid.z" },
{ 0x188, "%cluster_nctarank" },
{ 0x190, "%current_graph_exec" },
{ 0x198, "kparams_start" }, // ptr 64 bit
{ 0x1A0, "kparams_end" }, // ptr 64 bit
{ 0x208, "policy_default" },
};

// some support of sm100/sm103
static NvCBParamNames s_sm100 = {
{ 0xd0, "%envreg16" },
{ 0xd4, "%envreg17" },
{ 0xd8, "%envreg18" },
{ 0xdc, "%envreg19" },
{ 0xe0, "%envreg20" },
{ 0xe4, "%envreg21" },
{ 0xe8, "%envreg22" },
{ 0xec, "%envreg23" },
{ 0xf0, "%envreg24" },
{ 0xf4, "%envreg25" },
{ 0xf8, "%envreg26" },
{ 0xfc, "%envreg27" },
{ 0x100, "%envreg28" },
{ 0x104, "%envreg29" },
{ 0x108, "%envreg30" },
{ 0x10c, "%envreg31" },
{ 0x140, "%gridid" },
{ 0x16c, "%total_smem_size" },
{ 0x198, "kparams_start" }, // ptr 64 bit
{ 0x1A0, "kparams_end" }, // ptr 64 bit
{ 0x250, "%envreg0" },
{ 0x254, "%envreg1" },
{ 0x258, "%envreg2" },
{ 0x25c, "%envreg3" },
{ 0x260, "%envreg4" },
{ 0x264, "%envreg5" },
{ 0x268, "%envreg6" },
{ 0x26c, "%envreg7" },
{ 0x270, "%envreg8" },
{ 0x274, "%envreg9" },
{ 0x278, "%envreg10" }, // %reserved_smem_offset_end
{ 0x27c, "%envreg11" },
{ 0x280, "%envreg12" },
{ 0x284, "%envreg13" },
{ 0x288, "%envreg14" },
{ 0x28c, "%envreg15" },
{ 0x2a0, "%cluster_nctaid.x" },
{ 0x2a4, "%cluster_nctaid.y" },
{ 0x2a8, "%cluster_nctaid.z" },
{ 0x2ac, "%dynamic_smem_size" },
{ 0x2B0, "%clusterid.x" },
{ 0x2B4, "%clusterid.y" },
{ 0x2B8, "%clusterid.z" },
{ 0x2BC, "%aggr_smem_size" },
{ 0x2C0, "%nclusterid.x" },
{ 0x2C4, "%nclusterid.y" },
{ 0x2C8, "%nclusterid.z" },
{ 0x2cc, "%cluster_nctarank" },
{ 0x2d0, "%nsmid" },
{ 0x2d4, "is_coop_mode" }, // from CGS_get_intrinsic_handle
{ 0x2e8, "%current_graph_exec" },
{ 0x358, "policy_default" },
{ 0x360, "%ntid.x" },
{ 0x364, "%ntid.y" },
{ 0x368, "%ntid.z" },
{ 0x36c, "%is_explicit_cluster" },
{ 0x370, "%nctaid.x" },
{ 0x374, "%nctaid,y" },
{ 0x378, "%nctaid.z" },
{ 0x37c, "stack_ptr" },
};

const char *NV_renderer::cb0_name(unsigned short idx) const {
  NvCBParamNames::const_iterator what;
  if ( m_cb0.cnp_off && idx >= m_cb0.cnp_off && m_cb0.cnp ) {
    what = m_cb0.cnp->find(idx - m_cb0.cnp_off);
    if ( what == m_cb0.cnp->cend() ) return nullptr;
    return what->second;
  } else if ( m_cb0.bank0 ) {
    what = m_cb0.bank0->find(idx);
    if ( what == m_cb0.bank0->cend() ) return nullptr;
    return what->second;
  }
  return nullptr;
}

int NV_renderer::asgn_cb0() {
 // maxler - sm50, sm52, sm53
 if ( m_sm >= 0x32 && m_sm <= 0x35 ) {
   m_cb0.bank0 = &s_maxler;
   m_cb0.cnp_off = 0x1840;
   m_cb0.cnp = &s_maxler_dbg;
   return 1;
 }
 // pascal - sm60, sm61, sm62
 if ( m_sm >= 0x3c && m_sm <= 0x3e ) {
   m_cb0.bank0 = &s_pascal;
   m_cb0.cnp_off = 0x1840;
   m_cb0.cnp = &s_maxler_dbg;
   return 1;
 }
 m_cb0.cnp_off = 0x1860;
 m_cb0.cnp = &s_volta_dbg;
 // volta - sm70, sm72
 if ( m_sm >= 0x46 && m_sm <= 0x48 ) {
   m_cb0.bank0 = &s_volta;
   return 1;
 }
 // turing sm75
 if ( m_sm == 0x4b ) {
   m_cb0.bank0 = &s_turing;
   return 1;
 }
 // ampere - sm80, sm86 & sm87
 if ( m_sm >= 0x50 && m_sm <= 0x57 ) {
   m_cb0.bank0 = &s_ampere;
   return 1;
 }
 // ada - sm89
 if ( m_sm == 0x59 ) {
   m_cb0.bank0 = &s_ada;
   return 1;
 }
 // hopper - sm90
 if ( m_sm == 0x5a ) {
   m_cb0.bank0 = &s_hopper;
   return 1;
 }
 // sm100+
 if ( m_sm >= 0x64 ) {
   m_cb0.bank0 = &s_sm100;
   return 1;
 }
 return 0;
}