#include <functional>
#include <unordered_map>
#include <list>
#include <string_view>

#define NV_MASK(name, size) static const std::pair<short, short> name[size]
#define NV_ENUM(name)  static const std::unordered_map<int, const char *> name
#define NV_TAB(name)   static const std::unordered_map<int, const int *> name

enum NV_Format {
  NV_BITSET,
  NV_UImm,
  NV_SImm,
  NV_SSImm,
  NV_RSImm,
  NV_F64Imm,
  NV_F16Imm,
  NV_F32Imm,
};

struct nv_vattr {
 enum NV_Format kind;
 bool has_ast; // final *
};

struct nv_eattr {
 bool ignore; // has initial /
 bool print;  // /PRINT
 bool has_def_value;
 int def_value;
 const char *ename;
 const std::unordered_map<int, const char *> *em;
};

// disasm interface
struct NV_disasm {
};

typedef int (*nv_filter)(std::function<uint64_t(const std::pair<short, short> *, size_t)> &);
typedef std::unordered_map<std::string_view, uint64_t> NV_extracted;
typedef void (*nv_extract)(std::function<uint64_t(const std::pair<short, short> *, size_t)> &, NV_extracted &);

struct nv_instr {
 const char *name;
 const char *mask;
 int n; // number for formatting
 int line;
 int alt;
 int meaning_bits;
 const std::unordered_map<std::string_view, const nv_vattr> vas;
 const std::unordered_map<std::string_view, const nv_eattr *> eas;
 nv_filter filter;
 nv_extract extract;
};

// binary tree
struct NV_bt_node {
  bool is_leaf;
  std::list<const nv_instr *> ins;
};

struct NV_non_leaf: public NV_bt_node
{
  int bit;
  const NV_bt_node *left, *right;
};