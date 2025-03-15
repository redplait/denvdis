#include <array>
#include <unordered_map>
#include <string_view>

#define NV_MASK(name, size) static const std::array<std::pair<short, short>, size> name
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
 const std::unordered_map<int, const char *> *em;
};

// disasm interface
struct NV_disasm {
  virtual uint64_t extract_value(const std::pair<short, short> *start, const std::pair<short, short> *end) = 0;
  virtual bool match(const char *mask) = 0;
};

typedef int (*nv_filter)(NV_disasm *);

struct nv_instr {
 const char *name;
 const char *mask;
 int n; // number for formatting
 int line;
 int alt;
 int meaning_bits;
 const std::unordered_map<std::string_view, const nv_vattr> vas;
 const std::unordered_map<std::string_view, const nv_eattr *> eas;
};