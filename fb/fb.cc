#include <stdio.h>
#include <unistd.h>
#include "elfio/elfio.hpp"
#include <unordered_map>

static const char hexes[] = "0123456789ABCDEF";

void HexDump(FILE *f, const unsigned char *From, int Len)
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
          fprintf(f, "%s\n", buffer);
     }
     fprintf(f, "\n");
}


using namespace ELFIO;

// ripped from https://github.com/chei90/RemoteRendering/blob/master/inc/fatBinaryCtl.h
#define FATBINC_MAGIC   0x466243B1
typedef struct {
  int magic;
  int version;
  const unsigned long long* data;
  void *filename_or_fatbins;  /* version 1: offline filename,
                               * version 2: array of prelinked fatbins */
} __fatBinC_Wrapper_t;

#define FATBIN_CONTROL_SECTION_NAME     ".nvFatBinSegment"
/*
 * The section that contains the fatbin data itself
 * (put in separate section so easy to find)
 */
#define FATBIN_DATA_SECTION_NAME        ".nv_fatbin"

// from https://github.com/chei90/RemoteRendering/blob/master/inc/fatbinary.h
struct fatBinaryHeader
{
  unsigned int           magic;
  unsigned short         version;
  unsigned short         headerSize;
  unsigned long long int fatSize;
} __attribute__ ((aligned (8)));

#define FATBIN_MAGIC 0xBA55ED50

typedef enum {
  FATBIN_KIND_PTX      = 0x0001,
  FATBIN_KIND_ELF      = 0x0002,
  FATBIN_KIND_OLDCUBIN = 0x0004,
} fatBinaryCodeKind;

/* Flags */
#define FATBIN_FLAG_64BIT     0x0000000000000001LL
#define FATBIN_FLAG_DEBUG     0x0000000000000002LL
#define FATBIN_FLAG_CUDA      0x0000000000000004LL
#define FATBIN_FLAG_OPENCL    0x0000000000000008LL
#define FATBIN_FLAG_LINUX     0x0000000000000010LL
#define FATBIN_FLAG_MAC       0x0000000000000020LL
#define FATBIN_FLAG_WINDOWS   0x0000000000000040LL
#define FATBIN_FLAG_HOST_MASK 0x00000000000000f0LL
#define FATBIN_FLAG_OPT_MASK  0x0000000000000f00LL /* optimization level */
#define FATBIN_FLAG_COMPRESS  0x0000000000001000LL
#define FATBIN_FLAG_COMPRESS2 0x0000000000002000LL

struct  __attribute__((__packed__)) fat_text_header
{
    uint16_t kind;
    uint16_t unknown1;
    uint32_t header_size;
    uint64_t size;
    uint32_t compressed_size;       // Size of compressed data
    uint32_t unknown2;              // Address size for PTX?
    uint16_t minor;
    uint16_t major;
    uint32_t arch;
    uint32_t obj_name_offset;
    uint32_t obj_name_len;
    uint64_t flags;
    uint64_t zero;                  // Alignment for compression?
    uint64_t decompressed_size;
};

class CFatBin {
 public:
   // returns non-zero when succeed
   int open(const char *, int opt_h, int opt_v);
   // extract file at index idx to file of
   int extract(int idx, const char *of);
   // try to replace file at index idx to file rf
   int try_replace(int idx, const char *rf);
 protected:
   typedef std::unordered_map<int, std::pair<ptrdiff_t, fat_text_header> > FBItems;
   FBItems m_map;
   int _extract(const FBItems::iterator &, const char *, FILE *);
   inline bool compressed(const fat_text_header &ft) const {
     return ft.flags & FATBIN_FLAG_COMPRESS || ft.flags & FATBIN_FLAG_COMPRESS2;
   }
   // from https://zhuanlan.zhihu.com/p/29424681490
   size_t decompress(const uint8_t *input, size_t input_size, uint8_t *output, size_t output_size);
   Elf_Half n_sec = 0, m_ctrl = 0, m_fb = 0;
   unsigned long fb_size;
   elfio reader;
};

size_t CFatBin::decompress(const uint8_t *input, size_t input_size, uint8_t *output, size_t output_size)
{
    size_t ipos = 0, opos = 0;
    uint64_t next_nclen;  // length of next non-compressed segment
    uint64_t next_clen;   // length of next compressed segment
    uint64_t back_offset; // negative offset where redudant data is located, relative to current opos

    while (ipos < input_size) {
        next_nclen = (input[ipos] & 0xf0) >> 4;
        next_clen = 4 + (input[ipos] & 0xf);
        if (next_nclen == 0xf) {
            do {
                next_nclen += input[++ipos];
            } while (input[ipos] == 0xff);
        }

        if (memcpy(output + opos, input + (++ipos), next_nclen) == NULL) {
            fprintf(stderr, "Error copying data");
            return 0;
        }
#ifdef FATBIN_DECOMPRESS_DEBUG
        printf("%#04zx/%#04zx nocompress (len:%#zx):\n", opos, ipos, next_nclen);
        HexDump(stdout, output + opos, next_nclen);
#endif
        ipos += next_nclen;
        opos += next_nclen;
        if (ipos >= input_size || opos >= output_size) {
            break;
        }
        back_offset = input[ipos] + (input[ipos + 1] << 8);
        ipos += 2;
        if (next_clen == 0xf + 4) {
            do {
                next_clen += input[ipos++];
            } while (input[ipos - 1] == 0xff);
        }
#ifdef FATBIN_DECOMPRESS_DEBUG
        printf("%#04zx/%#04zx compress (decompressed len: %#zx, back_offset %#zx):\n", opos, ipos, next_clen,
               back_offset);
#endif
        if (next_clen <= back_offset) {
            if (memcpy(output + opos, output + opos - back_offset, next_clen) == NULL) {
                fprintf(stderr, "Error copying data");
                return 0;
            }
        } else {
            if (memcpy(output + opos, output + opos - back_offset, back_offset) == NULL) {
                fprintf(stderr, "Error copying data");
                return 0;
            }
            for (size_t i = back_offset; i < next_clen; i++) {
                output[opos + i] = output[opos + i - back_offset];
            }
        }
#ifdef FATBIN_DECOMPRESS_DEBUG
        HexDump(stdout, output + opos, next_clen);
#endif
        opos += next_clen;
    }
    return opos;
}

int CFatBin::open(const char *fn, int opt_h, int opt_v)
{
  if ( !reader.load(fn) ) {
    fprintf(stderr, "cannot open %s\n", fn);
    return 0;
  }
  // try to find control section
  n_sec = reader.sections.size();
  for ( Elf_Half i = 0; i < n_sec; ++i ) {
    section *sec = reader.sections[i];
    auto st = sec->get_type();
    if ( st == SHT_NOBITS || !sec->get_size() ) continue;
    auto sn = sec->get_name();
    if ( sn == FATBIN_CONTROL_SECTION_NAME ) {
      m_ctrl = i;
      if ( sec->get_size() < sizeof(__fatBinC_Wrapper_t) ) {
        fprintf(stderr, "control section is too small: %lX\n", sec->get_size());
        return 0;
      }
      break;
    }
  }
  if ( !m_ctrl ) {
    fprintf(stderr, "cannot find control section\n");
    return 0;
  }
  section *sec = reader.sections[m_ctrl];
  auto fbc = (const __fatBinC_Wrapper_t *)sec->get_data();
  if ( fbc->magic != FATBINC_MAGIC ) {
    fprintf(stderr, "invalid ctrl section magic %X\n", fbc->magic);
    return 0;
  }
  printf("version %d off %p\n", fbc->version, fbc->data);
  // try to find section at address fbc->data
  for ( Elf_Half i = 0; i < n_sec; ++i ) {
    section *sec = reader.sections[i];
    auto st = sec->get_type();
    if ( st == SHT_NOBITS || !sec->get_size() ) continue;
    auto sa = sec->get_address();
    if ( sa == (Elf64_Addr)fbc->data ||
         ((sa < (Elf64_Addr)fbc->data) && (sa + sec->get_size() > (Elf64_Addr)fbc->data))
       )
    {
      m_fb = i;
      printf("fatbin section %d size %lX %s\n", i, sec->get_size(), sec->get_name().c_str());
      if ( sec->get_size() < sizeof(fatBinaryHeader) ) {
        fprintf(stderr, "fatbim section is too small: %lX\n", sec->get_size());
        return 0;
      }
      break;
    }
  }
  if ( !m_fb ) {
    fprintf(stderr, "cannot find fatbin section\n");
    return 0;
  }
  // read fatBinaryHeader
  sec = reader.sections[m_fb];
  fb_size = sec->get_size();
  auto data = sec->get_data();
  auto dend = data + fb_size;
  auto fb_hdr = (const fatBinaryHeader *)data;
  int idx = 0;
  while ( (const char *)fb_hdr < dend ) {
    if ( opt_h )
      HexDump(stdout, (const unsigned char *)fb_hdr, sizeof(*fb_hdr));
    if ( fb_hdr->magic != FATBIN_MAGIC ) {
      fprintf(stderr, "unknown magic %X\n", fb_hdr->magic);
      return 0;
    }
    printf("version %d hdr_size %X fat_size %lX\n", fb_hdr->version, fb_hdr->headerSize, fb_hdr->fatSize);
    if ( fb_hdr->version != 1 || fb_hdr->headerSize != sizeof(fatBinaryHeader) ) {
      fprintf(stderr, "don't know sich fatbin header\n");
      return 0;
    }
    // try to parse adjacent fat_text_header
    if ( fb_size < fb_hdr->headerSize + sizeof(fat_text_header) ) {
      fprintf(stderr, "too short fatbin section, len %lX\n", fb_size);
      return 0;
    }
    const char *next_fb = (const char *)(fb_hdr + 1) + fb_hdr->fatSize;
    const fat_text_header *fth = (const fat_text_header *)((const char *)fb_hdr + fb_hdr->headerSize);
    while ( (const char *)(fth + 1) < next_fb )
    {
      if ( fth->header_size != sizeof(fat_text_header) ) break;
      if ( opt_v ) printf("off %p\n", sec->get_address() + (const char *)fth - data);
      if ( opt_h )
        HexDump(stdout, (const unsigned char *)fth, sizeof(*fth));
      // keep all fth data in single line for easy grepping
      printf("[%d] kind %X flag %lX header_size %X size %lX arch %X major %d minor %d",
        idx, fth->kind, fth->flags, fth->header_size,
        fth->size, fth->arch, fth->major, fth->minor
      );
      if ( fth->obj_name_offset || fth->obj_name_len )
        printf(" name_off %X mame_len %X", fth->obj_name_offset, fth->obj_name_len);
      if ( compressed(*fth) ) {
        printf(" compressed %X decompressed %lX", fth->compressed_size, fth->decompressed_size);
        if ( fth->zero ) printf(" zero %lX", fth->zero);
      // break;
      }
      putc('\n', stdout);
      m_map[idx] = { (const char *)fth - data, *fth };
      idx++;
      fth = (const fat_text_header *)((const char *)(fth + 1) + fth->size);
    }
    fb_hdr = (const fatBinaryHeader *)next_fb;
    if ( opt_v )
      printf("next %p\n", sec->get_address() + (const char *)fb_hdr - data);
  }
  return 1;
}

int CFatBin::extract(int idx, const char *of)
{
  // check idx
  auto ii = m_map.find(idx);
  if ( ii == m_map.end() ) {
    fprintf(stderr, "invalid index %d\n", idx);
    return 0;
  }
  // try open of
  FILE *ofp = fopen(of, "wb");
  if ( !ofp ) {
    fprintf(stderr, "cannot open %s, error %d (%s)\n", of, errno, strerror(errno));
    return 0;
  }
  int res = _extract(ii, of, ofp);
  fclose(ofp);
  return res;
}

int CFatBin::_extract(const FBItems::iterator &ii, const char *of, FILE *ofp)
{
  // seek to item in ii
  auto sec = reader.sections[m_fb];
  auto data = sec->get_data() + ii->second.first + ii->second.second.header_size;
  if ( compressed(ii->second.second) ) {
    uint8_t *out_buf = (uint8_t *)malloc(ii->second.second.decompressed_size);
    if ( !out_buf ) {
      fprintf(stderr, "cannot alloc %lX bytes for decompressed buffer\n", ii->second.second.decompressed_size);
      return 0;
    }
    int res = decompress((const uint8_t*)data, ii->second.second.compressed_size, out_buf, ii->second.second.decompressed_size);
    if ( !res ) {
      fprintf(stderr, "cannot decompress\n");
      free(out_buf);
      return 0;
    }
    if ( 1 != fwrite(out_buf, ii->second.second.decompressed_size, 1, ofp) ) {
      fprintf(stderr, "fwrite decompressed failed, error %d (%s)\n", errno, strerror(errno));
      free(out_buf);
      return 0;
    }
    free(out_buf);
  } else {
    if ( 1 != fwrite(data, ii->second.second.size, 1, ofp) ) {
      fprintf(stderr, "fwrite failed, error %d (%s)\n", errno, strerror(errno));
      return 0;
    }
  }
  return 1;
}

void usage(const char *prog)
{
  printf("%s usage: [options] fatbin", prog);
  printf("Options:\n");
  printf("-h - hex dump\n");
  printf("-i - index of entry\n");
  printf("-o - output file name\n");
  printf("-r - input file name to replace for some index\n");
  printf("-v - verbose mode\n");
  exit(6);
}

int main(int argc, char **argv) {
  int c, opt_h = 0, opt_v = 0, idx = -1;
  const char *o_fname = nullptr,
   *r_fname = nullptr;
  while(1) {
    c = getopt(argc, argv, "hvi:o:r:");
    if ( c == -1 ) break;
    switch(c) {
      case 'h': opt_h = 1; break;
      case 'v': opt_v = 1; break;
      case 'i': idx = atoi(optarg);
       break;
      case 'o': o_fname = optarg; break;
      case 'r': r_fname = optarg; break;
      default: usage(argv[0]);
    }
  }
  if ( argc == optind ) {
    usage(argv[0]);
    return 6;
  }
  CFatBin fb;
  if ( !fb.open(argv[optind], opt_h, opt_v) ) return 2;
  if ( idx == -1 ) return 0; // no index
  if ( o_fname && r_fname ) {
    printf("you cannot use both -o & -r options\n");
    return 6;
  }
  if ( o_fname )
    return fb.extract(idx, o_fname);
}