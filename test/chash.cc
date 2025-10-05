#include "chash.h"
#include <iostream>
#include <fstream>
#include <openssl/evp.h>

static constexpr size_t BUFFER_SIZE = 1024 * 4;
static bool s_loaded = false;

template <typename T>
struct ssl_del {
  char buf[BUFFER_SIZE];
  T *what = nullptr;
  ssl_del(T *t) { what = t; }
  ~ssl_del();
};

// borrowed from https://stackoverflow.com/questions/71718818/c-openssl-hash-of-the-file-is-not-the-right-one
template <>
ssl_del<EVP_MD_CTX>::~ssl_del() {
  if ( what ) EVP_MD_CTX_free(what);
}

template <>
ssl_del<BIO>::~ssl_del() {
  if ( what ) BIO_free(what);
}

bool hash_file(const std::string &fname, const char *algo, std::vector<uint8_t> &md)
{
  if ( !s_loaded ) {
    OpenSSL_add_all_digests();
    s_loaded = true;
  }
  const EVP_MD *mthd = EVP_get_digestbyname(algo);
  if ( !mthd ) return false;
  std::ifstream inp(fname, std::ios::in | std::ios::binary);
  if ( !inp.is_open() ) return false;
  ssl_del ctx(EVP_MD_CTX_new());
  EVP_DigestInit_ex(ctx.what, mthd, nullptr);

  while (inp.read(ctx.buf, BUFFER_SIZE).gcount() > 0)
    EVP_DigestUpdate(ctx.what, ctx.buf, inp.gcount());
  // size output vector
  unsigned int mdlen = EVP_MD_size(mthd);
  md.resize(mdlen);

  // general final digest
  EVP_DigestFinal_ex(ctx.what, md.data(), &mdlen);
  return true;
}