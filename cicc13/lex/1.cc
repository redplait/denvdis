#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <thread>
#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

typedef std::unordered_set<int> Visited;
typedef std::vector<std::thread *> Pool;

#define ROOT_IDX	2
#define MAX_LEN		32

struct yy_trans_info
{
  unsigned char yy_verify;
  int64_t yy_nxt;
} shit[] = {
#include "yy_state_list"
};
auto upto = sizeof(shit) / sizeof(shit[0]);

std::unordered_map<int, std::vector<std::pair<int, char> > > dist;
std::unordered_map<int, std::vector<int> > cache;
std::mutex glock;

void dump_res(int idx, std::string &res) {
  std::reverse(res.begin(), res.end());
  std::lock_guard<std::mutex> tmp(glock);
  printf("%d: %s\n", idx, res.c_str());
}

void find_state(int idx, std::vector<int> &res) {
  auto di = cache.find(idx);
  if ( di == cache.end() ) {
    printf("find_state %d failed\n", idx);
    return;
  }
  res.insert(res.end(), di->second.begin(), di->second.end());
}

void dump_res(const std::vector<int> &res) {
  for ( int r: res ) printf("> %d\n", r);
}

int rec_chain(int lvl, int old, std::string &prev, Visited &vis) {
  if ( lvl > MAX_LEN ) return 0;
  auto di = dist.find(old);
  if ( di == dist.end() ) return 0;
//  {
//    std::lock_guard<std::mutex> tmp(glock);
//    printf("rec_chain(%d, %d, %s, %d)\n", lvl, old, prev.c_str(), di->second.size());
//  }
  for ( auto pair: di->second ) {
    auto vi = vis.find(pair.first);
    if ( vi != vis.end() ) continue; // skip already visited to avoid loops
    if ( pair.first < ROOT_IDX ) continue;
    if ( ROOT_IDX == pair.first ) {
      auto res = prev + char(pair.second);
      dump_res(old, res);
      return 1;
    }
    std::string ns = prev + char(pair.second);
    Visited vnext = vis;
    vnext.insert(pair.first);
    if ( rec_chain(1 + lvl, pair.first, ns, vnext) ) return 1;
  }
  return 0;
}

void try_state(int idx, Pool &pool) {
  auto di = dist.find(idx);
  if ( di == dist.end() ) return;
printf("idx %d -> %d\n", idx, di->second.size());
  auto cloj = [di, idx]() {
    for ( auto start: di->second ) {
       std::string s; s.push_back(start.second);
       Visited vis;
       vis.insert(idx);
       vis.insert(start.first);
       rec_chain(0, start.first, s, vis);
     }
  };
  if ( 1 == di->second.size() ) {
    cloj();
  } else {
    // create new thread to process start
    auto *t = new std::thread(cloj);
    pool.push_back(t);
  }
}

int main(int argc, char **argv) {
//  int64_t test = 0xFFFFD8D4;
//  printf("%d\n", test - 0x100000000);
  // fix negatives
  for ( auto &neg: shit ) {
    if ( neg.yy_nxt > 0xf0000000 ) {
// printf("%X %d\n", neg.yy_nxt, neg.yy_nxt - 0x100000000);
      neg.yy_nxt -= 0x100000000;
    }
  }
  // form cache & back distance map
  int idx = 0;
  for ( auto &neg: shit ) {
    if ( neg.yy_nxt > 0 ) cache[neg.yy_nxt].push_back(idx);
    // to speed-up brute-force I use only symbols 32 .. 127
    if ( upto - idx > 128 ) {
      for ( int l = 32; l < 128; l++ ) {
        if ( shit[idx + l].yy_verify != l ) continue;
        auto d = idx + shit[idx + l].yy_nxt;
        dist[d].push_back({ idx, l });
      }
    }
    ++idx;
  }
  for ( int i = 1; i < argc; i++ ) {
     int idx = atoi(argv[i]);
 printf("try %d\n", idx);
     std::vector<int> res;
     find_state(idx, res);
//     dump_res(res);
     if ( res.empty() ) continue;
     Pool pool;
     for ( int r: res ) {
       try_state(1+r, pool);
     }
     if ( !pool.empty() ) {
       printf("%zu threads\n", pool.size());
       // wait
       for ( auto t: pool ) t->join();
       for ( auto t: pool ) delete t;
     }
  }
}