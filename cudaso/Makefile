ELFIO_PATH=../../../../../ELFIO/
UDIS_PATH=/home/redp/disc/udis86
CEREAL_PATH=cereal/include
DE_BG_OBJS = bm_search.o decuda_base.o de_bg.o x64arch.o rtmem.o mylog.o
AR_OBJS = $(DE_BG_OBJS) de_cupti.o decuda.o de_ptx.o
LIBS=-L $(UDIS_PATH)/libudis86 -ludis86 -lstdc++ -ldl

.cc.o:
	gcc -std=c++20 -g -fpic -c -I$(CEREAL_PATH) -I$(UDIS_PATH) -I$(ELFIO_PATH) $^

test: test.cc libdis.so
	gcc -std=c++20 -g -I$(ELFIO_PATH) -I$(CEREAL_PATH) -o $@ test.cc ./libdis.so -lstdc++

libde_bg.a: $(DE_BG_OBJS)
	ar rcs $@ $(AR_OBJS)

libdis.a: $(AR_OBJS)
	ar rcs $@ $(AR_OBJS)

my_hook.o: my_hook.cc
	gcc -std=c++20 -g -fpic -c -I$(ELFIO_PATH) -I$(CEREAL_PATH) $^

libdis.so: $(AR_OBJS) my_hook.o
	gcc -shared -o $@ my_hook.o $(AR_OBJS) $(LIBS) -lpthread