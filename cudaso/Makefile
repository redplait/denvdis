ELFIO_PATH=../../../../../ELFIO/
UDIS_PATH=/home/redp/disc/udis86
CEREAL_PATH=cereal/include
DE_BG_OBJS = bm_search.o decuda_base.o de_bg.o x64arch.o rtmem.o mylog.o
MOD_OBJS = de_cupti.o de_ptx.o de_merc.o
AR_OBJS = $(DE_BG_OBJS) de_cupti.o decuda.o de_ptx.o de_merc.o
SO_OBJS = $(DE_BG_OBJS) decuda.o
LIBS=-L $(UDIS_PATH)/libudis86 -ludis86 -lstdc++ -ldl

.cc.o:
	gcc -std=c++20 -g -fpic -c -I$(CEREAL_PATH) -I$(UDIS_PATH) -I$(ELFIO_PATH) $^

test: test.cc libdis.so $(MOD_OBJS)
	gcc -std=c++20 -g -I$(ELFIO_PATH) -I$(CEREAL_PATH) -o $@ test.cc $(MOD_OBJS) ./libdis.so -lstdc++

dec_nvml: dec_nvml.cc
	gcc -o $@ $^

libde_bg.a: $(DE_BG_OBJS)
	ar rcs $@ $(AR_OBJS)

libdis.a: $(AR_OBJS)
	ar rcs $@ $(AR_OBJS)

my_hook.o: my_hook.cc
	gcc -std=c++20 -g -fpic -c -I$(ELFIO_PATH) -I$(CEREAL_PATH) $^

libdis.so: $(SO_OBJS) my_hook.o
	gcc -shared -o $@ my_hook.o $(SO_OBJS) $(LIBS) -lpthread