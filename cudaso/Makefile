ELFIO_PATH=../../../../../ELFIO/
UDIS_PATH=/home/redp/disc/udis86
DE_BG_OBJS = bm_search.o decuda_base.o de_bg.o x64arch.o rtmem.o mylog.o
AR_OBJS = $(DE_BG_OBJS) de_cupti.o decuda.o
LIBS=-L $(UDIS_PATH)/libudis86 -ludis86 -lstdc++ -ldl

.cc.o:
	gcc -g -fpic -c -I$(UDIS_PATH) -I$(ELFIO_PATH) $^

test: test.cc libdis.so
	gcc -g -I$(ELFIO_PATH) -o $@ test.cc ./libdis.so -lstdc++

libde_bg.a: $(DE_BG_OBJS)
	ar rcs $@ $(AR_OBJS)

libdis.a: $(AR_OBJS)
	ar rcs $@ $(AR_OBJS)

libdis.so: $(AR_OBJS)
	gcc -shared -o $@ $(AR_OBJS) $(LIBS)