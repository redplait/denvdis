ELFIO_PATH=../../../../../ELFIO/
UDIS_PATH=/home/redp/disc/udis86
AR_OBJS = bm_search.o decuda.o x64arch.o rtmem.o mylog.o
LIBS=-L $(UDIS_PATH)/libudis86 -ludis86 -lstdc++ -ldl

.cc.o:
	gcc -g -fpic -c -I$(UDIS_PATH) -I$(ELFIO_PATH) $^

test: test.cc libdis.so
	gcc -g -I$(ELFIO_PATH) -o $@ test.cc ./libdis.so -lstdc++

libdis.a: $(AR_OBJS)
	ar rcs $@ $(AR_OBJS)

libdis.so: $(AR_OBJS)
	gcc -shared -o $@ $(AR_OBJS) $(LIBS)