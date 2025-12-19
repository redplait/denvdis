ELFIO_PATH=../../../../../ELFIO/
UDIS_PATH=/home/redp/disc/udis86
AR_OBJS = bm_search.o decuda.o x64arch.o

.cc.o:
	gcc -g -fpic -c -I$(UDIS_PATH) -I$(ELFIO_PATH) $^

test: test.cc libdis.a
	gcc -g -I$(ELFIO_PATH) -o $@ test.cc -L $(UDIS_PATH)/libudis86 ./libdis.a -ludis86 -lstdc++

libdis.a: $(AR_OBJS)
	ar rcs $@ $(AR_OBJS)
