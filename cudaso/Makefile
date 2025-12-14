ELFIO_PATH=../../../../../ELFIO/
UDIS_PATH=/home/redp/disc/udis86
AR_OBJS = bm_search.o decuda.o

.cc.o:
	gcc -g -fpic -c -I$(UDIS_PATH) -I$(ELFIO_PATH) $^

test: test.cc libdis.a
	gcc -g -I$(ELFIO_PATH) -o $@ test.cc ./libdis.a -lstdc++

libdis.a: $(AR_OBJS)
	ar rcs $@ $(AR_OBJS)
