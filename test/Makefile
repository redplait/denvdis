IFLAGS=-I ../scripts
ELFIO=-I ../../../../../ELFIO/
CC=gcc -mavx2
# CLANG=gcc -mavx2
CLANG=clang++-21 -mavx2
# path to fp16 (https://github.com/Maratyszcza/FP16)
FP16=-I ../../FP16/include
# for release add -Os and remove -g
CFLAGS=-Wall -Wno-missing-braces -std=c++20 -gdwarf-4
# SM_FLAGS=-Wall -shared -fpic -msse2avx -gdwarf-4 -pthread $(IFLAGS)
SM_FLAGS=-Wall -Wno-missing-braces -Wno-unused-const-variable -shared -fpic -gdwarf-4 -pthread $(IFLAGS)
SM_LIBS=-lstdc++ -lpthread
# ead.pl options
PERL_OPTS=-d:Confess
EAD=../scripts/ead.pl
EAD_OPTS=-BFEmgarizp

all: ina nvd pa ced libced.a

AR_OBJS = bf16.o ced_base.o nv_rend.o sass_parser.o

libced.a: $(AR_OBJS)
	ar rcs $@ $(AR_OBJS)

bf16.o: bf16.cc
	$(CC) -fpic -fpic $(CFLAGS) -c $^

ced_base.o: ced_base.cc
	$(CC) -fpic $(CFLAGS) $(ELFIO) $(IFLAGS) $(FP16) -c $^

nv_rend.o: nv_rend.cc
	$(CC) -fpic $(CFLAGS) $(IFLAGS) $(FP16) -c $^

sass_parser.o: sass_parser.cc
	$(CC) -fpic $(CFLAGS) $(IFLAGS) $(FP16) -c $^

ina: ina.cc libced.a
	$(CC) $(CFLAGS) $(IFLAGS) $(FP16) -o $@ $^ -lreadline -lstdc++ -ldl

nvd: nvd.cc libced.a
	$(CC) $(CFLAGS) $(IFLAGS) $(ELFIO) $(FP16) -o $@ $^ -lstdc++ -ldl

pa: pa.cc libced.a
	$(CC) $(CFLAGS) -O -msse4 -mavx2 $(IFLAGS) $(FP16) -o $@ $^ -lstdc++ -ldl

ced: ced.cc libced.a
	$(CC) $(CFLAGS) $(IFLAGS) $(ELFIO) $(FP16) -o $@ $^ -lstdc++ -ldl

sm120.so: sm120.cc
	$(CLANG) $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm103.so: sm103.cc
	$(CLANG) $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm101.so: sm101.cc
	$(CLANG) $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm100.so: sm100.cc
	$(CLANG) $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm90.so: sm90.cc
	$(CLANG) $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm89.so: sm89.cc
	$(CLANG) $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm86.so: sm86.cc
	$(CLANG) $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm80.so: sm80.cc
	$(CLANG) $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm75.so: sm75.cc
	$(CLANG) $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm72.so: sm72.cc
	$(CLANG) $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm70.so: sm70.cc
	$(CLANG) $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm57.so: sm57.cc
	$(CLANG) $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm55.so: sm55.cc
	$(CLANG) $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm52.so: sm52.cc
	$(CLANG) $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm5.so: sm5.cc
	$(CLANG) $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm4.so: sm4.cc
	$(CLANG) $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm3.so: sm3.cc
	$(CLANG) $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm2.so: sm2.cc
	$(CLANG) $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm2.cc: $(EAD)
	perl $(PERL_OPTS) $(EAD) -BFEmrizp -C sm2 ../data/sm2_1.txt

sm3.cc: $(EAD)
	perl $(PERL_OPTS) $(EAD) $(EAD_OPTS) -C sm3 ../data/sm3_1.txt

sm4.cc: $(EAD)
	perl $(PERL_OPTS) $(EAD) $(EAD_OPTS) -C sm4 ../data/sm4_1.txt

sm5.cc: $(EAD)
	perl $(PERL_OPTS) $(EAD) $(EAD_OPTS) -C sm5 ../data/sm5_1.txt

# in data11
sm52.cc: $(EAD)
	perl $(PERL_OPTS) $(EAD) $(EAD_OPTS) -C sm52 ../data11/sm52_1.txt

sm55.cc: $(EAD)
	perl $(PERL_OPTS) $(EAD) $(EAD_OPTS) -C sm55 ../data11/sm55_1.txt

sm57.cc: $(EAD)
	perl $(PERL_OPTS) $(EAD) $(EAD_OPTS) -C sm57 ../data11/sm57_1.txt

sm70.cc: $(EAD)
	perl $(PERL_OPTS) $(EAD) $(EAD_OPTS) -C sm70 ../data11/sm70_1.txt

sm72.cc: $(EAD)
	perl $(PERL_OPTS) $(EAD) $(EAD_OPTS) -C sm72 ../data11/sm72_1.txt

sm75.cc: $(EAD)
	perl $(PERL_OPTS) $(EAD) $(EAD_OPTS) -C sm75 ../data11/sm75_1.txt

sm80.cc: $(EAD)
	perl $(PERL_OPTS) $(EAD) $(EAD_OPTS) -C sm80 ../data11/sm80_1.txt

sm86.cc: $(EAD)
	perl $(PERL_OPTS) $(EAD) $(EAD_OPTS) -C sm86 ../data11/sm86_1.txt

sm89.cc: $(EAD)
	perl $(PERL_OPTS) $(EAD) $(EAD_OPTS) -C sm89 ../data11/sm89_1.txt

sm90.cc 90.props: $(EAD)
	perl $(PERL_OPTS) $(EAD) $(EAD_OPTS) -u 90.props -C sm90 ../data11/sm90_1.txt

# try to apply properties from sm90 to sm100-sm120 from data12

sm100.cc: $(EAD) 90.props
	perl $(PERL_OPTS) $(EAD) $(EAD_OPTS) -U 90.props -C sm100 ../data12/sm100_1.txt

sm101.cc: $(EAD) 90.props
	perl $(PERL_OPTS) $(EAD) $(EAD_OPTS) -U 90.props -C sm101 ../data12/sm101_1.txt

sm103.cc: $(EAD) 90.props
	perl $(PERL_OPTS) $(EAD) $(EAD_OPTS) -U 90.props -C sm103 ../data12/sm103_1.txt

sm120.cc: $(EAD) 90.props
	perl $(PERL_OPTS) $(EAD) $(EAD_OPTS) -U 90.props -C sm120 ../data12/sm120_1.txt
