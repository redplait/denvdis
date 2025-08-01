IFLAGS=-I ../scripts
ELFIO=-I ../../../../../ELFIO/
# path to fp16 (https://github.com/Maratyszcza/FP16)
FP16=-I ../../FP16/include
# for release add -Os and remove -g
SM_FLAGS=-Wall -shared -fpic -msse2avx -gdwarf-4 -pthread $(IFLAGS)
SM_LIBS=-lstdc++ -lpthread
# ead.pl options
PERL_OPTS=-d:Confess
EAD=../scripts/ead.pl
EAD_OPTS=-BFEmgrizp

all: ina nvd pa ced

nv_rend.o: nv_rend.cc
	gcc -Wall -std=c++20 -gdwarf-4 $(IFLAGS) $(FP16) -c $^

sass_parser.o: sass_parser.cc
	gcc -Wall -std=c++20 -gdwarf-4 $(IFLAGS) $(FP16) -c $^

ina: nv_rend.o ina.cc
	gcc -Wall -std=c++20 -gdwarf-4 $(IFLAGS) $(FP16) -o $@ $^ -lreadline -lstdc++ -ldl

nvd: nv_rend.o nvd.cc
	gcc -Wall -std=c++20 -g $(IFLAGS) $(ELFIO) $(FP16) -o $@ $^ -lstdc++ -ldl

pa: nv_rend.o pa.cc sass_parser.o
	gcc -Wall -std=c++20 -O -msse4 -mavx2 -gdwarf-4 $(IFLAGS) $(FP16) -o $@ $^ -lstdc++ -ldl

ced: nv_rend.o ced.cc sass_parser.o
	gcc -Wall -std=c++20 -gdwarf-4 $(IFLAGS) $(ELFIO) $(FP16) -o $@ $^ -lstdc++ -ldl

sm120.so: sm120.cc
	gcc $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm101.so: sm101.cc
	gcc $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm100.so: sm100.cc
	gcc $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm90.so: sm90.cc
	gcc $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm89.so: sm89.cc
	gcc $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm86.so: sm86.cc
	gcc $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm80.so: sm80.cc
	gcc $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm75.so: sm75.cc
	gcc $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm72.so: sm72.cc
	gcc $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm70.so: sm70.cc
	gcc $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm57.so: sm57.cc
	gcc $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm55.so: sm55.cc
	gcc $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm52.so: sm52.cc
	gcc $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm5.so: sm5.cc
	gcc $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm4.so: sm4.cc
	gcc $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm3.so: sm3.cc
	gcc $(SM_FLAGS) -o $@ $^ $(SM_LIBS)

sm3.cc: $(EAD)
	perl $(PERL_OPTS) $(EAD) $(EAD_OPTS) -C sm3 ../data/sm3_1.txt

sm4.cc: $(EAD)
	perl $(PERL_OPTS) $(EAD) $(EAD_OPTS) -C sm4 ../data/sm4_1.txt

sm5.cc: $(EAD)
	perl $(PERL_OPTS) $(EAD) $(EAD_OPTS) -C sm5 ../data/sm5_1.txt

sm52.cc: $(EAD)
	perl $(PERL_OPTS) $(EAD) $(EAD_OPTS) -C sm52 ../data/sm52_1.txt

sm55.cc: $(EAD)
	perl $(PERL_OPTS) $(EAD) $(EAD_OPTS) -C sm55 ../data/sm55_1.txt

sm57.cc: $(EAD)
	perl $(PERL_OPTS) $(EAD) $(EAD_OPTS) -C sm57 ../data/sm57_1.txt

sm70.cc: $(EAD)
	perl $(PERL_OPTS) $(EAD) $(EAD_OPTS) -C sm70 ../data/sm70_1.txt

sm72.cc: $(EAD)
	perl $(PERL_OPTS) $(EAD) $(EAD_OPTS) -C sm72 ../data/sm72_1.txt

sm75.cc: $(EAD)
	perl $(PERL_OPTS) $(EAD) $(EAD_OPTS) -C sm75 ../data/sm75_1.txt

# in data12
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

sm120.cc: $(EAD) 90.props
	perl $(PERL_OPTS) $(EAD) $(EAD_OPTS) -U 90.props -C sm120 ../data12/sm120_1.txt
