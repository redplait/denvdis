IFLAGS=-I ../scripts
ELFIO=-I ../../../../../ELFIO/
# for release add -Os and remove -g
SM_FLAGS=-shared -fpic -msse2avx -g $(IFLAGS)

nvd: nvd.cc
	gcc -g $(IFLAGS) $(ELFIO) -o $@ $^ -lstdc++ -ldl

sm120.so: sm120.cc
	gcc $(SM_FLAGS) -o $@ $^ -lstdc++

sm101.so: sm101.cc
	gcc $(SM_FLAGS) -o $@ $^ -lstdc++

sm100.so: sm100.cc
	gcc $(SM_FLAGS) -o $@ $^ -lstdc++

sm90.so: sm90.cc
	gcc $(SM_FLAGS) -o $@ $^ -lstdc++

sm89.so: sm89.cc
	gcc $(SM_FLAGS) -o $@ $^ -lstdc++

sm86.so: sm86.cc
	gcc $(SM_FLAGS) -o $@ $^ -lstdc++

sm80.so: sm80.cc
	gcc $(SM_FLAGS) -o $@ $^ -lstdc++

sm75.so: sm75.cc
	gcc $(SM_FLAGS) -o $@ $^ -lstdc++

sm72.so: sm72.cc
	gcc $(SM_FLAGS) -o $@ $^ -lstdc++

sm57.so: sm57.cc
	gcc $(SM_FLAGS) -o $@ $^ -lstdc++

sm55.so: sm55.cc
	gcc $(SM_FLAGS) -o $@ $^ -lstdc++

sm52.so: sm52.cc
	gcc $(SM_FLAGS) -o $@ $^ -lstdc++

sm5.so: sm5.cc
	gcc $(SM_FLAGS) -o $@ $^ -lstdc++

sm4.so: sm4.cc
	gcc $(SM_FLAGS) -o $@ $^ -lstdc++

sm3.so: sm3.cc
	gcc $(SM_FLAGS) -o $@ $^ -lstdc++

sm3.cc: ../scripts/ead.pl
	perl -d:Confess ../scripts/ead.pl -BFmriz -C sm3 ../data/sm3_1.txt

sm4.cc: ../scripts/ead.pl
	perl -d:Confess ../scripts/ead.pl -BFmriz -C sm4 ../data/sm4_1.txt

sm5.cc: ../scripts/ead.pl
	perl -d:Confess ../scripts/ead.pl -BFmriz -C sm5 ../data/sm5_1.txt

sm52.cc: ../scripts/ead.pl
	perl -d:Confess ../scripts/ead.pl -BFmriz -C sm52 ../data/sm52_1.txt

sm55.cc: ../scripts/ead.pl
	perl -d:Confess ../scripts/ead.pl -BFmriz -C sm55 ../data/sm55_1.txt

sm57.cc: ../scripts/ead.pl
	perl -d:Confess ../scripts/ead.pl -BFmriz -C sm57 ../data/sm57_1.txt

sm70.cc: ../scripts/ead.pl
	perl -d:Confess ../scripts/ead.pl -BFmriz -C sm70 ../data/sm70_1.txt

sm72.cc: ../scripts/ead.pl
	perl -d:Confess ../scripts/ead.pl -BFmriz -C sm72 ../data/sm72_1.txt

sm75.cc: ../scripts/ead.pl
	perl -d:Confess ../scripts/ead.pl -BFmriz -C sm75 ../data/sm75_1.txt

# in data12
sm80.cc: ../scripts/ead.pl
	perl -d:Confess ../scripts/ead.pl -BFmriz -C sm80 ../data12/sm80_1.txt

sm86.cc: ../scripts/ead.pl
	perl -d:Confess ../scripts/ead.pl -BFmriz -C sm86 ../data12/sm86_1.txt

sm89.cc: ../scripts/ead.pl
	perl -d:Confess ../scripts/ead.pl -BFmriz -C sm89 ../data12/sm89_1.txt

sm90.cc: ../scripts/ead.pl
	perl -d:Confess ../scripts/ead.pl -BFmriz -C sm90 ../data12/sm90_1.txt

sm100.cc: ../scripts/ead.pl
	perl -d:Confess ../scripts/ead.pl -BFmriz -C sm100 ../data12/sm100_1.txt

sm101.cc: ../scripts/ead.pl
	perl -d:Confess ../scripts/ead.pl -BFmriz -C sm101 ../data12/sm101_1.txt

sm120.cc: ../scripts/ead.pl
	perl -d:Confess ../scripts/ead.pl -BFmriz -C sm120 ../data12/sm120_1.txt
