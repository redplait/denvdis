ELFIO=-I ../../../../ELFIO

denv: denv.cc
	gcc -o $@ $(ELFIO) denv.cc -lstdc++

deptx: ptxas.cc
	gcc -g -o $@ $(ELFIO) ptxas.cc -lstdc++

deptx12: ptxas12.cc
	gcc -g -o $@ $(ELFIO) ptxas12.cc -lstdc++

cic12: cic12.cc
	gcc -g -o $@ $(ELFIO) cic12.cc -lstdc++

denv12: denv12.cc
	gcc -o $@ $(ELFIO) -I ../lz4/lib denv12.cc ../lz4/lib/liblz4.a -lstdc++
