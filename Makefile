denv: denv.cc
	gcc -o $@ -I ../../../../ELFIO denv.cc -lstdc++

denv12: denv12.cc
	gcc -o $@ -I ../../../../ELFIO -I ../lz4/lib denv12.cc ../lz4/lib/liblz4.a -lstdc++
