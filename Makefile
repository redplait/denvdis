denv: denv.cc
	gcc -o $@ -I ../../../../ELFIO denv.cc -lstdc++
