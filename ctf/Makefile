# replace strange shift right of r0 with ced - see script in 1.ced
a.out: cft.cu
	nvcc -arch sm_75 cft.cu
	-../fb/fb -i 1 -o 12 ./a.out
	-../test/ced -v 12 1.ced
	-../fb/fb -i 1 -r 12 ./a.out
