# replace strange shift right of r0 with ced - see script in 1.ced
a.out: cft.cu
	nvcc -no-compress -arch sm_75 $^
	-../fb/fb -i 1 -o 12 ./a.out
	-../test/ced -v 12 md.ced
	-../fb/fb -i 1 -r 12 ./a.out

a53: cft.cu
	nvcc -o $@ -arch sm_53 $^
	-../fb/fb -i 1 -o 53 ./a53
	-../test/ced -v 53 53.ced
	../fb/fb -i 1 -r 53 ./a53

a3: cft.cu
	nvcc -o $@ -arch sm_30 $^
	-../fb/fb -i 1 -o 3 ./a3
	-../test/ced -v 3 3.ced

a32: cft.cu
	nvcc -o $@ -arch sm_32 $^
