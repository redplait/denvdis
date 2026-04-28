a.out: tla2.cu ../../scripts/dg.pl 1.cfg
	nvcc -g -no-compress -arch sm_75 tla2.cu ../../cudaso/libdis.so /usr/local/lib64/libstdc++.so
	perl ../../fb/fb.pl -i 1 ./a.out
	perl -d:Confess ../../scripts/dg.pl -mldvbgrtpPC 1.cfg 1_75.cubin
	perl ../../fb/fb.pl -r 1_75.cubin -i 1 ./a.out

diff: 1_75.cubin
	../../../../bin/nvdisasm 1_75.cubin > 2
	-diff 1 2