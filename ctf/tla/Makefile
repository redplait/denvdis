a.out: tla2.cu ../../scripts/dg.pl
	nvcc -g -no-compress -arch sm_75 tla2.cu -lcudart
	perl ../../fb/fb.pl -i 1 a.out
	perl ../../scripts/dg.pl -vgrtUPC 1.cfg 1_75.cubin

diff: 1_75.cubin
	../../../../bin/nvdisasm 1_75.cubin > 2
	-diff 1 2