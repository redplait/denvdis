CFLAGS = -std=c++20 -gdwarf-4

tp: test.cc libparse.a
	gcc -o $@ $(CFLAGS) test.cc -L . -lparse -lstdc++

ptx_parser.o: ptx_parser.cc ops.inc
	gcc -c -o $@ -fPIC $(CFLAGS) ptx_parser.cc

libparse.a: ptx_parser.o
	ar rcs $@ ptx_parser.o

test: tp
	@echo " @!p cp.async.cg.shared.global.L2::64B  [%r2],  [%r3], 16;\ntex.1d.v4.s32.f32  {r1,r2,r3,r4}, [tex_a,smpl_x,{f1}];" | ./tp -t
