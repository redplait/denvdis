tp: ops.inc test.cc ptx_parser.cc
	gcc -o $@ -std=c++20 -gdwarf-4 test.cc ptx_parser.cc -lstdc++

test: tp
	@echo " @!p cp.async.cg.shared.global.L2::64B  [%r2],  [%r3], 16;\ntex.1d.v4.s32.f32  {r1,r2,r3,r4}, [tex_a,smpl_x,{f1}];" | ./tp -t
