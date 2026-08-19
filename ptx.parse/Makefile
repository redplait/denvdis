tp: ops.inc test.cc ptx_parser.cc
	gcc -o $@ -std=c++20 -gdwarf-4 test.cc ptx_parser.cc -lstdc++

test: tp
	@echo " @!p cp.async.cg.shared.global.L2::64B  [%r2],  [%r3], 16;" | ./tp -t
