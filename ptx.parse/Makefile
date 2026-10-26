CFLAGS = -std=c++20 -gdwarf-4
# from https://stackoverflow.com/questions/41708580/use-gcc-preprocessor-to-define-filename-for-include
OLD_PTX=-DMaskWIDTH=17 -DOPS_INC="\"ops.inc\""
PTX134=-DMaskWIDTH=19 -DOPS_INC="\"ops134.inc\""

tp: test.cc libparse.a
	gcc -o $@ $(CFLAGS) $(OLD_PTX) test.cc -L . -lparse -lstdc++

ptx_parser.o: ptx_parser.cc ops.inc
	gcc -c -o $@ -fPIC $(CFLAGS) $(OLD_PTX) ptx_parser.cc

libparse.a: ptx_parser.o
	ar rcs $@ ptx_parser.o

tp134: test.cc ptx_parser.cc ops134.inc
	gcc -o $@ -fPIC $(CFLAGS) $(PTX134) ptx_parser.cc test.cc -lstdc++

test: tp
	@echo " @!p cp.async.cg.shared.global.L2::64B  [%r2],  [%r3], 16;\ntex.1d.v4.s32.f32  {r1,r2,r3,r4}, [tex_a,smpl_x,{f1}];" | ./tp -t
