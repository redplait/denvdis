ELFIO=-I ../../../../../ELFIO/

fb: fb.cc
	gcc $(ELFIO) -Wall -std=c++20 -o $@ $^ -lstdc++

test: fb
	./fb ../../../src/cuda_latency_benchmark/cuda_task_queue.cpython-38-x86_64-linux-gnu.so

test2: fb
	./fb ../../../7/libcublas.so.7.5.18



