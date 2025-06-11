NVidia sass disassembler. See chaotic and incomplete documentation in my [blog](https://redplait.blogspot.com/search/label/cuda)

### Dependencies
* [ELFIO](https://github.com/serge1/ELFIO) for ELF files parsing
* [FP16](https://github.com/Maratyszcza/FP16) for rendering fp16 values

I used gcc built-in __uint128_t type, so probably on other platforms it's better to use something like abseil [Numeric](https://github.com/abseil/abseil-cpp/tree/master/absl/numeric)

As an illustration sub-dir [test](https://github.com/redplait/denvdis/tree/master/test) contains 3 small applications:
### nvd
sass disassembler - of course not a complete replica of nvdisasm, instead it can show
 * values of all encoding fields with -O option
 * [scheduling tables](https://redplait.blogspot.com/2025/05/nvidia-sass-latency-tables.html) with -S option
 * [predicates](https://redplait.blogspot.com/2025/04/nvidia-sass-disassembler-part-6.html) of instructions with -p option

### ina
interactive sass [assembler](https://redplait.blogspot.com/2025/05/nvidia-sass-assembler.html)

### pa
parser of nvdisasm output
