NVidia sass disassembler. See chaotic and incomplete documentation in my [blog](https://redplait.blogspot.com/search/label/cuda)

### Dependencies
* [ELFIO](https://github.com/serge1/ELFIO) for ELF files parsing
* [FP16](https://github.com/Maratyszcza/FP16) for rendering fp16 values

I used gcc built-in __uint128_t type, so probably on other platforms it's better to use something like abseil [Numeric](https://github.com/abseil/abseil-cpp/tree/master/absl/numeric)

As an illustration sub-dir [test](https://github.com/redplait/denvdis/tree/master/test) contains 4 small applications:
### ced
sed-like tool for inline patching of sass instructions within cubin files. See details in my [blog](https://redplait.blogspot.com/2025/07/ced-sed-like-cubin-editor.html)

### nvd
sass disassembler - of course not a complete replica of nvdisasm (however you can have syntax compatible with nvdisasm with -c option), instead it can show
 * values of all encoding fields with -O option
 * [scheduling tables](https://redplait.blogspot.com/2025/05/nvidia-sass-latency-tables.html) with -S option
 * [predicates](https://redplait.blogspot.com/2025/04/nvidia-sass-disassembler-part-6.html) of instructions with -p option
 * [registers tracking](https://redplait.blogspot.com/2025/07/sass-instructions-registers-tracking.html) with -T option
 * [LUT](https://forums.developer.nvidia.com/t/what-does-lop3-lut-mean-how-is-it-executed/227472) operations - see details in my [blog](https://redplait.blogspot.com/2025/07/sass-instructions-lut-operations.html)

Also note that original nvdisasm can produce [ambigious](https://redplait.blogspot.com/2025/06/curse-of-imad.html) output

### ina
interactive sass [assembler](https://redplait.blogspot.com/2025/05/nvidia-sass-assembler.html)

because the instruction can have many forms (for example LDG have 14 and F2FP - 60) I add filtering for forms selection, format is (+|-)letter where letter can be
 * f - floating point imm operand
 * i - integer imm operand
 * C - const bank
 * m - memory ref operand
 * d - desc ref operand
 * u - [uniform register](https://redplait.blogspot.com/2025/07/sass-instructions-uniform-registers.html)

You can reset filters with '!'

You can save your instructions to binary file with -o option

### pa
parser of nvdisasm output - some [details](https://redplait.blogspot.com/2025/06/nvdisasm-sass-parser.html)

You can run "strange loop" to consume output of nvd - for this later must use option *-c*

By default all 3 trying to load corresponding sm_xx.so from current directory - but you can peek those dir with env var SM_DIR
