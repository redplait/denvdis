NVidia sass disassembler. See chaotic and incomplete documentation in my [blog](https://redplait.blogspot.com/search/label/cuda)

### Dependencies
* [ELFIO](https://github.com/serge1/ELFIO) for ELF files parsing
* [FP16](https://github.com/Maratyszcza/FP16) for rendering fp16 values

I used gcc built-in __uint128_t type, so probably on other platforms it's better to use something like abseil [Numeric](https://github.com/abseil/abseil-cpp/tree/master/absl/numeric)
