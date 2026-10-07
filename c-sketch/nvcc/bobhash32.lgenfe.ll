; ModuleID = 'bobhash32.lgenfe.bc'
source_filename = "moduleOutput"
target datalayout = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@prime32 = addrspace(4) global [1229 x i32] [i32 2, i32 3, i32 5, i32 7, i32 11, i32 13, i32 17, i32 19, i32 23, i32 29, i32 31, i32 37, i32 41, i32 43, i32 47, i32 53, i32 59, i32 61, i32 67, i32 71, i32 73, i32 79, i32 83, i32 89, i32 97, i32 101, i32 103, i32 107, i32 109, i32 113, i32 127, i32 131, i32 137, i32 139, i32 149, i32 151, i32 157, i32 163, i32 167, i32 173, i32 179, i32 181, i32 191, i32 193, i32 197, i32 199, i32 211, i32 223, i32 227, i32 229, i32 233, i32 239, i32 241, i32 251, i32 257, i32 263, i32 269, i32 271, i32 277, i32 281, i32 283, i32 293, i32 307, i32 311, i32 313, i32 317, i32 331, i32 337, i32 347, i32 349, i32 353, i32 359, i32 367, i32 373, i32 379, i32 383, i32 389, i32 397, i32 401, i32 409, i32 419, i32 421, i32 431, i32 433, i32 439, i32 443, i32 449, i32 457, i32 461, i32 463, i32 467, i32 479, i32 487, i32 491, i32 499, i32 503, i32 509, i32 521, i32 523, i32 541, i32 547, i32 557, i32 563, i32 569, i32 571, i32 577, i32 587, i32 593, i32 599, i32 601, i32 607, i32 613, i32 617, i32 619, i32 631, i32 641, i32 643, i32 647, i32 653, i32 659, i32 661, i32 673, i32 677, i32 683, i32 691, i32 701, i32 709, i32 719, i32 727, i32 733, i32 739, i32 743, i32 751, i32 757, i32 761, i32 769, i32 773, i32 787, i32 797, i32 809, i32 811, i32 821, i32 823, i32 827, i32 829, i32 839, i32 853, i32 857, i32 859, i32 863, i32 877, i32 881, i32 883, i32 887, i32 907, i32 911, i32 919, i32 929, i32 937, i32 941, i32 947, i32 953, i32 967, i32 971, i32 977, i32 983, i32 991, i32 997, i32 1009, i32 1013, i32 1019, i32 1021, i32 1031, i32 1033, i32 1039, i32 1049, i32 1051, i32 1061, i32 1063, i32 1069, i32 1087, i32 1091, i32 1093, i32 1097, i32 1103, i32 1109, i32 1117, i32 1123, i32 1129, i32 1151, i32 1153, i32 1163, i32 1171, i32 1181, i32 1187, i32 1193, i32 1201, i32 1213, i32 1217, i32 1223, i32 1229, i32 1231, i32 1237, i32 1249, i32 1259, i32 1277, i32 1279, i32 1283, i32 1289, i32 1291, i32 1297, i32 1301, i32 1303, i32 1307, i32 1319, i32 1321, i32 1327, i32 1361, i32 1367, i32 1373, i32 1381, i32 1399, i32 1409, i32 1423, i32 1427, i32 1429, i32 1433, i32 1439, i32 1447, i32 1451, i32 1453, i32 1459, i32 1471, i32 1481, i32 1483, i32 1487, i32 1489, i32 1493, i32 1499, i32 1511, i32 1523, i32 1531, i32 1543, i32 1549, i32 1553, i32 1559, i32 1567, i32 1571, i32 1579, i32 1583, i32 1597, i32 1601, i32 1607, i32 1609, i32 1613, i32 1619, i32 1621, i32 1627, i32 1637, i32 1657, i32 1663, i32 1667, i32 1669, i32 1693, i32 1697, i32 1699, i32 1709, i32 1721, i32 1723, i32 1733, i32 1741, i32 1747, i32 1753, i32 1759, i32 1777, i32 1783, i32 1787, i32 1789, i32 1801, i32 1811, i32 1823, i32 1831, i32 1847, i32 1861, i32 1867, i32 1871, i32 1873, i32 1877, i32 1879, i32 1889, i32 1901, i32 1907, i32 1913, i32 1931, i32 1933, i32 1949, i32 1951, i32 1973, i32 1979, i32 1987, i32 1993, i32 1997, i32 1999, i32 2003, i32 2011, i32 2017, i32 2027, i32 2029, i32 2039, i32 2053, i32 2063, i32 2069, i32 2081, i32 2083, i32 2087, i32 2089, i32 2099, i32 2111, i32 2113, i32 2129, i32 2131, i32 2137, i32 2141, i32 2143, i32 2153, i32 2161, i32 2179, i32 2203, i32 2207, i32 2213, i32 2221, i32 2237, i32 2239, i32 2243, i32 2251, i32 2267, i32 2269, i32 2273, i32 2281, i32 2287, i32 2293, i32 2297, i32 2309, i32 2311, i32 2333, i32 2339, i32 2341, i32 2347, i32 2351, i32 2357, i32 2371, i32 2377, i32 2381, i32 2383, i32 2389, i32 2393, i32 2399, i32 2411, i32 2417, i32 2423, i32 2437, i32 2441, i32 2447, i32 2459, i32 2467, i32 2473, i32 2477, i32 2503, i32 2521, i32 2531, i32 2539, i32 2543, i32 2549, i32 2551, i32 2557, i32 2579, i32 2591, i32 2593, i32 2609, i32 2617, i32 2621, i32 2633, i32 2647, i32 2657, i32 2659, i32 2663, i32 2671, i32 2677, i32 2683, i32 2687, i32 2689, i32 2693, i32 2699, i32 2707, i32 2711, i32 2713, i32 2719, i32 2729, i32 2731, i32 2741, i32 2749, i32 2753, i32 2767, i32 2777, i32 2789, i32 2791, i32 2797, i32 2801, i32 2803, i32 2819, i32 2833, i32 2837, i32 2843, i32 2851, i32 2857, i32 2861, i32 2879, i32 2887, i32 2897, i32 2903, i32 2909, i32 2917, i32 2927, i32 2939, i32 2953, i32 2957, i32 2963, i32 2969, i32 2971, i32 2999, i32 3001, i32 3011, i32 3019, i32 3023, i32 3037, i32 3041, i32 3049, i32 3061, i32 3067, i32 3079, i32 3083, i32 3089, i32 3109, i32 3119, i32 3121, i32 3137, i32 3163, i32 3167, i32 3169, i32 3181, i32 3187, i32 3191, i32 3203, i32 3209, i32 3217, i32 3221, i32 3229, i32 3251, i32 3253, i32 3257, i32 3259, i32 3271, i32 3299, i32 3301, i32 3307, i32 3313, i32 3319, i32 3323, i32 3329, i32 3331, i32 3343, i32 3347, i32 3359, i32 3361, i32 3371, i32 3373, i32 3389, i32 3391, i32 3407, i32 3413, i32 3433, i32 3449, i32 3457, i32 3461, i32 3463, i32 3467, i32 3469, i32 3491, i32 3499, i32 3511, i32 3517, i32 3527, i32 3529, i32 3533, i32 3539, i32 3541, i32 3547, i32 3557, i32 3559, i32 3571, i32 3581, i32 3583, i32 3593, i32 3607, i32 3613, i32 3617, i32 3623, i32 3631, i32 3637, i32 3643, i32 3659, i32 3671, i32 3673, i32 3677, i32 3691, i32 3697, i32 3701, i32 3709, i32 3719, i32 3727, i32 3733, i32 3739, i32 3761, i32 3767, i32 3769, i32 3779, i32 3793, i32 3797, i32 3803, i32 3821, i32 3823, i32 3833, i32 3847, i32 3851, i32 3853, i32 3863, i32 3877, i32 3881, i32 3889, i32 3907, i32 3911, i32 3917, i32 3919, i32 3923, i32 3929, i32 3931, i32 3943, i32 3947, i32 3967, i32 3989, i32 4001, i32 4003, i32 4007, i32 4013, i32 4019, i32 4021, i32 4027, i32 4049, i32 4051, i32 4057, i32 4073, i32 4079, i32 4091, i32 4093, i32 4099, i32 4111, i32 4127, i32 4129, i32 4133, i32 4139, i32 4153, i32 4157, i32 4159, i32 4177, i32 4201, i32 4211, i32 4217, i32 4219, i32 4229, i32 4231, i32 4241, i32 4243, i32 4253, i32 4259, i32 4261, i32 4271, i32 4273, i32 4283, i32 4289, i32 4297, i32 4327, i32 4337, i32 4339, i32 4349, i32 4357, i32 4363, i32 4373, i32 4391, i32 4397, i32 4409, i32 4421, i32 4423, i32 4441, i32 4447, i32 4451, i32 4457, i32 4463, i32 4481, i32 4483, i32 4493, i32 4507, i32 4513, i32 4517, i32 4519, i32 4523, i32 4547, i32 4549, i32 4561, i32 4567, i32 4583, i32 4591, i32 4597, i32 4603, i32 4621, i32 4637, i32 4639, i32 4643, i32 4649, i32 4651, i32 4657, i32 4663, i32 4673, i32 4679, i32 4691, i32 4703, i32 4721, i32 4723, i32 4729, i32 4733, i32 4751, i32 4759, i32 4783, i32 4787, i32 4789, i32 4793, i32 4799, i32 4801, i32 4813, i32 4817, i32 4831, i32 4861, i32 4871, i32 4877, i32 4889, i32 4903, i32 4909, i32 4919, i32 4931, i32 4933, i32 4937, i32 4943, i32 4951, i32 4957, i32 4967, i32 4969, i32 4973, i32 4987, i32 4993, i32 4999, i32 5003, i32 5009, i32 5011, i32 5021, i32 5023, i32 5039, i32 5051, i32 5059, i32 5077, i32 5081, i32 5087, i32 5099, i32 5101, i32 5107, i32 5113, i32 5119, i32 5147, i32 5153, i32 5167, i32 5171, i32 5179, i32 5189, i32 5197, i32 5209, i32 5227, i32 5231, i32 5233, i32 5237, i32 5261, i32 5273, i32 5279, i32 5281, i32 5297, i32 5303, i32 5309, i32 5323, i32 5333, i32 5347, i32 5351, i32 5381, i32 5387, i32 5393, i32 5399, i32 5407, i32 5413, i32 5417, i32 5419, i32 5431, i32 5437, i32 5441, i32 5443, i32 5449, i32 5471, i32 5477, i32 5479, i32 5483, i32 5501, i32 5503, i32 5507, i32 5519, i32 5521, i32 5527, i32 5531, i32 5557, i32 5563, i32 5569, i32 5573, i32 5581, i32 5591, i32 5623, i32 5639, i32 5641, i32 5647, i32 5651, i32 5653, i32 5657, i32 5659, i32 5669, i32 5683, i32 5689, i32 5693, i32 5701, i32 5711, i32 5717, i32 5737, i32 5741, i32 5743, i32 5749, i32 5779, i32 5783, i32 5791, i32 5801, i32 5807, i32 5813, i32 5821, i32 5827, i32 5839, i32 5843, i32 5849, i32 5851, i32 5857, i32 5861, i32 5867, i32 5869, i32 5879, i32 5881, i32 5897, i32 5903, i32 5923, i32 5927, i32 5939, i32 5953, i32 5981, i32 5987, i32 6007, i32 6011, i32 6029, i32 6037, i32 6043, i32 6047, i32 6053, i32 6067, i32 6073, i32 6079, i32 6089, i32 6091, i32 6101, i32 6113, i32 6121, i32 6131, i32 6133, i32 6143, i32 6151, i32 6163, i32 6173, i32 6197, i32 6199, i32 6203, i32 6211, i32 6217, i32 6221, i32 6229, i32 6247, i32 6257, i32 6263, i32 6269, i32 6271, i32 6277, i32 6287, i32 6299, i32 6301, i32 6311, i32 6317, i32 6323, i32 6329, i32 6337, i32 6343, i32 6353, i32 6359, i32 6361, i32 6367, i32 6373, i32 6379, i32 6389, i32 6397, i32 6421, i32 6427, i32 6449, i32 6451, i32 6469, i32 6473, i32 6481, i32 6491, i32 6521, i32 6529, i32 6547, i32 6551, i32 6553, i32 6563, i32 6569, i32 6571, i32 6577, i32 6581, i32 6599, i32 6607, i32 6619, i32 6637, i32 6653, i32 6659, i32 6661, i32 6673, i32 6679, i32 6689, i32 6691, i32 6701, i32 6703, i32 6709, i32 6719, i32 6733, i32 6737, i32 6761, i32 6763, i32 6779, i32 6781, i32 6791, i32 6793, i32 6803, i32 6823, i32 6827, i32 6829, i32 6833, i32 6841, i32 6857, i32 6863, i32 6869, i32 6871, i32 6883, i32 6899, i32 6907, i32 6911, i32 6917, i32 6947, i32 6949, i32 6959, i32 6961, i32 6967, i32 6971, i32 6977, i32 6983, i32 6991, i32 6997, i32 7001, i32 7013, i32 7019, i32 7027, i32 7039, i32 7043, i32 7057, i32 7069, i32 7079, i32 7103, i32 7109, i32 7121, i32 7127, i32 7129, i32 7151, i32 7159, i32 7177, i32 7187, i32 7193, i32 7207, i32 7211, i32 7213, i32 7219, i32 7229, i32 7237, i32 7243, i32 7247, i32 7253, i32 7283, i32 7297, i32 7307, i32 7309, i32 7321, i32 7331, i32 7333, i32 7349, i32 7351, i32 7369, i32 7393, i32 7411, i32 7417, i32 7433, i32 7451, i32 7457, i32 7459, i32 7477, i32 7481, i32 7487, i32 7489, i32 7499, i32 7507, i32 7517, i32 7523, i32 7529, i32 7537, i32 7541, i32 7547, i32 7549, i32 7559, i32 7561, i32 7573, i32 7577, i32 7583, i32 7589, i32 7591, i32 7603, i32 7607, i32 7621, i32 7639, i32 7643, i32 7649, i32 7669, i32 7673, i32 7681, i32 7687, i32 7691, i32 7699, i32 7703, i32 7717, i32 7723, i32 7727, i32 7741, i32 7753, i32 7757, i32 7759, i32 7789, i32 7793, i32 7817, i32 7823, i32 7829, i32 7841, i32 7853, i32 7867, i32 7873, i32 7877, i32 7879, i32 7883, i32 7901, i32 7907, i32 7919, i32 7927, i32 7933, i32 7937, i32 7949, i32 7951, i32 7963, i32 7993, i32 8009, i32 8011, i32 8017, i32 8039, i32 8053, i32 8059, i32 8069, i32 8081, i32 8087, i32 8089, i32 8093, i32 8101, i32 8111, i32 8117, i32 8123, i32 8147, i32 8161, i32 8167, i32 8171, i32 8179, i32 8191, i32 8209, i32 8219, i32 8221, i32 8231, i32 8233, i32 8237, i32 8243, i32 8263, i32 8269, i32 8273, i32 8287, i32 8291, i32 8293, i32 8297, i32 8311, i32 8317, i32 8329, i32 8353, i32 8363, i32 8369, i32 8377, i32 8387, i32 8389, i32 8419, i32 8423, i32 8429, i32 8431, i32 8443, i32 8447, i32 8461, i32 8467, i32 8501, i32 8513, i32 8521, i32 8527, i32 8537, i32 8539, i32 8543, i32 8563, i32 8573, i32 8581, i32 8597, i32 8599, i32 8609, i32 8623, i32 8627, i32 8629, i32 8641, i32 8647, i32 8663, i32 8669, i32 8677, i32 8681, i32 8689, i32 8693, i32 8699, i32 8707, i32 8713, i32 8719, i32 8731, i32 8737, i32 8741, i32 8747, i32 8753, i32 8761, i32 8779, i32 8783, i32 8803, i32 8807, i32 8819, i32 8821, i32 8831, i32 8837, i32 8839, i32 8849, i32 8861, i32 8863, i32 8867, i32 8887, i32 8893, i32 8923, i32 8929, i32 8933, i32 8941, i32 8951, i32 8963, i32 8969, i32 8971, i32 8999, i32 9001, i32 9007, i32 9011, i32 9013, i32 9029, i32 9041, i32 9043, i32 9049, i32 9059, i32 9067, i32 9091, i32 9103, i32 9109, i32 9127, i32 9133, i32 9137, i32 9151, i32 9157, i32 9161, i32 9173, i32 9181, i32 9187, i32 9199, i32 9203, i32 9209, i32 9221, i32 9227, i32 9239, i32 9241, i32 9257, i32 9277, i32 9281, i32 9283, i32 9293, i32 9311, i32 9319, i32 9323, i32 9337, i32 9341, i32 9343, i32 9349, i32 9371, i32 9377, i32 9391, i32 9397, i32 9403, i32 9413, i32 9419, i32 9421, i32 9431, i32 9433, i32 9437, i32 9439, i32 9461, i32 9463, i32 9467, i32 9473, i32 9479, i32 9491, i32 9497, i32 9511, i32 9521, i32 9533, i32 9539, i32 9547, i32 9551, i32 9587, i32 9601, i32 9613, i32 9619, i32 9623, i32 9629, i32 9631, i32 9643, i32 9649, i32 9661, i32 9677, i32 9679, i32 9689, i32 9697, i32 9719, i32 9721, i32 9733, i32 9739, i32 9743, i32 9749, i32 9767, i32 9769, i32 9781, i32 9787, i32 9791, i32 9803, i32 9811, i32 9817, i32 9829, i32 9833, i32 9839, i32 9851, i32 9857, i32 9859, i32 9871, i32 9883, i32 9887, i32 9901, i32 9907, i32 9923, i32 9929, i32 9931, i32 9941, i32 9949, i32 9967, i32 9973], align 4
@llvm.used = appending global [1 x ptr] [ptr addrspacecast (ptr addrspace(4) @prime32 to ptr)], section "llvm.metadata"

; Function Attrs: alwaysinline inlinehint
define linkonce_odr i32 @_Z6fmix32j(i32 %h) #0 !dbg !5 {
  %retval = alloca i32, align 4
  %h.addr = alloca i32, align 4
  store i32 %h, ptr %h.addr, align 4
  %tmp = load i32, ptr %h.addr, align 4, !dbg !8
  %shr = lshr i32 %tmp, 16, !dbg !8
  %tmp1 = load i32, ptr %h.addr, align 4, !dbg !8
  %xor = xor i32 %tmp1, %shr, !dbg !8
  store i32 %xor, ptr %h.addr, align 4, !dbg !8
  %tmp2 = load i32, ptr %h.addr, align 4, !dbg !10
  %mul = mul i32 %tmp2, -2048144789, !dbg !10
  store i32 %mul, ptr %h.addr, align 4, !dbg !10
  %tmp3 = load i32, ptr %h.addr, align 4, !dbg !11
  %shr4 = lshr i32 %tmp3, 13, !dbg !11
  %tmp5 = load i32, ptr %h.addr, align 4, !dbg !11
  %xor6 = xor i32 %tmp5, %shr4, !dbg !11
  store i32 %xor6, ptr %h.addr, align 4, !dbg !11
  %tmp7 = load i32, ptr %h.addr, align 4, !dbg !12
  %mul8 = mul i32 %tmp7, -1028477387, !dbg !12
  store i32 %mul8, ptr %h.addr, align 4, !dbg !12
  %tmp9 = load i32, ptr %h.addr, align 4, !dbg !13
  %shr10 = lshr i32 %tmp9, 16, !dbg !13
  %tmp11 = load i32, ptr %h.addr, align 4, !dbg !13
  %xor12 = xor i32 %tmp11, %shr10, !dbg !13
  store i32 %xor12, ptr %h.addr, align 4, !dbg !13
  %tmp13 = load i32, ptr %h.addr, align 4, !dbg !14
  store i32 %tmp13, ptr %retval, align 4, !dbg !14
  %1 = load i32, ptr %retval, align 4, !dbg !14
  ret i32 %1, !dbg !14
}

define i32 @_Z18MurmurHash3_x86_32PKvij(ptr %key, i32 %len, i32 %seed) !dbg !15 {
  %retval = alloca i32, align 4
  %key.addr = alloca ptr, align 8
  %len.addr = alloca i32, align 4
  %seed.addr = alloca i32, align 4
  %data = alloca ptr, align 8
  %nblocks = alloca i32, align 4
  %h1 = alloca i32, align 4
  %c1 = alloca i32, align 4
  %c2 = alloca i32, align 4
  %blocks = alloca ptr, align 8
  %tail = alloca ptr, align 8
  %k1 = alloca i32, align 4
  %i = alloca i32, align 4
  %k17 = alloca i32, align 4
  store ptr %key, ptr %key.addr, align 8
  store i32 %len, ptr %len.addr, align 4
  store i32 %seed, ptr %seed.addr, align 4
  %tmp = load ptr, ptr %key.addr, align 8, !dbg !16
  store ptr %tmp, ptr %data, align 8, !dbg !16
  %tmp1 = load i32, ptr %len.addr, align 4, !dbg !18
  %div = sdiv i32 %tmp1, 4, !dbg !18
  store i32 %div, ptr %nblocks, align 4, !dbg !18
  %tmp2 = load i32, ptr %seed.addr, align 4, !dbg !19
  store i32 %tmp2, ptr %h1, align 4, !dbg !19
  store i32 -862048943, ptr %c1, align 4, !dbg !20
  store i32 461845907, ptr %c2, align 4, !dbg !21
  %tmp3 = load ptr, ptr %data, align 8, !dbg !22
  %tmp4 = load i32, ptr %nblocks, align 4, !dbg !22
  %mul = mul nsw i32 %tmp4, 4, !dbg !22
  %add.ptr = getelementptr inbounds i8, ptr %tmp3, i32 %mul, !dbg !22
  %conv = bitcast ptr %add.ptr to ptr, !dbg !22
  store ptr %conv, ptr %blocks, align 8, !dbg !22
  %tmp5 = load i32, ptr %nblocks, align 4, !dbg !23
  %neg = sub nsw i32 0, %tmp5, !dbg !23
  store i32 %neg, ptr %i, align 4, !dbg !23
  br label %1, !dbg !23

1:                                                ; preds = %3, %0
  %tmp6 = load i32, ptr %i, align 4, !dbg !23
  %tobool = icmp ne i32 %tmp6, 0, !dbg !23
  br i1 %tobool, label %2, label %4, !dbg !23

2:                                                ; preds = %1
  %tmp8 = load ptr, ptr %blocks, align 8, !dbg !24
  %tmp9 = load i32, ptr %i, align 4, !dbg !24
  %arrayidx = getelementptr inbounds i32, ptr %tmp8, i32 %tmp9, !dbg !24
  %tmp10 = load i32, ptr %arrayidx, align 4, !dbg !24
  store i32 %tmp10, ptr %k17, align 4, !dbg !24
  %tmp11 = load i32, ptr %k17, align 4, !dbg !28
  %mul12 = mul i32 %tmp11, -862048943, !dbg !28
  store i32 %mul12, ptr %k17, align 4, !dbg !28
  %tmp13 = load i32, ptr %k17, align 4, !dbg !29
  %shl = shl i32 %tmp13, 15, !dbg !29
  %tmp14 = load i32, ptr %k17, align 4, !dbg !29
  %shr = lshr i32 %tmp14, 17, !dbg !29
  %or = or i32 %shl, %shr, !dbg !29
  store i32 %or, ptr %k17, align 4, !dbg !29
  %tmp15 = load i32, ptr %k17, align 4, !dbg !30
  %mul16 = mul i32 %tmp15, 461845907, !dbg !30
  store i32 %mul16, ptr %k17, align 4, !dbg !30
  %tmp17 = load i32, ptr %k17, align 4, !dbg !31
  %tmp18 = load i32, ptr %h1, align 4, !dbg !31
  %xor = xor i32 %tmp18, %tmp17, !dbg !31
  store i32 %xor, ptr %h1, align 4, !dbg !31
  %tmp19 = load i32, ptr %h1, align 4, !dbg !32
  %shl20 = shl i32 %tmp19, 13, !dbg !32
  %tmp21 = load i32, ptr %h1, align 4, !dbg !32
  %shr22 = lshr i32 %tmp21, 19, !dbg !32
  %or23 = or i32 %shl20, %shr22, !dbg !32
  store i32 %or23, ptr %h1, align 4, !dbg !32
  %tmp24 = load i32, ptr %h1, align 4, !dbg !33
  %mul25 = mul i32 %tmp24, 5, !dbg !33
  %add = add i32 %mul25, -430675100, !dbg !33
  store i32 %add, ptr %h1, align 4, !dbg !33
  br label %3, !dbg !34

3:                                                ; preds = %2
  %tmp26 = load i32, ptr %i, align 4, !dbg !34
  %inc = add nsw i32 %tmp26, 1, !dbg !34
  store i32 %inc, ptr %i, align 4, !dbg !34
  br label %1, !dbg !34

4:                                                ; preds = %1
  %tmp27 = load ptr, ptr %data, align 8, !dbg !35
  %tmp28 = load i32, ptr %nblocks, align 4, !dbg !35
  %mul29 = mul nsw i32 %tmp28, 4, !dbg !35
  %add.ptr30 = getelementptr inbounds i8, ptr %tmp27, i32 %mul29, !dbg !35
  store ptr %add.ptr30, ptr %tail, align 8, !dbg !35
  store i32 0, ptr %k1, align 4, !dbg !36
  %tmp31 = load i32, ptr %len.addr, align 4, !dbg !37
  %and = and i32 %tmp31, 3, !dbg !37
  switch i32 %and, label %9 [
    i32 1, label %8
    i32 2, label %7
    i32 3, label %6
  ], !dbg !37

5:                                                ; No predecessors!
  br label %6, !dbg !38

6:                                                ; preds = %5, %4
  %tmp32 = load ptr, ptr %tail, align 8, !dbg !38
  %arrayidx33 = getelementptr inbounds i8, ptr %tmp32, i32 2, !dbg !38
  %tmp34 = load i8, ptr %arrayidx33, align 1, !dbg !38
  %conv35 = zext i8 %tmp34 to i32, !dbg !38
  %shl36 = shl i32 %conv35, 16, !dbg !38
  %tmp37 = load i32, ptr %k1, align 4, !dbg !38
  %xor38 = xor i32 %tmp37, %shl36, !dbg !38
  store i32 %xor38, ptr %k1, align 4, !dbg !38
  br label %7, !dbg !40

7:                                                ; preds = %6, %4
  %tmp39 = load ptr, ptr %tail, align 8, !dbg !40
  %arrayidx40 = getelementptr inbounds i8, ptr %tmp39, i32 1, !dbg !40
  %tmp41 = load i8, ptr %arrayidx40, align 1, !dbg !40
  %conv42 = zext i8 %tmp41 to i32, !dbg !40
  %shl43 = shl i32 %conv42, 8, !dbg !40
  %tmp44 = load i32, ptr %k1, align 4, !dbg !40
  %xor45 = xor i32 %tmp44, %shl43, !dbg !40
  store i32 %xor45, ptr %k1, align 4, !dbg !40
  br label %8, !dbg !41

8:                                                ; preds = %7, %4
  %tmp46 = load ptr, ptr %tail, align 8, !dbg !41
  %arrayidx47 = getelementptr inbounds i8, ptr %tmp46, i32 0, !dbg !41
  %tmp48 = load i8, ptr %arrayidx47, align 1, !dbg !41
  %conv49 = zext i8 %tmp48 to i32, !dbg !41
  %tmp50 = load i32, ptr %k1, align 4, !dbg !41
  %xor51 = xor i32 %tmp50, %conv49, !dbg !41
  store i32 %xor51, ptr %k1, align 4, !dbg !41
  %tmp52 = load i32, ptr %k1, align 4, !dbg !42
  %mul53 = mul i32 %tmp52, -862048943, !dbg !42
  store i32 %mul53, ptr %k1, align 4, !dbg !42
  %tmp54 = load i32, ptr %k1, align 4, !dbg !42
  %shl55 = shl i32 %tmp54, 15, !dbg !42
  %tmp56 = load i32, ptr %k1, align 4, !dbg !42
  %shr57 = lshr i32 %tmp56, 17, !dbg !42
  %or58 = or i32 %shl55, %shr57, !dbg !42
  store i32 %or58, ptr %k1, align 4, !dbg !42
  %tmp59 = load i32, ptr %k1, align 4, !dbg !42
  %mul60 = mul i32 %tmp59, 461845907, !dbg !42
  store i32 %mul60, ptr %k1, align 4, !dbg !42
  %tmp61 = load i32, ptr %k1, align 4, !dbg !42
  %tmp62 = load i32, ptr %h1, align 4, !dbg !42
  %xor63 = xor i32 %tmp62, %tmp61, !dbg !42
  store i32 %xor63, ptr %h1, align 4, !dbg !42
  br label %9, !dbg !42

9:                                                ; preds = %8, %4
  %tmp64 = load i32, ptr %len.addr, align 4, !dbg !43
  %tmp65 = load i32, ptr %h1, align 4, !dbg !43
  %xor66 = xor i32 %tmp65, %tmp64, !dbg !43
  store i32 %xor66, ptr %h1, align 4, !dbg !43
  %tmp67 = load i32, ptr %h1, align 4, !dbg !44
  %call = call i32 @_Z6fmix32j(i32 %tmp67), !dbg !44
  store i32 %call, ptr %h1, align 4, !dbg !44
  %tmp68 = load i32, ptr %h1, align 4, !dbg !45
  store i32 %tmp68, ptr %retval, align 4, !dbg !45
  %10 = load i32, ptr %retval, align 4, !dbg !45
  ret i32 %10, !dbg !45
}

define i32 @_Z9BOBHash32PKcjj(ptr %str, i32 %len, i32 %seed) !dbg !46 {
  %retval = alloca i32, align 4
  %str.addr = alloca ptr, align 8
  %len.addr = alloca i32, align 4
  %seed.addr = alloca i32, align 4
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %c = alloca i32, align 4
  store ptr %str, ptr %str.addr, align 8
  store i32 %len, ptr %len.addr, align 4
  store i32 %seed, ptr %seed.addr, align 4
  store i32 -1640531527, ptr %b, align 4, !dbg !47
  %tmp = load i32, ptr %b, align 4, !dbg !47
  store i32 %tmp, ptr %a, align 4, !dbg !47
  %tmp1 = load i32, ptr %seed.addr, align 4, !dbg !49
  %cmp = icmp ult i32 %tmp1, 1229, !dbg !49
  br i1 %cmp, label %1, label %2, !dbg !49

1:                                                ; preds = %0
  %tmp2 = load i32, ptr %seed.addr, align 4, !dbg !50
  %idxprom = zext i32 %tmp2 to i64, !dbg !50
  %arrayidx = getelementptr inbounds i32, ptr addrspacecast (ptr addrspace(4) @prime32 to ptr), i64 %idxprom, !dbg !50
  %tmp3 = load i32, ptr %arrayidx, align 4, !dbg !50
  store i32 %tmp3, ptr %c, align 4, !dbg !50
  br label %3, !dbg !50

2:                                                ; preds = %0
  %tmp4 = load i32, ptr %seed.addr, align 4, !dbg !52
  store i32 %tmp4, ptr %c, align 4, !dbg !52
  br label %3, !dbg !52

3:                                                ; preds = %2, %1
  br label %4, !dbg !54

4:                                                ; preds = %5, %3
  %tmp5 = load i32, ptr %len.addr, align 4, !dbg !54
  %cmp6 = icmp uge i32 %tmp5, 12, !dbg !54
  br i1 %cmp6, label %5, label %6, !dbg !54

5:                                                ; preds = %4
  %tmp7 = load ptr, ptr %str.addr, align 8, !dbg !55
  %arrayidx8 = getelementptr inbounds i8, ptr %tmp7, i32 0, !dbg !55
  %tmp9 = load i8, ptr %arrayidx8, align 1, !dbg !55
  %conv = sext i8 %tmp9 to i32, !dbg !55
  %tmp10 = load ptr, ptr %str.addr, align 8, !dbg !55
  %arrayidx11 = getelementptr inbounds i8, ptr %tmp10, i32 1, !dbg !55
  %tmp12 = load i8, ptr %arrayidx11, align 1, !dbg !55
  %conv13 = sext i8 %tmp12 to i32, !dbg !55
  %shl = shl i32 %conv13, 8, !dbg !55
  %add = add i32 %conv, %shl, !dbg !55
  %tmp14 = load ptr, ptr %str.addr, align 8, !dbg !55
  %arrayidx15 = getelementptr inbounds i8, ptr %tmp14, i32 2, !dbg !55
  %tmp16 = load i8, ptr %arrayidx15, align 1, !dbg !55
  %conv17 = sext i8 %tmp16 to i32, !dbg !55
  %shl18 = shl i32 %conv17, 16, !dbg !55
  %add19 = add i32 %add, %shl18, !dbg !55
  %tmp20 = load ptr, ptr %str.addr, align 8, !dbg !55
  %arrayidx21 = getelementptr inbounds i8, ptr %tmp20, i32 3, !dbg !55
  %tmp22 = load i8, ptr %arrayidx21, align 1, !dbg !55
  %conv23 = sext i8 %tmp22 to i32, !dbg !55
  %shl24 = shl i32 %conv23, 24, !dbg !55
  %add25 = add i32 %add19, %shl24, !dbg !55
  %tmp26 = load i32, ptr %a, align 4, !dbg !55
  %add27 = add i32 %tmp26, %add25, !dbg !55
  store i32 %add27, ptr %a, align 4, !dbg !55
  %tmp28 = load ptr, ptr %str.addr, align 8, !dbg !57
  %arrayidx29 = getelementptr inbounds i8, ptr %tmp28, i32 4, !dbg !57
  %tmp30 = load i8, ptr %arrayidx29, align 1, !dbg !57
  %conv31 = sext i8 %tmp30 to i32, !dbg !57
  %tmp32 = load ptr, ptr %str.addr, align 8, !dbg !57
  %arrayidx33 = getelementptr inbounds i8, ptr %tmp32, i32 5, !dbg !57
  %tmp34 = load i8, ptr %arrayidx33, align 1, !dbg !57
  %conv35 = sext i8 %tmp34 to i32, !dbg !57
  %shl36 = shl i32 %conv35, 8, !dbg !57
  %add37 = add i32 %conv31, %shl36, !dbg !57
  %tmp38 = load ptr, ptr %str.addr, align 8, !dbg !57
  %arrayidx39 = getelementptr inbounds i8, ptr %tmp38, i32 6, !dbg !57
  %tmp40 = load i8, ptr %arrayidx39, align 1, !dbg !57
  %conv41 = sext i8 %tmp40 to i32, !dbg !57
  %shl42 = shl i32 %conv41, 16, !dbg !57
  %add43 = add i32 %add37, %shl42, !dbg !57
  %tmp44 = load ptr, ptr %str.addr, align 8, !dbg !57
  %arrayidx45 = getelementptr inbounds i8, ptr %tmp44, i32 7, !dbg !57
  %tmp46 = load i8, ptr %arrayidx45, align 1, !dbg !57
  %conv47 = sext i8 %tmp46 to i32, !dbg !57
  %shl48 = shl i32 %conv47, 24, !dbg !57
  %add49 = add i32 %add43, %shl48, !dbg !57
  %tmp50 = load i32, ptr %b, align 4, !dbg !57
  %add51 = add i32 %tmp50, %add49, !dbg !57
  store i32 %add51, ptr %b, align 4, !dbg !57
  %tmp52 = load ptr, ptr %str.addr, align 8, !dbg !58
  %arrayidx53 = getelementptr inbounds i8, ptr %tmp52, i32 8, !dbg !58
  %tmp54 = load i8, ptr %arrayidx53, align 1, !dbg !58
  %conv55 = sext i8 %tmp54 to i32, !dbg !58
  %tmp56 = load ptr, ptr %str.addr, align 8, !dbg !58
  %arrayidx57 = getelementptr inbounds i8, ptr %tmp56, i32 9, !dbg !58
  %tmp58 = load i8, ptr %arrayidx57, align 1, !dbg !58
  %conv59 = sext i8 %tmp58 to i32, !dbg !58
  %shl60 = shl i32 %conv59, 8, !dbg !58
  %add61 = add i32 %conv55, %shl60, !dbg !58
  %tmp62 = load ptr, ptr %str.addr, align 8, !dbg !58
  %arrayidx63 = getelementptr inbounds i8, ptr %tmp62, i32 10, !dbg !58
  %tmp64 = load i8, ptr %arrayidx63, align 1, !dbg !58
  %conv65 = sext i8 %tmp64 to i32, !dbg !58
  %shl66 = shl i32 %conv65, 16, !dbg !58
  %add67 = add i32 %add61, %shl66, !dbg !58
  %tmp68 = load ptr, ptr %str.addr, align 8, !dbg !58
  %arrayidx69 = getelementptr inbounds i8, ptr %tmp68, i32 11, !dbg !58
  %tmp70 = load i8, ptr %arrayidx69, align 1, !dbg !58
  %conv71 = sext i8 %tmp70 to i32, !dbg !58
  %shl72 = shl i32 %conv71, 24, !dbg !58
  %add73 = add i32 %add67, %shl72, !dbg !58
  %tmp74 = load i32, ptr %c, align 4, !dbg !58
  %add75 = add i32 %tmp74, %add73, !dbg !58
  store i32 %add75, ptr %c, align 4, !dbg !58
  %tmp76 = load i32, ptr %b, align 4, !dbg !59
  %tmp77 = load i32, ptr %a, align 4, !dbg !59
  %sub = sub i32 %tmp77, %tmp76, !dbg !59
  store i32 %sub, ptr %a, align 4, !dbg !59
  %tmp78 = load i32, ptr %c, align 4, !dbg !59
  %tmp79 = load i32, ptr %a, align 4, !dbg !59
  %sub80 = sub i32 %tmp79, %tmp78, !dbg !59
  store i32 %sub80, ptr %a, align 4, !dbg !59
  %tmp81 = load i32, ptr %c, align 4, !dbg !59
  %shr = lshr i32 %tmp81, 13, !dbg !59
  %tmp82 = load i32, ptr %a, align 4, !dbg !59
  %xor = xor i32 %tmp82, %shr, !dbg !59
  store i32 %xor, ptr %a, align 4, !dbg !59
  %tmp83 = load i32, ptr %c, align 4, !dbg !59
  %tmp84 = load i32, ptr %b, align 4, !dbg !59
  %sub85 = sub i32 %tmp84, %tmp83, !dbg !59
  store i32 %sub85, ptr %b, align 4, !dbg !59
  %tmp86 = load i32, ptr %a, align 4, !dbg !59
  %tmp87 = load i32, ptr %b, align 4, !dbg !59
  %sub88 = sub i32 %tmp87, %tmp86, !dbg !59
  store i32 %sub88, ptr %b, align 4, !dbg !59
  %tmp89 = load i32, ptr %a, align 4, !dbg !59
  %shl90 = shl i32 %tmp89, 8, !dbg !59
  %tmp91 = load i32, ptr %b, align 4, !dbg !59
  %xor92 = xor i32 %tmp91, %shl90, !dbg !59
  store i32 %xor92, ptr %b, align 4, !dbg !59
  %tmp93 = load i32, ptr %a, align 4, !dbg !59
  %tmp94 = load i32, ptr %c, align 4, !dbg !59
  %sub95 = sub i32 %tmp94, %tmp93, !dbg !59
  store i32 %sub95, ptr %c, align 4, !dbg !59
  %tmp96 = load i32, ptr %b, align 4, !dbg !59
  %tmp97 = load i32, ptr %c, align 4, !dbg !59
  %sub98 = sub i32 %tmp97, %tmp96, !dbg !59
  store i32 %sub98, ptr %c, align 4, !dbg !59
  %tmp99 = load i32, ptr %b, align 4, !dbg !59
  %shr100 = lshr i32 %tmp99, 13, !dbg !59
  %tmp101 = load i32, ptr %c, align 4, !dbg !59
  %xor102 = xor i32 %tmp101, %shr100, !dbg !59
  store i32 %xor102, ptr %c, align 4, !dbg !59
  %tmp103 = load i32, ptr %b, align 4, !dbg !59
  %tmp104 = load i32, ptr %a, align 4, !dbg !59
  %sub105 = sub i32 %tmp104, %tmp103, !dbg !59
  store i32 %sub105, ptr %a, align 4, !dbg !59
  %tmp106 = load i32, ptr %c, align 4, !dbg !59
  %tmp107 = load i32, ptr %a, align 4, !dbg !59
  %sub108 = sub i32 %tmp107, %tmp106, !dbg !59
  store i32 %sub108, ptr %a, align 4, !dbg !59
  %tmp109 = load i32, ptr %c, align 4, !dbg !59
  %shr110 = lshr i32 %tmp109, 12, !dbg !59
  %tmp111 = load i32, ptr %a, align 4, !dbg !59
  %xor112 = xor i32 %tmp111, %shr110, !dbg !59
  store i32 %xor112, ptr %a, align 4, !dbg !59
  %tmp113 = load i32, ptr %c, align 4, !dbg !59
  %tmp114 = load i32, ptr %b, align 4, !dbg !59
  %sub115 = sub i32 %tmp114, %tmp113, !dbg !59
  store i32 %sub115, ptr %b, align 4, !dbg !59
  %tmp116 = load i32, ptr %a, align 4, !dbg !59
  %tmp117 = load i32, ptr %b, align 4, !dbg !59
  %sub118 = sub i32 %tmp117, %tmp116, !dbg !59
  store i32 %sub118, ptr %b, align 4, !dbg !59
  %tmp119 = load i32, ptr %a, align 4, !dbg !59
  %shl120 = shl i32 %tmp119, 16, !dbg !59
  %tmp121 = load i32, ptr %b, align 4, !dbg !59
  %xor122 = xor i32 %tmp121, %shl120, !dbg !59
  store i32 %xor122, ptr %b, align 4, !dbg !59
  %tmp123 = load i32, ptr %a, align 4, !dbg !59
  %tmp124 = load i32, ptr %c, align 4, !dbg !59
  %sub125 = sub i32 %tmp124, %tmp123, !dbg !59
  store i32 %sub125, ptr %c, align 4, !dbg !59
  %tmp126 = load i32, ptr %b, align 4, !dbg !59
  %tmp127 = load i32, ptr %c, align 4, !dbg !59
  %sub128 = sub i32 %tmp127, %tmp126, !dbg !59
  store i32 %sub128, ptr %c, align 4, !dbg !59
  %tmp129 = load i32, ptr %b, align 4, !dbg !59
  %shr130 = lshr i32 %tmp129, 5, !dbg !59
  %tmp131 = load i32, ptr %c, align 4, !dbg !59
  %xor132 = xor i32 %tmp131, %shr130, !dbg !59
  store i32 %xor132, ptr %c, align 4, !dbg !59
  %tmp133 = load i32, ptr %b, align 4, !dbg !59
  %tmp134 = load i32, ptr %a, align 4, !dbg !59
  %sub135 = sub i32 %tmp134, %tmp133, !dbg !59
  store i32 %sub135, ptr %a, align 4, !dbg !59
  %tmp136 = load i32, ptr %c, align 4, !dbg !59
  %tmp137 = load i32, ptr %a, align 4, !dbg !59
  %sub138 = sub i32 %tmp137, %tmp136, !dbg !59
  store i32 %sub138, ptr %a, align 4, !dbg !59
  %tmp139 = load i32, ptr %c, align 4, !dbg !59
  %shr140 = lshr i32 %tmp139, 3, !dbg !59
  %tmp141 = load i32, ptr %a, align 4, !dbg !59
  %xor142 = xor i32 %tmp141, %shr140, !dbg !59
  store i32 %xor142, ptr %a, align 4, !dbg !59
  %tmp143 = load i32, ptr %c, align 4, !dbg !59
  %tmp144 = load i32, ptr %b, align 4, !dbg !59
  %sub145 = sub i32 %tmp144, %tmp143, !dbg !59
  store i32 %sub145, ptr %b, align 4, !dbg !59
  %tmp146 = load i32, ptr %a, align 4, !dbg !59
  %tmp147 = load i32, ptr %b, align 4, !dbg !59
  %sub148 = sub i32 %tmp147, %tmp146, !dbg !59
  store i32 %sub148, ptr %b, align 4, !dbg !59
  %tmp149 = load i32, ptr %a, align 4, !dbg !59
  %shl150 = shl i32 %tmp149, 10, !dbg !59
  %tmp151 = load i32, ptr %b, align 4, !dbg !59
  %xor152 = xor i32 %tmp151, %shl150, !dbg !59
  store i32 %xor152, ptr %b, align 4, !dbg !59
  %tmp153 = load i32, ptr %a, align 4, !dbg !59
  %tmp154 = load i32, ptr %c, align 4, !dbg !59
  %sub155 = sub i32 %tmp154, %tmp153, !dbg !59
  store i32 %sub155, ptr %c, align 4, !dbg !59
  %tmp156 = load i32, ptr %b, align 4, !dbg !59
  %tmp157 = load i32, ptr %c, align 4, !dbg !59
  %sub158 = sub i32 %tmp157, %tmp156, !dbg !59
  store i32 %sub158, ptr %c, align 4, !dbg !59
  %tmp159 = load i32, ptr %b, align 4, !dbg !59
  %shr160 = lshr i32 %tmp159, 15, !dbg !59
  %tmp161 = load i32, ptr %c, align 4, !dbg !59
  %xor162 = xor i32 %tmp161, %shr160, !dbg !59
  store i32 %xor162, ptr %c, align 4, !dbg !59
  %tmp163 = load ptr, ptr %str.addr, align 8, !dbg !60
  %add.ptr = getelementptr inbounds i8, ptr %tmp163, i32 12, !dbg !60
  store ptr %add.ptr, ptr %str.addr, align 8, !dbg !60
  %tmp164 = load i32, ptr %len.addr, align 4, !dbg !60
  %sub165 = sub i32 %tmp164, 12, !dbg !60
  store i32 %sub165, ptr %len.addr, align 4, !dbg !60
  br label %4, !dbg !60

6:                                                ; preds = %4
  %tmp166 = load i32, ptr %len.addr, align 4, !dbg !61
  %tmp167 = load i32, ptr %c, align 4, !dbg !61
  %add168 = add i32 %tmp167, %tmp166, !dbg !61
  store i32 %add168, ptr %c, align 4, !dbg !61
  %tmp169 = load i32, ptr %len.addr, align 4, !dbg !62
  switch i32 %tmp169, label %19 [
    i32 1, label %18
    i32 2, label %17
    i32 3, label %16
    i32 4, label %15
    i32 5, label %14
    i32 6, label %13
    i32 7, label %12
    i32 8, label %11
    i32 9, label %10
    i32 10, label %9
    i32 11, label %8
  ], !dbg !62

7:                                                ; No predecessors!
  br label %8, !dbg !63

8:                                                ; preds = %7, %6
  %tmp170 = load ptr, ptr %str.addr, align 8, !dbg !63
  %arrayidx171 = getelementptr inbounds i8, ptr %tmp170, i32 10, !dbg !63
  %tmp172 = load i8, ptr %arrayidx171, align 1, !dbg !63
  %conv173 = sext i8 %tmp172 to i32, !dbg !63
  %shl174 = shl i32 %conv173, 24, !dbg !63
  %tmp175 = load i32, ptr %c, align 4, !dbg !63
  %add176 = add i32 %tmp175, %shl174, !dbg !63
  store i32 %add176, ptr %c, align 4, !dbg !63
  br label %9, !dbg !65

9:                                                ; preds = %8, %6
  %tmp177 = load ptr, ptr %str.addr, align 8, !dbg !65
  %arrayidx178 = getelementptr inbounds i8, ptr %tmp177, i32 9, !dbg !65
  %tmp179 = load i8, ptr %arrayidx178, align 1, !dbg !65
  %conv180 = sext i8 %tmp179 to i32, !dbg !65
  %shl181 = shl i32 %conv180, 16, !dbg !65
  %tmp182 = load i32, ptr %c, align 4, !dbg !65
  %add183 = add i32 %tmp182, %shl181, !dbg !65
  store i32 %add183, ptr %c, align 4, !dbg !65
  br label %10, !dbg !66

10:                                               ; preds = %9, %6
  %tmp184 = load ptr, ptr %str.addr, align 8, !dbg !66
  %arrayidx185 = getelementptr inbounds i8, ptr %tmp184, i32 8, !dbg !66
  %tmp186 = load i8, ptr %arrayidx185, align 1, !dbg !66
  %conv187 = sext i8 %tmp186 to i32, !dbg !66
  %shl188 = shl i32 %conv187, 8, !dbg !66
  %tmp189 = load i32, ptr %c, align 4, !dbg !66
  %add190 = add i32 %tmp189, %shl188, !dbg !66
  store i32 %add190, ptr %c, align 4, !dbg !66
  br label %11, !dbg !67

11:                                               ; preds = %10, %6
  %tmp191 = load ptr, ptr %str.addr, align 8, !dbg !67
  %arrayidx192 = getelementptr inbounds i8, ptr %tmp191, i32 7, !dbg !67
  %tmp193 = load i8, ptr %arrayidx192, align 1, !dbg !67
  %conv194 = sext i8 %tmp193 to i32, !dbg !67
  %shl195 = shl i32 %conv194, 24, !dbg !67
  %tmp196 = load i32, ptr %b, align 4, !dbg !67
  %add197 = add i32 %tmp196, %shl195, !dbg !67
  store i32 %add197, ptr %b, align 4, !dbg !67
  br label %12, !dbg !68

12:                                               ; preds = %11, %6
  %tmp198 = load ptr, ptr %str.addr, align 8, !dbg !68
  %arrayidx199 = getelementptr inbounds i8, ptr %tmp198, i32 6, !dbg !68
  %tmp200 = load i8, ptr %arrayidx199, align 1, !dbg !68
  %conv201 = sext i8 %tmp200 to i32, !dbg !68
  %shl202 = shl i32 %conv201, 16, !dbg !68
  %tmp203 = load i32, ptr %b, align 4, !dbg !68
  %add204 = add i32 %tmp203, %shl202, !dbg !68
  store i32 %add204, ptr %b, align 4, !dbg !68
  br label %13, !dbg !69

13:                                               ; preds = %12, %6
  %tmp205 = load ptr, ptr %str.addr, align 8, !dbg !69
  %arrayidx206 = getelementptr inbounds i8, ptr %tmp205, i32 5, !dbg !69
  %tmp207 = load i8, ptr %arrayidx206, align 1, !dbg !69
  %conv208 = sext i8 %tmp207 to i32, !dbg !69
  %shl209 = shl i32 %conv208, 8, !dbg !69
  %tmp210 = load i32, ptr %b, align 4, !dbg !69
  %add211 = add i32 %tmp210, %shl209, !dbg !69
  store i32 %add211, ptr %b, align 4, !dbg !69
  br label %14, !dbg !70

14:                                               ; preds = %13, %6
  %tmp212 = load ptr, ptr %str.addr, align 8, !dbg !70
  %arrayidx213 = getelementptr inbounds i8, ptr %tmp212, i32 4, !dbg !70
  %tmp214 = load i8, ptr %arrayidx213, align 1, !dbg !70
  %conv215 = sext i8 %tmp214 to i32, !dbg !70
  %tmp216 = load i32, ptr %b, align 4, !dbg !70
  %add217 = add i32 %tmp216, %conv215, !dbg !70
  store i32 %add217, ptr %b, align 4, !dbg !70
  br label %15, !dbg !71

15:                                               ; preds = %14, %6
  %tmp218 = load ptr, ptr %str.addr, align 8, !dbg !71
  %arrayidx219 = getelementptr inbounds i8, ptr %tmp218, i32 3, !dbg !71
  %tmp220 = load i8, ptr %arrayidx219, align 1, !dbg !71
  %conv221 = sext i8 %tmp220 to i32, !dbg !71
  %shl222 = shl i32 %conv221, 24, !dbg !71
  %tmp223 = load i32, ptr %a, align 4, !dbg !71
  %add224 = add i32 %tmp223, %shl222, !dbg !71
  store i32 %add224, ptr %a, align 4, !dbg !71
  br label %16, !dbg !72

16:                                               ; preds = %15, %6
  %tmp225 = load ptr, ptr %str.addr, align 8, !dbg !72
  %arrayidx226 = getelementptr inbounds i8, ptr %tmp225, i32 2, !dbg !72
  %tmp227 = load i8, ptr %arrayidx226, align 1, !dbg !72
  %conv228 = sext i8 %tmp227 to i32, !dbg !72
  %shl229 = shl i32 %conv228, 16, !dbg !72
  %tmp230 = load i32, ptr %a, align 4, !dbg !72
  %add231 = add i32 %tmp230, %shl229, !dbg !72
  store i32 %add231, ptr %a, align 4, !dbg !72
  br label %17, !dbg !73

17:                                               ; preds = %16, %6
  %tmp232 = load ptr, ptr %str.addr, align 8, !dbg !73
  %arrayidx233 = getelementptr inbounds i8, ptr %tmp232, i32 1, !dbg !73
  %tmp234 = load i8, ptr %arrayidx233, align 1, !dbg !73
  %conv235 = sext i8 %tmp234 to i32, !dbg !73
  %shl236 = shl i32 %conv235, 8, !dbg !73
  %tmp237 = load i32, ptr %a, align 4, !dbg !73
  %add238 = add i32 %tmp237, %shl236, !dbg !73
  store i32 %add238, ptr %a, align 4, !dbg !73
  br label %18, !dbg !74

18:                                               ; preds = %17, %6
  %tmp239 = load ptr, ptr %str.addr, align 8, !dbg !74
  %arrayidx240 = getelementptr inbounds i8, ptr %tmp239, i32 0, !dbg !74
  %tmp241 = load i8, ptr %arrayidx240, align 1, !dbg !74
  %conv242 = sext i8 %tmp241 to i32, !dbg !74
  %tmp243 = load i32, ptr %a, align 4, !dbg !74
  %add244 = add i32 %tmp243, %conv242, !dbg !74
  store i32 %add244, ptr %a, align 4, !dbg !74
  br label %19, !dbg !74

19:                                               ; preds = %18, %6
  %tmp245 = load i32, ptr %b, align 4, !dbg !75
  %tmp246 = load i32, ptr %a, align 4, !dbg !75
  %sub247 = sub i32 %tmp246, %tmp245, !dbg !75
  store i32 %sub247, ptr %a, align 4, !dbg !75
  %tmp248 = load i32, ptr %c, align 4, !dbg !75
  %tmp249 = load i32, ptr %a, align 4, !dbg !75
  %sub250 = sub i32 %tmp249, %tmp248, !dbg !75
  store i32 %sub250, ptr %a, align 4, !dbg !75
  %tmp251 = load i32, ptr %c, align 4, !dbg !75
  %shr252 = lshr i32 %tmp251, 13, !dbg !75
  %tmp253 = load i32, ptr %a, align 4, !dbg !75
  %xor254 = xor i32 %tmp253, %shr252, !dbg !75
  store i32 %xor254, ptr %a, align 4, !dbg !75
  %tmp255 = load i32, ptr %c, align 4, !dbg !75
  %tmp256 = load i32, ptr %b, align 4, !dbg !75
  %sub257 = sub i32 %tmp256, %tmp255, !dbg !75
  store i32 %sub257, ptr %b, align 4, !dbg !75
  %tmp258 = load i32, ptr %a, align 4, !dbg !75
  %tmp259 = load i32, ptr %b, align 4, !dbg !75
  %sub260 = sub i32 %tmp259, %tmp258, !dbg !75
  store i32 %sub260, ptr %b, align 4, !dbg !75
  %tmp261 = load i32, ptr %a, align 4, !dbg !75
  %shl262 = shl i32 %tmp261, 8, !dbg !75
  %tmp263 = load i32, ptr %b, align 4, !dbg !75
  %xor264 = xor i32 %tmp263, %shl262, !dbg !75
  store i32 %xor264, ptr %b, align 4, !dbg !75
  %tmp265 = load i32, ptr %a, align 4, !dbg !75
  %tmp266 = load i32, ptr %c, align 4, !dbg !75
  %sub267 = sub i32 %tmp266, %tmp265, !dbg !75
  store i32 %sub267, ptr %c, align 4, !dbg !75
  %tmp268 = load i32, ptr %b, align 4, !dbg !75
  %tmp269 = load i32, ptr %c, align 4, !dbg !75
  %sub270 = sub i32 %tmp269, %tmp268, !dbg !75
  store i32 %sub270, ptr %c, align 4, !dbg !75
  %tmp271 = load i32, ptr %b, align 4, !dbg !75
  %shr272 = lshr i32 %tmp271, 13, !dbg !75
  %tmp273 = load i32, ptr %c, align 4, !dbg !75
  %xor274 = xor i32 %tmp273, %shr272, !dbg !75
  store i32 %xor274, ptr %c, align 4, !dbg !75
  %tmp275 = load i32, ptr %b, align 4, !dbg !75
  %tmp276 = load i32, ptr %a, align 4, !dbg !75
  %sub277 = sub i32 %tmp276, %tmp275, !dbg !75
  store i32 %sub277, ptr %a, align 4, !dbg !75
  %tmp278 = load i32, ptr %c, align 4, !dbg !75
  %tmp279 = load i32, ptr %a, align 4, !dbg !75
  %sub280 = sub i32 %tmp279, %tmp278, !dbg !75
  store i32 %sub280, ptr %a, align 4, !dbg !75
  %tmp281 = load i32, ptr %c, align 4, !dbg !75
  %shr282 = lshr i32 %tmp281, 12, !dbg !75
  %tmp283 = load i32, ptr %a, align 4, !dbg !75
  %xor284 = xor i32 %tmp283, %shr282, !dbg !75
  store i32 %xor284, ptr %a, align 4, !dbg !75
  %tmp285 = load i32, ptr %c, align 4, !dbg !75
  %tmp286 = load i32, ptr %b, align 4, !dbg !75
  %sub287 = sub i32 %tmp286, %tmp285, !dbg !75
  store i32 %sub287, ptr %b, align 4, !dbg !75
  %tmp288 = load i32, ptr %a, align 4, !dbg !75
  %tmp289 = load i32, ptr %b, align 4, !dbg !75
  %sub290 = sub i32 %tmp289, %tmp288, !dbg !75
  store i32 %sub290, ptr %b, align 4, !dbg !75
  %tmp291 = load i32, ptr %a, align 4, !dbg !75
  %shl292 = shl i32 %tmp291, 16, !dbg !75
  %tmp293 = load i32, ptr %b, align 4, !dbg !75
  %xor294 = xor i32 %tmp293, %shl292, !dbg !75
  store i32 %xor294, ptr %b, align 4, !dbg !75
  %tmp295 = load i32, ptr %a, align 4, !dbg !75
  %tmp296 = load i32, ptr %c, align 4, !dbg !75
  %sub297 = sub i32 %tmp296, %tmp295, !dbg !75
  store i32 %sub297, ptr %c, align 4, !dbg !75
  %tmp298 = load i32, ptr %b, align 4, !dbg !75
  %tmp299 = load i32, ptr %c, align 4, !dbg !75
  %sub300 = sub i32 %tmp299, %tmp298, !dbg !75
  store i32 %sub300, ptr %c, align 4, !dbg !75
  %tmp301 = load i32, ptr %b, align 4, !dbg !75
  %shr302 = lshr i32 %tmp301, 5, !dbg !75
  %tmp303 = load i32, ptr %c, align 4, !dbg !75
  %xor304 = xor i32 %tmp303, %shr302, !dbg !75
  store i32 %xor304, ptr %c, align 4, !dbg !75
  %tmp305 = load i32, ptr %b, align 4, !dbg !75
  %tmp306 = load i32, ptr %a, align 4, !dbg !75
  %sub307 = sub i32 %tmp306, %tmp305, !dbg !75
  store i32 %sub307, ptr %a, align 4, !dbg !75
  %tmp308 = load i32, ptr %c, align 4, !dbg !75
  %tmp309 = load i32, ptr %a, align 4, !dbg !75
  %sub310 = sub i32 %tmp309, %tmp308, !dbg !75
  store i32 %sub310, ptr %a, align 4, !dbg !75
  %tmp311 = load i32, ptr %c, align 4, !dbg !75
  %shr312 = lshr i32 %tmp311, 3, !dbg !75
  %tmp313 = load i32, ptr %a, align 4, !dbg !75
  %xor314 = xor i32 %tmp313, %shr312, !dbg !75
  store i32 %xor314, ptr %a, align 4, !dbg !75
  %tmp315 = load i32, ptr %c, align 4, !dbg !75
  %tmp316 = load i32, ptr %b, align 4, !dbg !75
  %sub317 = sub i32 %tmp316, %tmp315, !dbg !75
  store i32 %sub317, ptr %b, align 4, !dbg !75
  %tmp318 = load i32, ptr %a, align 4, !dbg !75
  %tmp319 = load i32, ptr %b, align 4, !dbg !75
  %sub320 = sub i32 %tmp319, %tmp318, !dbg !75
  store i32 %sub320, ptr %b, align 4, !dbg !75
  %tmp321 = load i32, ptr %a, align 4, !dbg !75
  %shl322 = shl i32 %tmp321, 10, !dbg !75
  %tmp323 = load i32, ptr %b, align 4, !dbg !75
  %xor324 = xor i32 %tmp323, %shl322, !dbg !75
  store i32 %xor324, ptr %b, align 4, !dbg !75
  %tmp325 = load i32, ptr %a, align 4, !dbg !75
  %tmp326 = load i32, ptr %c, align 4, !dbg !75
  %sub327 = sub i32 %tmp326, %tmp325, !dbg !75
  store i32 %sub327, ptr %c, align 4, !dbg !75
  %tmp328 = load i32, ptr %b, align 4, !dbg !75
  %tmp329 = load i32, ptr %c, align 4, !dbg !75
  %sub330 = sub i32 %tmp329, %tmp328, !dbg !75
  store i32 %sub330, ptr %c, align 4, !dbg !75
  %tmp331 = load i32, ptr %b, align 4, !dbg !75
  %shr332 = lshr i32 %tmp331, 15, !dbg !75
  %tmp333 = load i32, ptr %c, align 4, !dbg !75
  %xor334 = xor i32 %tmp333, %shr332, !dbg !75
  store i32 %xor334, ptr %c, align 4, !dbg !75
  %tmp335 = load i32, ptr %c, align 4, !dbg !76
  store i32 %tmp335, ptr %retval, align 4, !dbg !76
  %20 = load i32, ptr %retval, align 4, !dbg !76
  ret i32 %20, !dbg !76
}

attributes #0 = { alwaysinline inlinehint }

!llvm.dbg.cu = !{!0}
!nvvmir.version = !{!3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "lgenfe: EDG 6.7", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, enums: !2)
!1 = !DIFile(filename: "bobhash32.cpp4.ii", directory: "/home/redp/disc/src/cuda-ptx/src/denvdis/c-sketch")
!2 = !{}
!3 = !{i32 2, i32 0, i32 3, i32 2}
!4 = !{i32 1, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "fmix32", linkageName: "_Z6fmix32j", scope: !6, file: !6, line: 10, type: !7, scopeLine: 10, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!6 = !DIFile(filename: "bobhash32.cu", directory: "/home/redp/disc/src/cuda-ptx/src/denvdis/c-sketch")
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 12, column: 3, scope: !9)
!9 = distinct !DILexicalBlock(scope: !5, file: !6, line: 11, column: 1)
!10 = !DILocation(line: 13, column: 3, scope: !9)
!11 = !DILocation(line: 14, column: 3, scope: !9)
!12 = !DILocation(line: 15, column: 3, scope: !9)
!13 = !DILocation(line: 16, column: 3, scope: !9)
!14 = !DILocation(line: 18, column: 3, scope: !9)
!15 = distinct !DISubprogram(name: "MurmurHash3_x86_32", linkageName: "_Z18MurmurHash3_x86_32PKvij", scope: !6, file: !6, line: 21, type: !7, scopeLine: 21, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!16 = !DILocation(line: 23, column: 3, scope: !17)
!17 = distinct !DILexicalBlock(scope: !15, file: !6, line: 22, column: 1)
!18 = !DILocation(line: 24, column: 3, scope: !17)
!19 = !DILocation(line: 26, column: 3, scope: !17)
!20 = !DILocation(line: 28, column: 3, scope: !17)
!21 = !DILocation(line: 29, column: 3, scope: !17)
!22 = !DILocation(line: 34, column: 3, scope: !17)
!23 = !DILocation(line: 36, column: 3, scope: !17)
!24 = !DILocation(line: 38, column: 5, scope: !25)
!25 = distinct !DILexicalBlock(scope: !26, file: !6, line: 37, column: 3)
!26 = distinct !DILexicalBlock(scope: !27, file: !6, line: 36, column: 3)
!27 = distinct !DILexicalBlock(scope: !17, file: !6, line: 36, column: 3)
!28 = !DILocation(line: 40, column: 5, scope: !25)
!29 = !DILocation(line: 41, column: 5, scope: !25)
!30 = !DILocation(line: 42, column: 5, scope: !25)
!31 = !DILocation(line: 43, column: 5, scope: !25)
!32 = !DILocation(line: 44, column: 5, scope: !25)
!33 = !DILocation(line: 45, column: 5, scope: !25)
!34 = !DILocation(line: 36, column: 28, scope: !26)
!35 = !DILocation(line: 51, column: 3, scope: !17)
!36 = !DILocation(line: 53, column: 3, scope: !17)
!37 = !DILocation(line: 55, column: 3, scope: !17)
!38 = !DILocation(line: 57, column: 3, scope: !39)
!39 = distinct !DILexicalBlock(scope: !17, file: !6, line: 56, column: 3)
!40 = !DILocation(line: 58, column: 3, scope: !39)
!41 = !DILocation(line: 59, column: 3, scope: !39)
!42 = !DILocation(line: 60, column: 11, scope: !39)
!43 = !DILocation(line: 66, column: 3, scope: !17)
!44 = !DILocation(line: 68, column: 3, scope: !17)
!45 = !DILocation(line: 70, column: 3, scope: !17)
!46 = distinct !DISubprogram(name: "BOBHash32", linkageName: "_Z9BOBHash32PKcjj", scope: !6, file: !6, line: 218, type: !7, scopeLine: 218, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!47 = !DILocation(line: 222, column: 5, scope: !48)
!48 = distinct !DILexicalBlock(scope: !46, file: !6, line: 218, column: 25)
!49 = !DILocation(line: 223, column: 5, scope: !48)
!50 = !DILocation(line: 224, column: 6, scope: !51)
!51 = distinct !DILexicalBlock(scope: !48, file: !6, line: 223, column: 5)
!52 = !DILocation(line: 226, column: 6, scope: !53)
!53 = distinct !DILexicalBlock(scope: !48, file: !6, line: 224, column: 6)
!54 = !DILocation(line: 229, column: 5, scope: !48)
!55 = !DILocation(line: 231, column: 2, scope: !56)
!56 = distinct !DILexicalBlock(scope: !48, file: !6, line: 230, column: 5)
!57 = !DILocation(line: 232, column: 2, scope: !56)
!58 = !DILocation(line: 233, column: 2, scope: !56)
!59 = !DILocation(line: 234, column: 2, scope: !56)
!60 = !DILocation(line: 235, column: 2, scope: !56)
!61 = !DILocation(line: 239, column: 5, scope: !48)
!62 = !DILocation(line: 240, column: 5, scope: !48)
!63 = !DILocation(line: 242, column: 2, scope: !64)
!64 = distinct !DILexicalBlock(scope: !48, file: !6, line: 241, column: 5)
!65 = !DILocation(line: 243, column: 2, scope: !64)
!66 = !DILocation(line: 244, column: 2, scope: !64)
!67 = !DILocation(line: 246, column: 2, scope: !64)
!68 = !DILocation(line: 247, column: 2, scope: !64)
!69 = !DILocation(line: 248, column: 2, scope: !64)
!70 = !DILocation(line: 249, column: 2, scope: !64)
!71 = !DILocation(line: 250, column: 2, scope: !64)
!72 = !DILocation(line: 251, column: 2, scope: !64)
!73 = !DILocation(line: 252, column: 2, scope: !64)
!74 = !DILocation(line: 253, column: 2, scope: !64)
!75 = !DILocation(line: 256, column: 5, scope: !48)
!76 = !DILocation(line: 258, column: 5, scope: !48)
