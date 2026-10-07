; ModuleID = 'bobhash32.cu'
source_filename = "bobhash32.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@prime32 = dso_local addrspace(4) externally_initialized constant [1229 x i32] [i32 2, i32 3, i32 5, i32 7, i32 11, i32 13, i32 17, i32 19, i32 23, i32 29, i32 31, i32 37, i32 41, i32 43, i32 47, i32 53, i32 59, i32 61, i32 67, i32 71, i32 73, i32 79, i32 83, i32 89, i32 97, i32 101, i32 103, i32 107, i32 109, i32 113, i32 127, i32 131, i32 137, i32 139, i32 149, i32 151, i32 157, i32 163, i32 167, i32 173, i32 179, i32 181, i32 191, i32 193, i32 197, i32 199, i32 211, i32 223, i32 227, i32 229, i32 233, i32 239, i32 241, i32 251, i32 257, i32 263, i32 269, i32 271, i32 277, i32 281, i32 283, i32 293, i32 307, i32 311, i32 313, i32 317, i32 331, i32 337, i32 347, i32 349, i32 353, i32 359, i32 367, i32 373, i32 379, i32 383, i32 389, i32 397, i32 401, i32 409, i32 419, i32 421, i32 431, i32 433, i32 439, i32 443, i32 449, i32 457, i32 461, i32 463, i32 467, i32 479, i32 487, i32 491, i32 499, i32 503, i32 509, i32 521, i32 523, i32 541, i32 547, i32 557, i32 563, i32 569, i32 571, i32 577, i32 587, i32 593, i32 599, i32 601, i32 607, i32 613, i32 617, i32 619, i32 631, i32 641, i32 643, i32 647, i32 653, i32 659, i32 661, i32 673, i32 677, i32 683, i32 691, i32 701, i32 709, i32 719, i32 727, i32 733, i32 739, i32 743, i32 751, i32 757, i32 761, i32 769, i32 773, i32 787, i32 797, i32 809, i32 811, i32 821, i32 823, i32 827, i32 829, i32 839, i32 853, i32 857, i32 859, i32 863, i32 877, i32 881, i32 883, i32 887, i32 907, i32 911, i32 919, i32 929, i32 937, i32 941, i32 947, i32 953, i32 967, i32 971, i32 977, i32 983, i32 991, i32 997, i32 1009, i32 1013, i32 1019, i32 1021, i32 1031, i32 1033, i32 1039, i32 1049, i32 1051, i32 1061, i32 1063, i32 1069, i32 1087, i32 1091, i32 1093, i32 1097, i32 1103, i32 1109, i32 1117, i32 1123, i32 1129, i32 1151, i32 1153, i32 1163, i32 1171, i32 1181, i32 1187, i32 1193, i32 1201, i32 1213, i32 1217, i32 1223, i32 1229, i32 1231, i32 1237, i32 1249, i32 1259, i32 1277, i32 1279, i32 1283, i32 1289, i32 1291, i32 1297, i32 1301, i32 1303, i32 1307, i32 1319, i32 1321, i32 1327, i32 1361, i32 1367, i32 1373, i32 1381, i32 1399, i32 1409, i32 1423, i32 1427, i32 1429, i32 1433, i32 1439, i32 1447, i32 1451, i32 1453, i32 1459, i32 1471, i32 1481, i32 1483, i32 1487, i32 1489, i32 1493, i32 1499, i32 1511, i32 1523, i32 1531, i32 1543, i32 1549, i32 1553, i32 1559, i32 1567, i32 1571, i32 1579, i32 1583, i32 1597, i32 1601, i32 1607, i32 1609, i32 1613, i32 1619, i32 1621, i32 1627, i32 1637, i32 1657, i32 1663, i32 1667, i32 1669, i32 1693, i32 1697, i32 1699, i32 1709, i32 1721, i32 1723, i32 1733, i32 1741, i32 1747, i32 1753, i32 1759, i32 1777, i32 1783, i32 1787, i32 1789, i32 1801, i32 1811, i32 1823, i32 1831, i32 1847, i32 1861, i32 1867, i32 1871, i32 1873, i32 1877, i32 1879, i32 1889, i32 1901, i32 1907, i32 1913, i32 1931, i32 1933, i32 1949, i32 1951, i32 1973, i32 1979, i32 1987, i32 1993, i32 1997, i32 1999, i32 2003, i32 2011, i32 2017, i32 2027, i32 2029, i32 2039, i32 2053, i32 2063, i32 2069, i32 2081, i32 2083, i32 2087, i32 2089, i32 2099, i32 2111, i32 2113, i32 2129, i32 2131, i32 2137, i32 2141, i32 2143, i32 2153, i32 2161, i32 2179, i32 2203, i32 2207, i32 2213, i32 2221, i32 2237, i32 2239, i32 2243, i32 2251, i32 2267, i32 2269, i32 2273, i32 2281, i32 2287, i32 2293, i32 2297, i32 2309, i32 2311, i32 2333, i32 2339, i32 2341, i32 2347, i32 2351, i32 2357, i32 2371, i32 2377, i32 2381, i32 2383, i32 2389, i32 2393, i32 2399, i32 2411, i32 2417, i32 2423, i32 2437, i32 2441, i32 2447, i32 2459, i32 2467, i32 2473, i32 2477, i32 2503, i32 2521, i32 2531, i32 2539, i32 2543, i32 2549, i32 2551, i32 2557, i32 2579, i32 2591, i32 2593, i32 2609, i32 2617, i32 2621, i32 2633, i32 2647, i32 2657, i32 2659, i32 2663, i32 2671, i32 2677, i32 2683, i32 2687, i32 2689, i32 2693, i32 2699, i32 2707, i32 2711, i32 2713, i32 2719, i32 2729, i32 2731, i32 2741, i32 2749, i32 2753, i32 2767, i32 2777, i32 2789, i32 2791, i32 2797, i32 2801, i32 2803, i32 2819, i32 2833, i32 2837, i32 2843, i32 2851, i32 2857, i32 2861, i32 2879, i32 2887, i32 2897, i32 2903, i32 2909, i32 2917, i32 2927, i32 2939, i32 2953, i32 2957, i32 2963, i32 2969, i32 2971, i32 2999, i32 3001, i32 3011, i32 3019, i32 3023, i32 3037, i32 3041, i32 3049, i32 3061, i32 3067, i32 3079, i32 3083, i32 3089, i32 3109, i32 3119, i32 3121, i32 3137, i32 3163, i32 3167, i32 3169, i32 3181, i32 3187, i32 3191, i32 3203, i32 3209, i32 3217, i32 3221, i32 3229, i32 3251, i32 3253, i32 3257, i32 3259, i32 3271, i32 3299, i32 3301, i32 3307, i32 3313, i32 3319, i32 3323, i32 3329, i32 3331, i32 3343, i32 3347, i32 3359, i32 3361, i32 3371, i32 3373, i32 3389, i32 3391, i32 3407, i32 3413, i32 3433, i32 3449, i32 3457, i32 3461, i32 3463, i32 3467, i32 3469, i32 3491, i32 3499, i32 3511, i32 3517, i32 3527, i32 3529, i32 3533, i32 3539, i32 3541, i32 3547, i32 3557, i32 3559, i32 3571, i32 3581, i32 3583, i32 3593, i32 3607, i32 3613, i32 3617, i32 3623, i32 3631, i32 3637, i32 3643, i32 3659, i32 3671, i32 3673, i32 3677, i32 3691, i32 3697, i32 3701, i32 3709, i32 3719, i32 3727, i32 3733, i32 3739, i32 3761, i32 3767, i32 3769, i32 3779, i32 3793, i32 3797, i32 3803, i32 3821, i32 3823, i32 3833, i32 3847, i32 3851, i32 3853, i32 3863, i32 3877, i32 3881, i32 3889, i32 3907, i32 3911, i32 3917, i32 3919, i32 3923, i32 3929, i32 3931, i32 3943, i32 3947, i32 3967, i32 3989, i32 4001, i32 4003, i32 4007, i32 4013, i32 4019, i32 4021, i32 4027, i32 4049, i32 4051, i32 4057, i32 4073, i32 4079, i32 4091, i32 4093, i32 4099, i32 4111, i32 4127, i32 4129, i32 4133, i32 4139, i32 4153, i32 4157, i32 4159, i32 4177, i32 4201, i32 4211, i32 4217, i32 4219, i32 4229, i32 4231, i32 4241, i32 4243, i32 4253, i32 4259, i32 4261, i32 4271, i32 4273, i32 4283, i32 4289, i32 4297, i32 4327, i32 4337, i32 4339, i32 4349, i32 4357, i32 4363, i32 4373, i32 4391, i32 4397, i32 4409, i32 4421, i32 4423, i32 4441, i32 4447, i32 4451, i32 4457, i32 4463, i32 4481, i32 4483, i32 4493, i32 4507, i32 4513, i32 4517, i32 4519, i32 4523, i32 4547, i32 4549, i32 4561, i32 4567, i32 4583, i32 4591, i32 4597, i32 4603, i32 4621, i32 4637, i32 4639, i32 4643, i32 4649, i32 4651, i32 4657, i32 4663, i32 4673, i32 4679, i32 4691, i32 4703, i32 4721, i32 4723, i32 4729, i32 4733, i32 4751, i32 4759, i32 4783, i32 4787, i32 4789, i32 4793, i32 4799, i32 4801, i32 4813, i32 4817, i32 4831, i32 4861, i32 4871, i32 4877, i32 4889, i32 4903, i32 4909, i32 4919, i32 4931, i32 4933, i32 4937, i32 4943, i32 4951, i32 4957, i32 4967, i32 4969, i32 4973, i32 4987, i32 4993, i32 4999, i32 5003, i32 5009, i32 5011, i32 5021, i32 5023, i32 5039, i32 5051, i32 5059, i32 5077, i32 5081, i32 5087, i32 5099, i32 5101, i32 5107, i32 5113, i32 5119, i32 5147, i32 5153, i32 5167, i32 5171, i32 5179, i32 5189, i32 5197, i32 5209, i32 5227, i32 5231, i32 5233, i32 5237, i32 5261, i32 5273, i32 5279, i32 5281, i32 5297, i32 5303, i32 5309, i32 5323, i32 5333, i32 5347, i32 5351, i32 5381, i32 5387, i32 5393, i32 5399, i32 5407, i32 5413, i32 5417, i32 5419, i32 5431, i32 5437, i32 5441, i32 5443, i32 5449, i32 5471, i32 5477, i32 5479, i32 5483, i32 5501, i32 5503, i32 5507, i32 5519, i32 5521, i32 5527, i32 5531, i32 5557, i32 5563, i32 5569, i32 5573, i32 5581, i32 5591, i32 5623, i32 5639, i32 5641, i32 5647, i32 5651, i32 5653, i32 5657, i32 5659, i32 5669, i32 5683, i32 5689, i32 5693, i32 5701, i32 5711, i32 5717, i32 5737, i32 5741, i32 5743, i32 5749, i32 5779, i32 5783, i32 5791, i32 5801, i32 5807, i32 5813, i32 5821, i32 5827, i32 5839, i32 5843, i32 5849, i32 5851, i32 5857, i32 5861, i32 5867, i32 5869, i32 5879, i32 5881, i32 5897, i32 5903, i32 5923, i32 5927, i32 5939, i32 5953, i32 5981, i32 5987, i32 6007, i32 6011, i32 6029, i32 6037, i32 6043, i32 6047, i32 6053, i32 6067, i32 6073, i32 6079, i32 6089, i32 6091, i32 6101, i32 6113, i32 6121, i32 6131, i32 6133, i32 6143, i32 6151, i32 6163, i32 6173, i32 6197, i32 6199, i32 6203, i32 6211, i32 6217, i32 6221, i32 6229, i32 6247, i32 6257, i32 6263, i32 6269, i32 6271, i32 6277, i32 6287, i32 6299, i32 6301, i32 6311, i32 6317, i32 6323, i32 6329, i32 6337, i32 6343, i32 6353, i32 6359, i32 6361, i32 6367, i32 6373, i32 6379, i32 6389, i32 6397, i32 6421, i32 6427, i32 6449, i32 6451, i32 6469, i32 6473, i32 6481, i32 6491, i32 6521, i32 6529, i32 6547, i32 6551, i32 6553, i32 6563, i32 6569, i32 6571, i32 6577, i32 6581, i32 6599, i32 6607, i32 6619, i32 6637, i32 6653, i32 6659, i32 6661, i32 6673, i32 6679, i32 6689, i32 6691, i32 6701, i32 6703, i32 6709, i32 6719, i32 6733, i32 6737, i32 6761, i32 6763, i32 6779, i32 6781, i32 6791, i32 6793, i32 6803, i32 6823, i32 6827, i32 6829, i32 6833, i32 6841, i32 6857, i32 6863, i32 6869, i32 6871, i32 6883, i32 6899, i32 6907, i32 6911, i32 6917, i32 6947, i32 6949, i32 6959, i32 6961, i32 6967, i32 6971, i32 6977, i32 6983, i32 6991, i32 6997, i32 7001, i32 7013, i32 7019, i32 7027, i32 7039, i32 7043, i32 7057, i32 7069, i32 7079, i32 7103, i32 7109, i32 7121, i32 7127, i32 7129, i32 7151, i32 7159, i32 7177, i32 7187, i32 7193, i32 7207, i32 7211, i32 7213, i32 7219, i32 7229, i32 7237, i32 7243, i32 7247, i32 7253, i32 7283, i32 7297, i32 7307, i32 7309, i32 7321, i32 7331, i32 7333, i32 7349, i32 7351, i32 7369, i32 7393, i32 7411, i32 7417, i32 7433, i32 7451, i32 7457, i32 7459, i32 7477, i32 7481, i32 7487, i32 7489, i32 7499, i32 7507, i32 7517, i32 7523, i32 7529, i32 7537, i32 7541, i32 7547, i32 7549, i32 7559, i32 7561, i32 7573, i32 7577, i32 7583, i32 7589, i32 7591, i32 7603, i32 7607, i32 7621, i32 7639, i32 7643, i32 7649, i32 7669, i32 7673, i32 7681, i32 7687, i32 7691, i32 7699, i32 7703, i32 7717, i32 7723, i32 7727, i32 7741, i32 7753, i32 7757, i32 7759, i32 7789, i32 7793, i32 7817, i32 7823, i32 7829, i32 7841, i32 7853, i32 7867, i32 7873, i32 7877, i32 7879, i32 7883, i32 7901, i32 7907, i32 7919, i32 7927, i32 7933, i32 7937, i32 7949, i32 7951, i32 7963, i32 7993, i32 8009, i32 8011, i32 8017, i32 8039, i32 8053, i32 8059, i32 8069, i32 8081, i32 8087, i32 8089, i32 8093, i32 8101, i32 8111, i32 8117, i32 8123, i32 8147, i32 8161, i32 8167, i32 8171, i32 8179, i32 8191, i32 8209, i32 8219, i32 8221, i32 8231, i32 8233, i32 8237, i32 8243, i32 8263, i32 8269, i32 8273, i32 8287, i32 8291, i32 8293, i32 8297, i32 8311, i32 8317, i32 8329, i32 8353, i32 8363, i32 8369, i32 8377, i32 8387, i32 8389, i32 8419, i32 8423, i32 8429, i32 8431, i32 8443, i32 8447, i32 8461, i32 8467, i32 8501, i32 8513, i32 8521, i32 8527, i32 8537, i32 8539, i32 8543, i32 8563, i32 8573, i32 8581, i32 8597, i32 8599, i32 8609, i32 8623, i32 8627, i32 8629, i32 8641, i32 8647, i32 8663, i32 8669, i32 8677, i32 8681, i32 8689, i32 8693, i32 8699, i32 8707, i32 8713, i32 8719, i32 8731, i32 8737, i32 8741, i32 8747, i32 8753, i32 8761, i32 8779, i32 8783, i32 8803, i32 8807, i32 8819, i32 8821, i32 8831, i32 8837, i32 8839, i32 8849, i32 8861, i32 8863, i32 8867, i32 8887, i32 8893, i32 8923, i32 8929, i32 8933, i32 8941, i32 8951, i32 8963, i32 8969, i32 8971, i32 8999, i32 9001, i32 9007, i32 9011, i32 9013, i32 9029, i32 9041, i32 9043, i32 9049, i32 9059, i32 9067, i32 9091, i32 9103, i32 9109, i32 9127, i32 9133, i32 9137, i32 9151, i32 9157, i32 9161, i32 9173, i32 9181, i32 9187, i32 9199, i32 9203, i32 9209, i32 9221, i32 9227, i32 9239, i32 9241, i32 9257, i32 9277, i32 9281, i32 9283, i32 9293, i32 9311, i32 9319, i32 9323, i32 9337, i32 9341, i32 9343, i32 9349, i32 9371, i32 9377, i32 9391, i32 9397, i32 9403, i32 9413, i32 9419, i32 9421, i32 9431, i32 9433, i32 9437, i32 9439, i32 9461, i32 9463, i32 9467, i32 9473, i32 9479, i32 9491, i32 9497, i32 9511, i32 9521, i32 9533, i32 9539, i32 9547, i32 9551, i32 9587, i32 9601, i32 9613, i32 9619, i32 9623, i32 9629, i32 9631, i32 9643, i32 9649, i32 9661, i32 9677, i32 9679, i32 9689, i32 9697, i32 9719, i32 9721, i32 9733, i32 9739, i32 9743, i32 9749, i32 9767, i32 9769, i32 9781, i32 9787, i32 9791, i32 9803, i32 9811, i32 9817, i32 9829, i32 9833, i32 9839, i32 9851, i32 9857, i32 9859, i32 9871, i32 9883, i32 9887, i32 9901, i32 9907, i32 9923, i32 9929, i32 9931, i32 9941, i32 9949, i32 9967, i32 9973], align 4
@llvm.compiler.used = appending global [1 x ptr] [ptr addrspacecast (ptr addrspace(4) @prime32 to ptr)], section "llvm.metadata"

; Function Attrs: mustprogress nounwind
define weak dso_local i32 @cudaMalloc(ptr noundef %0, i64 noundef %1) local_unnamed_addr #0 {
  ret i32 999
}

; Function Attrs: mustprogress nounwind
define weak dso_local i32 @cudaFuncGetAttributes(ptr noundef %0, ptr noundef %1) local_unnamed_addr #0 {
  ret i32 999
}

; Function Attrs: mustprogress nounwind
define weak dso_local i32 @cudaDeviceGetAttribute(ptr noundef %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  ret i32 999
}

; Function Attrs: mustprogress nounwind
define weak dso_local i32 @cudaGetDevice(ptr noundef %0) local_unnamed_addr #0 {
  ret i32 999
}

; Function Attrs: mustprogress nounwind
define weak dso_local i32 @cudaOccupancyMaxActiveBlocksPerMultiprocessor(ptr noundef %0, ptr noundef %1, i32 noundef %2, i64 noundef %3) local_unnamed_addr #0 {
  ret i32 999
}

; Function Attrs: mustprogress nounwind
define weak dso_local i32 @cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(ptr noundef %0, ptr noundef %1, i32 noundef %2, i64 noundef %3, i32 noundef %4) local_unnamed_addr #0 {
  ret i32 999
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read)
define dso_local noundef i32 @_Z18MurmurHash3_x86_32PKvij(ptr noundef readonly captures(none) %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr #1 {
  %4 = sdiv i32 %1, 4
  %5 = shl nsw i32 %4, 2
  %6 = sext i32 %5 to i64
  %7 = getelementptr inbounds i8, ptr %0, i64 %6
  %8 = add i32 %1, 3
  %9 = icmp ult i32 %8, 7
  br i1 %9, label %12, label %10

10:                                               ; preds = %3
  %11 = sub nsw i32 0, %4
  br label %15

12:                                               ; preds = %15, %3
  %13 = phi i32 [ %2, %3 ], [ %29, %15 ]
  %14 = and i32 %1, 3
  switch i32 %14, label %55 [
    i32 3, label %32
    i32 2, label %37
    i32 1, label %44
    i32 0, label %56
  ]

15:                                               ; preds = %10, %15
  %16 = phi i32 [ %29, %15 ], [ %2, %10 ]
  %17 = phi i32 [ %30, %15 ], [ %11, %10 ]
  %18 = sext i32 %17 to i64
  %19 = getelementptr inbounds i32, ptr %7, i64 %18
  %20 = load i32, ptr %19, align 4, !tbaa !9
  %21 = mul i32 %20, -862048943
  %22 = mul i32 %20, 380141568
  %23 = lshr i32 %21, 17
  %24 = or disjoint i32 %23, %22
  %25 = mul i32 %24, 461845907
  %26 = xor i32 %25, %16
  %27 = tail call i32 @llvm.fshl.i32(i32 %26, i32 %26, i32 13)
  %28 = mul i32 %27, 5
  %29 = add i32 %28, -430675100
  %30 = add nsw i32 %17, 1
  %31 = icmp eq i32 %30, 0
  br i1 %31, label %12, label %15, !llvm.loop !13

32:                                               ; preds = %12
  %33 = getelementptr inbounds nuw i8, ptr %7, i64 2
  %34 = load i8, ptr %33, align 1, !tbaa !16
  %35 = zext i8 %34 to i32
  %36 = shl nuw nsw i32 %35, 16
  br label %37

37:                                               ; preds = %12, %32
  %38 = phi i32 [ %36, %32 ], [ 0, %12 ]
  %39 = getelementptr inbounds nuw i8, ptr %7, i64 1
  %40 = load i8, ptr %39, align 1, !tbaa !16
  %41 = zext i8 %40 to i32
  %42 = shl nuw nsw i32 %41, 8
  %43 = or disjoint i32 %42, %38
  br label %44

44:                                               ; preds = %12, %37
  %45 = phi i32 [ %43, %37 ], [ 0, %12 ]
  %46 = load i8, ptr %7, align 1, !tbaa !16
  %47 = zext i8 %46 to i32
  %48 = xor i32 %45, %47
  %49 = mul i32 %48, -862048943
  %50 = mul i32 %48, 380141568
  %51 = lshr i32 %49, 17
  %52 = or disjoint i32 %51, %50
  %53 = mul i32 %52, 461845907
  %54 = xor i32 %53, %13
  br label %56

55:                                               ; preds = %12
  unreachable

56:                                               ; preds = %12, %44
  %57 = phi i32 [ %54, %44 ], [ %13, %12 ]
  %58 = xor i32 %57, %1
  %59 = lshr i32 %58, 16
  %60 = xor i32 %59, %58
  %61 = mul i32 %60, -2048144789
  %62 = lshr i32 %61, 13
  %63 = xor i32 %62, %61
  %64 = mul i32 %63, -1028477387
  %65 = lshr i32 %64, 16
  %66 = xor i32 %65, %64
  ret i32 %66
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read)
define dso_local noundef i32 @_Z9BOBHash32PKcjj(ptr noundef readonly captures(none) %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr #1 {
  %4 = icmp ult i32 %2, 1229
  br i1 %4, label %5, label %9

5:                                                ; preds = %3
  %6 = zext nneg i32 %2 to i64
  %7 = getelementptr inbounds nuw [1229 x i32], ptr addrspacecast (ptr addrspace(4) @prime32 to ptr), i64 0, i64 %6
  %8 = load i32, ptr %7, align 4, !tbaa !9
  br label %9

9:                                                ; preds = %3, %5
  %10 = phi i32 [ %8, %5 ], [ %2, %3 ]
  %11 = icmp ugt i32 %1, 11
  br i1 %11, label %12, label %113

12:                                               ; preds = %9, %12
  %13 = phi i32 [ %109, %12 ], [ %10, %9 ]
  %14 = phi i32 [ %105, %12 ], [ -1640531527, %9 ]
  %15 = phi i32 [ %101, %12 ], [ -1640531527, %9 ]
  %16 = phi ptr [ %110, %12 ], [ %0, %9 ]
  %17 = phi i32 [ %111, %12 ], [ %1, %9 ]
  %18 = load i8, ptr %16, align 1, !tbaa !16
  %19 = sext i8 %18 to i32
  %20 = getelementptr inbounds nuw i8, ptr %16, i64 1
  %21 = load i8, ptr %20, align 1, !tbaa !16
  %22 = sext i8 %21 to i32
  %23 = shl nsw i32 %22, 8
  %24 = getelementptr inbounds nuw i8, ptr %16, i64 2
  %25 = load i8, ptr %24, align 1, !tbaa !16
  %26 = sext i8 %25 to i32
  %27 = shl nsw i32 %26, 16
  %28 = getelementptr inbounds nuw i8, ptr %16, i64 3
  %29 = load i8, ptr %28, align 1, !tbaa !16
  %30 = sext i8 %29 to i32
  %31 = shl nsw i32 %30, 24
  %32 = getelementptr inbounds nuw i8, ptr %16, i64 4
  %33 = load i8, ptr %32, align 1, !tbaa !16
  %34 = sext i8 %33 to i32
  %35 = getelementptr inbounds nuw i8, ptr %16, i64 5
  %36 = load i8, ptr %35, align 1, !tbaa !16
  %37 = sext i8 %36 to i32
  %38 = shl nsw i32 %37, 8
  %39 = getelementptr inbounds nuw i8, ptr %16, i64 6
  %40 = load i8, ptr %39, align 1, !tbaa !16
  %41 = sext i8 %40 to i32
  %42 = shl nsw i32 %41, 16
  %43 = getelementptr inbounds nuw i8, ptr %16, i64 7
  %44 = load i8, ptr %43, align 1, !tbaa !16
  %45 = sext i8 %44 to i32
  %46 = shl nsw i32 %45, 24
  %47 = add i32 %14, %34
  %48 = add i32 %47, %38
  %49 = add i32 %48, %42
  %50 = add i32 %49, %46
  %51 = getelementptr inbounds nuw i8, ptr %16, i64 8
  %52 = load i8, ptr %51, align 1, !tbaa !16
  %53 = sext i8 %52 to i32
  %54 = getelementptr inbounds nuw i8, ptr %16, i64 9
  %55 = load i8, ptr %54, align 1, !tbaa !16
  %56 = sext i8 %55 to i32
  %57 = shl nsw i32 %56, 8
  %58 = getelementptr inbounds nuw i8, ptr %16, i64 10
  %59 = load i8, ptr %58, align 1, !tbaa !16
  %60 = sext i8 %59 to i32
  %61 = shl nsw i32 %60, 16
  %62 = getelementptr inbounds nuw i8, ptr %16, i64 11
  %63 = load i8, ptr %62, align 1, !tbaa !16
  %64 = sext i8 %63 to i32
  %65 = shl nsw i32 %64, 24
  %66 = add i32 %13, %53
  %67 = add i32 %66, %57
  %68 = add i32 %67, %61
  %69 = add i32 %68, %65
  %70 = add i32 %15, %19
  %71 = add i32 %70, %23
  %72 = add i32 %71, %27
  %73 = add i32 %72, %31
  %74 = add i32 %50, %69
  %75 = sub i32 %73, %74
  %76 = lshr i32 %69, 13
  %77 = xor i32 %75, %76
  %78 = add i32 %69, %77
  %79 = sub i32 %50, %78
  %80 = shl i32 %77, 8
  %81 = xor i32 %79, %80
  %82 = add i32 %77, %81
  %83 = sub i32 %69, %82
  %84 = lshr i32 %81, 13
  %85 = xor i32 %83, %84
  %86 = add i32 %81, %85
  %87 = sub i32 %77, %86
  %88 = lshr i32 %85, 12
  %89 = xor i32 %87, %88
  %90 = add i32 %85, %89
  %91 = sub i32 %81, %90
  %92 = shl i32 %89, 16
  %93 = xor i32 %91, %92
  %94 = add i32 %89, %93
  %95 = sub i32 %85, %94
  %96 = lshr i32 %93, 5
  %97 = xor i32 %95, %96
  %98 = add i32 %93, %97
  %99 = sub i32 %89, %98
  %100 = lshr i32 %97, 3
  %101 = xor i32 %99, %100
  %102 = add i32 %97, %101
  %103 = sub i32 %93, %102
  %104 = shl i32 %101, 10
  %105 = xor i32 %103, %104
  %106 = add i32 %101, %105
  %107 = sub i32 %97, %106
  %108 = lshr i32 %105, 15
  %109 = xor i32 %107, %108
  %110 = getelementptr inbounds nuw i8, ptr %16, i64 12
  %111 = add i32 %17, -12
  %112 = icmp ugt i32 %111, 11
  br i1 %112, label %12, label %113, !llvm.loop !17

113:                                              ; preds = %12, %9
  %114 = phi i32 [ %1, %9 ], [ %111, %12 ]
  %115 = phi ptr [ %0, %9 ], [ %110, %12 ]
  %116 = phi i32 [ -1640531527, %9 ], [ %101, %12 ]
  %117 = phi i32 [ -1640531527, %9 ], [ %105, %12 ]
  %118 = phi i32 [ %10, %9 ], [ %109, %12 ]
  %119 = add i32 %118, %114
  switch i32 %114, label %203 [
    i32 11, label %120
    i32 10, label %126
    i32 9, label %133
    i32 8, label %140
    i32 7, label %147
    i32 6, label %155
    i32 5, label %163
    i32 4, label %170
    i32 3, label %178
    i32 2, label %187
    i32 1, label %196
  ]

120:                                              ; preds = %113
  %121 = getelementptr inbounds nuw i8, ptr %115, i64 10
  %122 = load i8, ptr %121, align 1, !tbaa !16
  %123 = sext i8 %122 to i32
  %124 = shl nsw i32 %123, 24
  %125 = add i32 %124, %119
  br label %126

126:                                              ; preds = %113, %120
  %127 = phi i32 [ %125, %120 ], [ %119, %113 ]
  %128 = getelementptr inbounds nuw i8, ptr %115, i64 9
  %129 = load i8, ptr %128, align 1, !tbaa !16
  %130 = sext i8 %129 to i32
  %131 = shl nsw i32 %130, 16
  %132 = add i32 %131, %127
  br label %133

133:                                              ; preds = %113, %126
  %134 = phi i32 [ %132, %126 ], [ %119, %113 ]
  %135 = getelementptr inbounds nuw i8, ptr %115, i64 8
  %136 = load i8, ptr %135, align 1, !tbaa !16
  %137 = sext i8 %136 to i32
  %138 = shl nsw i32 %137, 8
  %139 = add i32 %138, %134
  br label %140

140:                                              ; preds = %113, %133
  %141 = phi i32 [ %139, %133 ], [ %119, %113 ]
  %142 = getelementptr inbounds nuw i8, ptr %115, i64 7
  %143 = load i8, ptr %142, align 1, !tbaa !16
  %144 = sext i8 %143 to i32
  %145 = shl nsw i32 %144, 24
  %146 = add i32 %145, %117
  br label %147

147:                                              ; preds = %113, %140
  %148 = phi i32 [ %146, %140 ], [ %117, %113 ]
  %149 = phi i32 [ %141, %140 ], [ %119, %113 ]
  %150 = getelementptr inbounds nuw i8, ptr %115, i64 6
  %151 = load i8, ptr %150, align 1, !tbaa !16
  %152 = sext i8 %151 to i32
  %153 = shl nsw i32 %152, 16
  %154 = add i32 %153, %148
  br label %155

155:                                              ; preds = %113, %147
  %156 = phi i32 [ %154, %147 ], [ %117, %113 ]
  %157 = phi i32 [ %149, %147 ], [ %119, %113 ]
  %158 = getelementptr inbounds nuw i8, ptr %115, i64 5
  %159 = load i8, ptr %158, align 1, !tbaa !16
  %160 = sext i8 %159 to i32
  %161 = shl nsw i32 %160, 8
  %162 = add i32 %161, %156
  br label %163

163:                                              ; preds = %113, %155
  %164 = phi i32 [ %162, %155 ], [ %117, %113 ]
  %165 = phi i32 [ %157, %155 ], [ %119, %113 ]
  %166 = getelementptr inbounds nuw i8, ptr %115, i64 4
  %167 = load i8, ptr %166, align 1, !tbaa !16
  %168 = sext i8 %167 to i32
  %169 = add i32 %164, %168
  br label %170

170:                                              ; preds = %113, %163
  %171 = phi i32 [ %169, %163 ], [ %117, %113 ]
  %172 = phi i32 [ %165, %163 ], [ %119, %113 ]
  %173 = getelementptr inbounds nuw i8, ptr %115, i64 3
  %174 = load i8, ptr %173, align 1, !tbaa !16
  %175 = sext i8 %174 to i32
  %176 = shl nsw i32 %175, 24
  %177 = add i32 %176, %116
  br label %178

178:                                              ; preds = %113, %170
  %179 = phi i32 [ %177, %170 ], [ %116, %113 ]
  %180 = phi i32 [ %171, %170 ], [ %117, %113 ]
  %181 = phi i32 [ %172, %170 ], [ %119, %113 ]
  %182 = getelementptr inbounds nuw i8, ptr %115, i64 2
  %183 = load i8, ptr %182, align 1, !tbaa !16
  %184 = sext i8 %183 to i32
  %185 = shl nsw i32 %184, 16
  %186 = add i32 %185, %179
  br label %187

187:                                              ; preds = %113, %178
  %188 = phi i32 [ %186, %178 ], [ %116, %113 ]
  %189 = phi i32 [ %180, %178 ], [ %117, %113 ]
  %190 = phi i32 [ %181, %178 ], [ %119, %113 ]
  %191 = getelementptr inbounds nuw i8, ptr %115, i64 1
  %192 = load i8, ptr %191, align 1, !tbaa !16
  %193 = sext i8 %192 to i32
  %194 = shl nsw i32 %193, 8
  %195 = add i32 %194, %188
  br label %196

196:                                              ; preds = %113, %187
  %197 = phi i32 [ %195, %187 ], [ %116, %113 ]
  %198 = phi i32 [ %189, %187 ], [ %117, %113 ]
  %199 = phi i32 [ %190, %187 ], [ %119, %113 ]
  %200 = load i8, ptr %115, align 1, !tbaa !16
  %201 = sext i8 %200 to i32
  %202 = add i32 %197, %201
  br label %203

203:                                              ; preds = %196, %113
  %204 = phi i32 [ %116, %113 ], [ %202, %196 ]
  %205 = phi i32 [ %117, %113 ], [ %198, %196 ]
  %206 = phi i32 [ %119, %113 ], [ %199, %196 ]
  %207 = add i32 %205, %206
  %208 = sub i32 %204, %207
  %209 = lshr i32 %206, 13
  %210 = xor i32 %208, %209
  %211 = add i32 %206, %210
  %212 = sub i32 %205, %211
  %213 = shl i32 %210, 8
  %214 = xor i32 %212, %213
  %215 = add i32 %210, %214
  %216 = sub i32 %206, %215
  %217 = lshr i32 %214, 13
  %218 = xor i32 %216, %217
  %219 = add i32 %214, %218
  %220 = sub i32 %210, %219
  %221 = lshr i32 %218, 12
  %222 = xor i32 %220, %221
  %223 = add i32 %218, %222
  %224 = sub i32 %214, %223
  %225 = shl i32 %222, 16
  %226 = xor i32 %224, %225
  %227 = add i32 %222, %226
  %228 = sub i32 %218, %227
  %229 = lshr i32 %226, 5
  %230 = xor i32 %228, %229
  %231 = add i32 %226, %230
  %232 = sub i32 %222, %231
  %233 = lshr i32 %230, 3
  %234 = xor i32 %232, %233
  %235 = add i32 %230, %234
  %236 = sub i32 %226, %235
  %237 = shl i32 %234, 10
  %238 = xor i32 %236, %237
  %239 = add i32 %234, %238
  %240 = sub i32 %230, %239
  %241 = lshr i32 %238, 15
  %242 = xor i32 %240, %241
  ret i32 %242
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.fshl.i32(i32, i32, i32) #2

attributes #0 = { mustprogress nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_75" "target-features"="+ptx87,+sm_75" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_75" "target-features"="+ptx87,+sm_75" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}
!nvvmir.version = !{!4}
!nvvm.annotations = !{!5, !6, !5, !7, !7, !7, !7, !8, !8, !7}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 8]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!3 = !{!"Ubuntu clang version 21.1.5 (++20251023083255+45afac62e373-1~exp1~20251023083404.50)"}
!4 = !{i32 1, i32 4}
!5 = !{null, !"align", i32 8}
!6 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!7 = !{null, !"align", i32 16}
!8 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !11, i64 0}
!11 = !{!"omnipotent char", !12, i64 0}
!12 = !{!"Simple C++ TBAA"}
!13 = distinct !{!13, !14, !15}
!14 = !{!"llvm.loop.mustprogress"}
!15 = !{!"llvm.loop.unroll.disable"}
!16 = !{!11, !11, i64 0}
!17 = distinct !{!17, !14, !15}
