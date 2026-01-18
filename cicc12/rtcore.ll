; ModuleID = './rtcore.bc'
source_filename = "llvm-link"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-libdevice"

%struct.ulonglong2 = type { i64, i64 }

; Function Attrs: noinline
define weak dso_local float @__cuda_sm20_div_rd_f32(float %0, float %1) #0 {
  %3 = bitcast float %0 to i32
  %4 = bitcast float %1 to i32
  %5 = lshr i32 %3, 23
  %6 = and i32 %5, 255
  %7 = add nsw i32 %6, -1
  %8 = lshr i32 %4, 23
  %9 = and i32 %8, 255
  %10 = add nsw i32 %9, -1
  %11 = icmp ult i32 %10, 254
  %12 = icmp ult i32 %7, 254
  %13 = and i1 %12, %11
  br i1 %13, label %65, label %14

14:                                               ; preds = %2
  %15 = call float @llvm.nvvm.fabs.ftz.f32(float %0)
  %16 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %15, float 0.000000e+00)
  %17 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %18 = fcmp ugt float %16, %17
  br i1 %18, label %24, label %19

19:                                               ; preds = %14
  %20 = call float @llvm.nvvm.fabs.ftz.f32(float %1)
  %21 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %20, float 0.000000e+00)
  %22 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %23 = fcmp ugt float %21, %22
  br i1 %23, label %24, label %27

24:                                               ; preds = %19, %14
  %25 = call float @llvm.nvvm.add.ftz.f32(i32 1, float %0, float %1)
  %26 = bitcast float %25 to i32
  br label %133

27:                                               ; preds = %19
  %.mask = and i32 %3, 2147483647
  %28 = icmp eq i32 %.mask, 0
  %.mask30 = and i32 %4, 2147483647
  %29 = icmp eq i32 %.mask30, 0
  %30 = or i32 %3, %4
  %31 = and i32 %30, 2147483647
  %32 = icmp eq i32 %31, 0
  br i1 %32, label %40, label %33

33:                                               ; preds = %27
  %34 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %15, float 0.000000e+00)
  %35 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %36 = fcmp oeq float %34, %35
  %37 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %20, float 0.000000e+00)
  %38 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %39 = fcmp oeq float %37, %38
  %.not = xor i1 %36, true
  %.not31 = xor i1 %39, true
  %brmerge = or i1 %.not, %.not31
  br i1 %brmerge, label %43, label %40

40:                                               ; preds = %33, %27
  %41 = call float @llvm.nvvm.rsqrt.approx.ftz.f32(float 0xFFF8000000000000)
  %42 = bitcast float %41 to i32
  br label %133

43:                                               ; preds = %33
  %44 = or i1 %39, %28
  br i1 %44, label %45, label %48

45:                                               ; preds = %43
  %46 = xor i32 %4, %3
  %47 = and i32 %46, -2147483648
  br label %133

48:                                               ; preds = %43
  %49 = or i1 %36, %29
  br i1 %49, label %50, label %54

50:                                               ; preds = %48
  %51 = xor i32 %4, %3
  %52 = and i32 %51, -2147483648
  %53 = or i32 %52, 2139095040
  br label %133

54:                                               ; preds = %48
  %55 = icmp eq i32 %6, 0
  br i1 %55, label %56, label %59

56:                                               ; preds = %54
  %57 = call float @llvm.nvvm.fma.f32(i32 1, float %0, float 0x43F0000000000000, float 0.000000e+00)
  %58 = bitcast float %57 to i32
  br label %59

59:                                               ; preds = %54, %56
  %.027 = phi i32 [ %58, %56 ], [ %3, %54 ]
  %.02 = phi i32 [ -64, %56 ], [ 0, %54 ]
  %60 = icmp eq i32 %9, 0
  br i1 %60, label %61, label %65

61:                                               ; preds = %59
  %62 = call float @llvm.nvvm.fma.f32(i32 1, float %1, float 0x43F0000000000000, float 0.000000e+00)
  %63 = bitcast float %62 to i32
  %64 = add nsw i32 %.02, 64
  br label %65

65:                                               ; preds = %2, %59, %61
  %.029 = phi i32 [ %63, %61 ], [ %4, %59 ], [ %4, %2 ]
  %.128 = phi i32 [ %.027, %61 ], [ %.027, %59 ], [ %3, %2 ]
  %.13 = phi i32 [ %64, %61 ], [ %.02, %59 ], [ 0, %2 ]
  %66 = shl nuw nsw i32 %6, 23
  %67 = add nsw i32 %66, -1065353216
  %68 = sub i32 %.128, %67
  %69 = shl nuw nsw i32 %9, 23
  %70 = add nsw i32 %69, -1065353216
  %71 = sub i32 %.029, %70
  %72 = bitcast i32 %71 to float
  %73 = call float @llvm.nvvm.sub.ftz.f32(i32 1, float 0.000000e+00, float %72)
  %74 = bitcast i32 %71 to float
  %75 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %74)
  %76 = bitcast i32 %68 to float
  %77 = call float @llvm.nvvm.fma.f32(i32 1, float %76, float %75, float 0.000000e+00)
  %78 = bitcast i32 %68 to float
  %79 = call float @llvm.nvvm.fma.f32(i32 1, float %73, float %77, float %78)
  %80 = call float @llvm.nvvm.fma.f32(i32 1, float %79, float %75, float %77)
  %81 = bitcast i32 %68 to float
  %82 = call float @llvm.nvvm.fma.f32(i32 1, float %73, float %80, float %81)
  %83 = call float @llvm.nvvm.fma.f32(i32 2, float %82, float %75, float %80)
  %84 = bitcast float %83 to i32
  %85 = sub nsw i32 %7, %10
  %86 = lshr i32 %84, 23
  %87 = and i32 %86, 255
  %88 = add nsw i32 %85, %87
  %89 = add nsw i32 %.13, %88
  %90 = add nsw i32 %89, -1
  %91 = icmp ugt i32 %90, 253
  br i1 %91, label %96, label %92

92:                                               ; preds = %65
  %93 = sub nsw i32 %89, %87
  %94 = shl i32 %93, 23
  %95 = add i32 %94, %84
  br label %133

96:                                               ; preds = %65
  %97 = and i32 %84, -2147483648
  %98 = icmp slt i32 %89, 255
  br i1 %98, label %101, label %99

99:                                               ; preds = %96
  %100 = icmp eq i32 %97, 0
  %. = select i1 %100, i32 2139095039, i32 2139095040
  br label %131

101:                                              ; preds = %96
  %102 = icmp sgt i32 %89, 0
  br i1 %102, label %131, label %103

103:                                              ; preds = %101
  %104 = icmp sgt i32 %89, -25
  br i1 %104, label %107, label %105

105:                                              ; preds = %103
  %106 = lshr i32 %84, 31
  br label %131

107:                                              ; preds = %103
  %108 = call float @llvm.nvvm.fma.f32(i32 3, float %82, float %75, float %80)
  %109 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %83, float 0.000000e+00)
  %110 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %108, float 0.000000e+00)
  %111 = fcmp une float %109, %110
  %112 = call float @llvm.nvvm.fma.f32(i32 4, float %82, float %75, float %80)
  %113 = bitcast float %112 to i32
  %114 = and i32 %113, 8388607
  %115 = or i32 %114, 8388608
  %116 = icmp eq i32 %89, 1
  br i1 %116, label %124, label %117

117:                                              ; preds = %107
  %118 = add nsw i32 %89, 31
  %119 = shl i32 %115, %118
  %120 = icmp ne i32 %119, 0
  %121 = or i1 %111, %120
  %122 = sub nsw i32 1, %89
  %123 = lshr i32 %115, %122
  br label %124

124:                                              ; preds = %107, %117
  %.01 = phi i32 [ %123, %117 ], [ %115, %107 ]
  %.0.in = phi i1 [ %121, %117 ], [ %111, %107 ]
  %125 = lshr i32 %84, 31
  %126 = zext i1 %.0.in to i32
  %127 = and i32 %125, %126
  %128 = icmp eq i32 %127, 1
  %129 = add i32 %.01, 1
  %130 = select i1 %128, i32 %129, i32 %.01
  br label %131

131:                                              ; preds = %124, %101, %105, %99
  %.1 = phi i32 [ %106, %105 ], [ %., %99 ], [ %84, %101 ], [ %130, %124 ]
  %132 = or i32 %97, %.1
  br label %133

133:                                              ; preds = %92, %131, %50, %45, %40, %24
  %.sroa.026.0 = phi i32 [ %42, %40 ], [ %53, %50 ], [ %47, %45 ], [ %26, %24 ], [ %132, %131 ], [ %95, %92 ]
  %134 = bitcast i32 %.sroa.026.0 to float
  ret float %134
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare dso_local float @llvm.nvvm.fabs.ftz.f32(float) #1

; Unknown intrinsic
declare dso_local float @llvm.nvvm.add.ftz.f32(i32, float, float) #2

; Unknown intrinsic
declare dso_local float @llvm.nvvm.rsqrt.approx.ftz.f32(float) #2

; Unknown intrinsic
declare dso_local float @llvm.nvvm.fma.f32(i32, float, float, float) #2

; Unknown intrinsic
declare dso_local float @llvm.nvvm.sub.ftz.f32(i32, float, float) #2

; Unknown intrinsic
declare dso_local float @llvm.nvvm.rcp.approx.ftz.f32(float) #2

; Function Attrs: noinline
define weak dso_local float @__cuda_sm20_div_ru_f32(float %0, float %1) #0 {
  %3 = bitcast float %0 to i32
  %4 = bitcast float %1 to i32
  %5 = lshr i32 %3, 23
  %6 = and i32 %5, 255
  %7 = add nsw i32 %6, -1
  %8 = lshr i32 %4, 23
  %9 = and i32 %8, 255
  %10 = add nsw i32 %9, -1
  %11 = icmp ult i32 %10, 254
  %12 = icmp ult i32 %7, 254
  %13 = and i1 %12, %11
  br i1 %13, label %65, label %14

14:                                               ; preds = %2
  %15 = call float @llvm.nvvm.fabs.ftz.f32(float %0)
  %16 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %15, float 0.000000e+00)
  %17 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %18 = fcmp ugt float %16, %17
  br i1 %18, label %24, label %19

19:                                               ; preds = %14
  %20 = call float @llvm.nvvm.fabs.ftz.f32(float %1)
  %21 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %20, float 0.000000e+00)
  %22 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %23 = fcmp ugt float %21, %22
  br i1 %23, label %24, label %27

24:                                               ; preds = %19, %14
  %25 = call float @llvm.nvvm.add.ftz.f32(i32 1, float %0, float %1)
  %26 = bitcast float %25 to i32
  br label %135

27:                                               ; preds = %19
  %.mask = and i32 %3, 2147483647
  %28 = icmp eq i32 %.mask, 0
  %.mask30 = and i32 %4, 2147483647
  %29 = icmp eq i32 %.mask30, 0
  %30 = or i32 %3, %4
  %31 = and i32 %30, 2147483647
  %32 = icmp eq i32 %31, 0
  br i1 %32, label %40, label %33

33:                                               ; preds = %27
  %34 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %15, float 0.000000e+00)
  %35 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %36 = fcmp oeq float %34, %35
  %37 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %20, float 0.000000e+00)
  %38 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %39 = fcmp oeq float %37, %38
  %.not = xor i1 %36, true
  %.not31 = xor i1 %39, true
  %brmerge = or i1 %.not, %.not31
  br i1 %brmerge, label %43, label %40

40:                                               ; preds = %33, %27
  %41 = call float @llvm.nvvm.rsqrt.approx.ftz.f32(float 0xFFF8000000000000)
  %42 = bitcast float %41 to i32
  br label %135

43:                                               ; preds = %33
  %44 = or i1 %39, %28
  br i1 %44, label %45, label %48

45:                                               ; preds = %43
  %46 = xor i32 %4, %3
  %47 = and i32 %46, -2147483648
  br label %135

48:                                               ; preds = %43
  %49 = or i1 %36, %29
  br i1 %49, label %50, label %54

50:                                               ; preds = %48
  %51 = xor i32 %4, %3
  %52 = and i32 %51, -2147483648
  %53 = or i32 %52, 2139095040
  br label %135

54:                                               ; preds = %48
  %55 = icmp eq i32 %6, 0
  br i1 %55, label %56, label %59

56:                                               ; preds = %54
  %57 = call float @llvm.nvvm.fma.f32(i32 1, float %0, float 0x43F0000000000000, float 0.000000e+00)
  %58 = bitcast float %57 to i32
  br label %59

59:                                               ; preds = %54, %56
  %.027 = phi i32 [ %58, %56 ], [ %3, %54 ]
  %.02 = phi i32 [ -64, %56 ], [ 0, %54 ]
  %60 = icmp eq i32 %9, 0
  br i1 %60, label %61, label %65

61:                                               ; preds = %59
  %62 = call float @llvm.nvvm.fma.f32(i32 1, float %1, float 0x43F0000000000000, float 0.000000e+00)
  %63 = bitcast float %62 to i32
  %64 = add nsw i32 %.02, 64
  br label %65

65:                                               ; preds = %2, %59, %61
  %.029 = phi i32 [ %63, %61 ], [ %4, %59 ], [ %4, %2 ]
  %.128 = phi i32 [ %.027, %61 ], [ %.027, %59 ], [ %3, %2 ]
  %.13 = phi i32 [ %64, %61 ], [ %.02, %59 ], [ 0, %2 ]
  %66 = shl nuw nsw i32 %6, 23
  %67 = add nsw i32 %66, -1065353216
  %68 = sub i32 %.128, %67
  %69 = shl nuw nsw i32 %9, 23
  %70 = add nsw i32 %69, -1065353216
  %71 = sub i32 %.029, %70
  %72 = bitcast i32 %71 to float
  %73 = call float @llvm.nvvm.sub.ftz.f32(i32 1, float 0.000000e+00, float %72)
  %74 = bitcast i32 %71 to float
  %75 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %74)
  %76 = bitcast i32 %68 to float
  %77 = call float @llvm.nvvm.fma.f32(i32 1, float %76, float %75, float 0.000000e+00)
  %78 = bitcast i32 %68 to float
  %79 = call float @llvm.nvvm.fma.f32(i32 1, float %73, float %77, float %78)
  %80 = call float @llvm.nvvm.fma.f32(i32 1, float %79, float %75, float %77)
  %81 = bitcast i32 %68 to float
  %82 = call float @llvm.nvvm.fma.f32(i32 1, float %73, float %80, float %81)
  %83 = call float @llvm.nvvm.fma.f32(i32 3, float %82, float %75, float %80)
  %84 = bitcast float %83 to i32
  %85 = sub nsw i32 %7, %10
  %86 = lshr i32 %84, 23
  %87 = and i32 %86, 255
  %88 = add nsw i32 %85, %87
  %89 = add nsw i32 %.13, %88
  %90 = add nsw i32 %89, -1
  %91 = icmp ugt i32 %90, 253
  br i1 %91, label %96, label %92

92:                                               ; preds = %65
  %93 = sub nsw i32 %89, %87
  %94 = shl i32 %93, 23
  %95 = add i32 %94, %84
  br label %135

96:                                               ; preds = %65
  %97 = and i32 %84, -2147483648
  %98 = icmp slt i32 %89, 255
  br i1 %98, label %101, label %99

99:                                               ; preds = %96
  %100 = icmp eq i32 %97, 0
  %. = select i1 %100, i32 2139095040, i32 2139095039
  br label %133

101:                                              ; preds = %96
  %102 = icmp sgt i32 %89, 0
  br i1 %102, label %133, label %103

103:                                              ; preds = %101
  %104 = icmp sgt i32 %89, -25
  br i1 %104, label %108, label %105

105:                                              ; preds = %103
  %106 = lshr i32 %84, 31
  %107 = xor i32 %106, 1
  br label %133

108:                                              ; preds = %103
  %109 = call float @llvm.nvvm.fma.f32(i32 2, float %82, float %75, float %80)
  %110 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %83, float 0.000000e+00)
  %111 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %109, float 0.000000e+00)
  %112 = fcmp une float %110, %111
  %113 = call float @llvm.nvvm.fma.f32(i32 4, float %82, float %75, float %80)
  %114 = bitcast float %113 to i32
  %115 = and i32 %114, 8388607
  %116 = or i32 %115, 8388608
  %117 = icmp eq i32 %89, 1
  br i1 %117, label %125, label %118

118:                                              ; preds = %108
  %119 = add nsw i32 %89, 31
  %120 = shl i32 %116, %119
  %121 = icmp ne i32 %120, 0
  %122 = or i1 %112, %121
  %123 = sub nsw i32 1, %89
  %124 = lshr i32 %116, %123
  br label %125

125:                                              ; preds = %108, %118
  %.01 = phi i32 [ %124, %118 ], [ %116, %108 ]
  %.0.in = phi i1 [ %122, %118 ], [ %112, %108 ]
  %126 = lshr i32 %84, 31
  %127 = xor i32 %126, 1
  %128 = zext i1 %.0.in to i32
  %129 = and i32 %127, %128
  %130 = icmp eq i32 %129, 1
  %131 = add i32 %.01, 1
  %132 = select i1 %130, i32 %131, i32 %.01
  br label %133

133:                                              ; preds = %125, %101, %105, %99
  %.1 = phi i32 [ %107, %105 ], [ %., %99 ], [ %84, %101 ], [ %132, %125 ]
  %134 = or i32 %97, %.1
  br label %135

135:                                              ; preds = %92, %133, %50, %45, %40, %24
  %.sroa.026.0 = phi i32 [ %42, %40 ], [ %53, %50 ], [ %47, %45 ], [ %26, %24 ], [ %134, %133 ], [ %95, %92 ]
  %136 = bitcast i32 %.sroa.026.0 to float
  ret float %136
}

; Function Attrs: noinline
define weak dso_local float @__cuda_sm20_div_rz_f32(float %0, float %1) #0 {
  %3 = bitcast float %0 to i32
  %4 = bitcast float %1 to i32
  %5 = lshr i32 %3, 23
  %6 = and i32 %5, 255
  %7 = add nsw i32 %6, -1
  %8 = lshr i32 %4, 23
  %9 = and i32 %8, 255
  %10 = add nsw i32 %9, -1
  %11 = icmp ult i32 %10, 254
  %12 = icmp ult i32 %7, 254
  %13 = and i1 %12, %11
  br i1 %13, label %65, label %14

14:                                               ; preds = %2
  %15 = call float @llvm.nvvm.fabs.ftz.f32(float %0)
  %16 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %15, float 0.000000e+00)
  %17 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %18 = fcmp ugt float %16, %17
  br i1 %18, label %24, label %19

19:                                               ; preds = %14
  %20 = call float @llvm.nvvm.fabs.ftz.f32(float %1)
  %21 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %20, float 0.000000e+00)
  %22 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %23 = fcmp ugt float %21, %22
  br i1 %23, label %24, label %27

24:                                               ; preds = %19, %14
  %25 = call float @llvm.nvvm.add.ftz.f32(i32 1, float %0, float %1)
  %26 = bitcast float %25 to i32
  br label %110

27:                                               ; preds = %19
  %.mask = and i32 %3, 2147483647
  %28 = icmp eq i32 %.mask, 0
  %.mask30 = and i32 %4, 2147483647
  %29 = icmp eq i32 %.mask30, 0
  %30 = or i32 %3, %4
  %31 = and i32 %30, 2147483647
  %32 = icmp eq i32 %31, 0
  br i1 %32, label %40, label %33

33:                                               ; preds = %27
  %34 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %15, float 0.000000e+00)
  %35 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %36 = fcmp oeq float %34, %35
  %37 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %20, float 0.000000e+00)
  %38 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %39 = fcmp oeq float %37, %38
  %.not = xor i1 %36, true
  %.not31 = xor i1 %39, true
  %brmerge = or i1 %.not, %.not31
  br i1 %brmerge, label %43, label %40

40:                                               ; preds = %33, %27
  %41 = call float @llvm.nvvm.rsqrt.approx.ftz.f32(float 0xFFF8000000000000)
  %42 = bitcast float %41 to i32
  br label %110

43:                                               ; preds = %33
  %44 = or i1 %39, %28
  br i1 %44, label %45, label %48

45:                                               ; preds = %43
  %46 = xor i32 %4, %3
  %47 = and i32 %46, -2147483648
  br label %110

48:                                               ; preds = %43
  %49 = or i1 %36, %29
  br i1 %49, label %50, label %54

50:                                               ; preds = %48
  %51 = xor i32 %4, %3
  %52 = and i32 %51, -2147483648
  %53 = or i32 %52, 2139095040
  br label %110

54:                                               ; preds = %48
  %55 = icmp eq i32 %6, 0
  br i1 %55, label %56, label %59

56:                                               ; preds = %54
  %57 = call float @llvm.nvvm.fma.f32(i32 1, float %0, float 0x43F0000000000000, float 0.000000e+00)
  %58 = bitcast float %57 to i32
  br label %59

59:                                               ; preds = %54, %56
  %.027 = phi i32 [ %58, %56 ], [ %3, %54 ]
  %.01 = phi i32 [ -64, %56 ], [ 0, %54 ]
  %60 = icmp eq i32 %9, 0
  br i1 %60, label %61, label %65

61:                                               ; preds = %59
  %62 = call float @llvm.nvvm.fma.f32(i32 1, float %1, float 0x43F0000000000000, float 0.000000e+00)
  %63 = bitcast float %62 to i32
  %64 = add nsw i32 %.01, 64
  br label %65

65:                                               ; preds = %2, %59, %61
  %.029 = phi i32 [ %63, %61 ], [ %4, %59 ], [ %4, %2 ]
  %.128 = phi i32 [ %.027, %61 ], [ %.027, %59 ], [ %3, %2 ]
  %.1 = phi i32 [ %64, %61 ], [ %.01, %59 ], [ 0, %2 ]
  %66 = shl nuw nsw i32 %6, 23
  %67 = add nsw i32 %66, -1065353216
  %68 = sub i32 %.128, %67
  %69 = shl nuw nsw i32 %9, 23
  %70 = add nsw i32 %69, -1065353216
  %71 = sub i32 %.029, %70
  %72 = bitcast i32 %71 to float
  %73 = call float @llvm.nvvm.sub.ftz.f32(i32 1, float 0.000000e+00, float %72)
  %74 = bitcast i32 %71 to float
  %75 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %74)
  %76 = bitcast i32 %68 to float
  %77 = call float @llvm.nvvm.fma.f32(i32 1, float %76, float %75, float 0.000000e+00)
  %78 = bitcast i32 %68 to float
  %79 = call float @llvm.nvvm.fma.f32(i32 1, float %73, float %77, float %78)
  %80 = call float @llvm.nvvm.fma.f32(i32 1, float %79, float %75, float %77)
  %81 = bitcast i32 %68 to float
  %82 = call float @llvm.nvvm.fma.f32(i32 1, float %73, float %80, float %81)
  %83 = call float @llvm.nvvm.fma.f32(i32 4, float %82, float %75, float %80)
  %84 = bitcast float %83 to i32
  %85 = sub nsw i32 %7, %10
  %86 = lshr i32 %84, 23
  %87 = and i32 %86, 255
  %88 = add nsw i32 %85, %87
  %89 = add nsw i32 %.1, %88
  %90 = add nsw i32 %89, -1
  %91 = icmp ugt i32 %90, 253
  br i1 %91, label %96, label %92

92:                                               ; preds = %65
  %93 = sub nsw i32 %89, %87
  %94 = shl i32 %93, 23
  %95 = add i32 %94, %84
  br label %110

96:                                               ; preds = %65
  %97 = icmp slt i32 %89, 255
  br i1 %97, label %98, label %107

98:                                               ; preds = %96
  %99 = icmp sgt i32 %89, 0
  br i1 %99, label %107, label %100

100:                                              ; preds = %98
  %101 = icmp sgt i32 %89, -25
  br i1 %101, label %102, label %107

102:                                              ; preds = %100
  %103 = and i32 %84, 8388607
  %104 = or i32 %103, 8388608
  %105 = sub nsw i32 1, %89
  %106 = lshr i32 %104, %105
  br label %107

107:                                              ; preds = %100, %98, %96, %102
  %.0 = phi i32 [ %106, %102 ], [ 2139095039, %96 ], [ %84, %98 ], [ 0, %100 ]
  %108 = and i32 %84, -2147483648
  %109 = or i32 %.0, %108
  br label %110

110:                                              ; preds = %92, %107, %50, %45, %40, %24
  %.sroa.026.0 = phi i32 [ %42, %40 ], [ %53, %50 ], [ %47, %45 ], [ %26, %24 ], [ %109, %107 ], [ %95, %92 ]
  %111 = bitcast i32 %.sroa.026.0 to float
  ret float %111
}

; Function Attrs: noinline
define weak dso_local float @__cuda_sm20_div_rd_ftz_f32(float %0, float %1) #0 {
  %3 = bitcast float %0 to i32
  %4 = bitcast float %1 to i32
  %5 = lshr i32 %3, 23
  %6 = and i32 %5, 255
  %7 = lshr i32 %4, 23
  %8 = and i32 %7, 255
  %9 = add nsw i32 %6, -1
  %10 = icmp ult i32 %9, 254
  %11 = add nsw i32 %8, -1
  %12 = icmp ult i32 %11, 254
  %13 = and i1 %12, %10
  br i1 %13, label %56, label %14

14:                                               ; preds = %2
  %15 = call float @llvm.nvvm.fabs.ftz.f32(float %0)
  %16 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %15, float 0.000000e+00)
  %17 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %18 = fcmp ugt float %16, %17
  br i1 %18, label %24, label %19

19:                                               ; preds = %14
  %20 = call float @llvm.nvvm.fabs.ftz.f32(float %1)
  %21 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %20, float 0.000000e+00)
  %22 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %23 = fcmp ugt float %21, %22
  br i1 %23, label %24, label %27

24:                                               ; preds = %19, %14
  %25 = call float @llvm.nvvm.add.ftz.f32(i32 1, float %0, float %1)
  %26 = bitcast float %25 to i32
  br label %91

27:                                               ; preds = %19
  %28 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %0, float 0.000000e+00)
  %29 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0.000000e+00, float 0.000000e+00)
  %30 = fcmp oeq float %28, %29
  %31 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %1, float 0.000000e+00)
  %32 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0.000000e+00, float 0.000000e+00)
  %33 = fcmp oeq float %31, %32
  %34 = and i1 %30, %33
  br i1 %34, label %42, label %35

35:                                               ; preds = %27
  %36 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %15, float 0.000000e+00)
  %37 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %38 = fcmp oeq float %36, %37
  %39 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %20, float 0.000000e+00)
  %40 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %41 = fcmp oeq float %39, %40
  %.not = xor i1 %38, true
  %.not20 = xor i1 %41, true
  %brmerge = or i1 %.not, %.not20
  br i1 %brmerge, label %45, label %42

42:                                               ; preds = %35, %27
  %43 = call float @llvm.nvvm.rsqrt.approx.ftz.f32(float 0xFFF8000000000000)
  %44 = bitcast float %43 to i32
  br label %91

45:                                               ; preds = %35
  %46 = or i1 %41, %30
  br i1 %46, label %47, label %50

47:                                               ; preds = %45
  %48 = xor i32 %3, %4
  %49 = and i32 %48, -2147483648
  br label %91

50:                                               ; preds = %45
  %51 = or i1 %38, %33
  br i1 %51, label %52, label %56

52:                                               ; preds = %50
  %53 = xor i32 %3, %4
  %54 = and i32 %53, -2147483648
  %55 = or i32 %54, 2139095040
  br label %91

56:                                               ; preds = %50, %2
  %57 = shl nuw nsw i32 %6, 23
  %58 = add nsw i32 %57, -1065353216
  %59 = shl nuw nsw i32 %8, 23
  %60 = add nsw i32 %59, -1065353216
  %61 = sub i32 %3, %58
  %62 = sub i32 %4, %60
  %63 = bitcast i32 %62 to float
  %64 = call float @llvm.nvvm.sub.ftz.f32(i32 1, float 0.000000e+00, float %63)
  %65 = bitcast i32 %62 to float
  %66 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %65)
  %67 = bitcast i32 %61 to float
  %68 = call float @llvm.nvvm.fma.f32(i32 1, float %67, float %66, float 0.000000e+00)
  %69 = bitcast i32 %61 to float
  %70 = call float @llvm.nvvm.fma.f32(i32 1, float %64, float %68, float %69)
  %71 = call float @llvm.nvvm.fma.f32(i32 1, float %70, float %66, float %68)
  %72 = bitcast i32 %61 to float
  %73 = call float @llvm.nvvm.fma.f32(i32 1, float %64, float %71, float %72)
  %74 = call float @llvm.nvvm.fma.f32(i32 2, float %73, float %66, float %71)
  %75 = bitcast float %74 to i32
  %76 = lshr i32 %75, 23
  %77 = and i32 %76, 255
  %78 = sub nsw i32 %6, %8
  %79 = add nsw i32 %78, %77
  %80 = add nsw i32 %79, -1
  %81 = icmp ugt i32 %80, 253
  br i1 %81, label %85, label %82

82:                                               ; preds = %56
  %83 = shl i32 %78, 23
  %84 = add i32 %83, %75
  br label %91

85:                                               ; preds = %56
  %86 = and i32 %75, -2147483648
  %87 = icmp slt i32 %79, 255
  %88 = icmp slt i32 %79, 1
  %.19 = select i1 %88, i32 0, i32 %75
  %89 = icmp eq i32 %86, 0
  %. = select i1 %89, i32 2139095039, i32 2139095040
  %.0 = select i1 %87, i32 %.19, i32 %.
  %90 = or i32 %86, %.0
  br label %91

91:                                               ; preds = %82, %85, %52, %47, %42, %24
  %.sroa.018.0 = phi i32 [ %44, %42 ], [ %55, %52 ], [ %49, %47 ], [ %26, %24 ], [ %90, %85 ], [ %84, %82 ]
  %92 = bitcast i32 %.sroa.018.0 to float
  ret float %92
}

; Function Attrs: noinline
define weak dso_local float @__cuda_sm20_div_ru_ftz_f32(float %0, float %1) #0 {
  %3 = bitcast float %0 to i32
  %4 = bitcast float %1 to i32
  %5 = lshr i32 %3, 23
  %6 = and i32 %5, 255
  %7 = lshr i32 %4, 23
  %8 = and i32 %7, 255
  %9 = add nsw i32 %6, -1
  %10 = icmp ult i32 %9, 254
  %11 = add nsw i32 %8, -1
  %12 = icmp ult i32 %11, 254
  %13 = and i1 %12, %10
  br i1 %13, label %56, label %14

14:                                               ; preds = %2
  %15 = call float @llvm.nvvm.fabs.ftz.f32(float %0)
  %16 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %15, float 0.000000e+00)
  %17 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %18 = fcmp ugt float %16, %17
  br i1 %18, label %24, label %19

19:                                               ; preds = %14
  %20 = call float @llvm.nvvm.fabs.ftz.f32(float %1)
  %21 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %20, float 0.000000e+00)
  %22 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %23 = fcmp ugt float %21, %22
  br i1 %23, label %24, label %27

24:                                               ; preds = %19, %14
  %25 = call float @llvm.nvvm.add.ftz.f32(i32 1, float %0, float %1)
  %26 = bitcast float %25 to i32
  br label %91

27:                                               ; preds = %19
  %28 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %0, float 0.000000e+00)
  %29 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0.000000e+00, float 0.000000e+00)
  %30 = fcmp oeq float %28, %29
  %31 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %1, float 0.000000e+00)
  %32 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0.000000e+00, float 0.000000e+00)
  %33 = fcmp oeq float %31, %32
  %34 = and i1 %30, %33
  br i1 %34, label %42, label %35

35:                                               ; preds = %27
  %36 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %15, float 0.000000e+00)
  %37 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %38 = fcmp oeq float %36, %37
  %39 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %20, float 0.000000e+00)
  %40 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %41 = fcmp oeq float %39, %40
  %.not = xor i1 %38, true
  %.not20 = xor i1 %41, true
  %brmerge = or i1 %.not, %.not20
  br i1 %brmerge, label %45, label %42

42:                                               ; preds = %35, %27
  %43 = call float @llvm.nvvm.rsqrt.approx.ftz.f32(float 0xFFF8000000000000)
  %44 = bitcast float %43 to i32
  br label %91

45:                                               ; preds = %35
  %46 = or i1 %41, %30
  br i1 %46, label %47, label %50

47:                                               ; preds = %45
  %48 = xor i32 %3, %4
  %49 = and i32 %48, -2147483648
  br label %91

50:                                               ; preds = %45
  %51 = or i1 %38, %33
  br i1 %51, label %52, label %56

52:                                               ; preds = %50
  %53 = xor i32 %3, %4
  %54 = and i32 %53, -2147483648
  %55 = or i32 %54, 2139095040
  br label %91

56:                                               ; preds = %50, %2
  %57 = shl nuw nsw i32 %6, 23
  %58 = add nsw i32 %57, -1065353216
  %59 = shl nuw nsw i32 %8, 23
  %60 = add nsw i32 %59, -1065353216
  %61 = sub i32 %3, %58
  %62 = sub i32 %4, %60
  %63 = bitcast i32 %62 to float
  %64 = call float @llvm.nvvm.sub.ftz.f32(i32 1, float 0.000000e+00, float %63)
  %65 = bitcast i32 %62 to float
  %66 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %65)
  %67 = bitcast i32 %61 to float
  %68 = call float @llvm.nvvm.fma.f32(i32 1, float %67, float %66, float 0.000000e+00)
  %69 = bitcast i32 %61 to float
  %70 = call float @llvm.nvvm.fma.f32(i32 1, float %64, float %68, float %69)
  %71 = call float @llvm.nvvm.fma.f32(i32 1, float %70, float %66, float %68)
  %72 = bitcast i32 %61 to float
  %73 = call float @llvm.nvvm.fma.f32(i32 1, float %64, float %71, float %72)
  %74 = call float @llvm.nvvm.fma.f32(i32 3, float %73, float %66, float %71)
  %75 = bitcast float %74 to i32
  %76 = lshr i32 %75, 23
  %77 = and i32 %76, 255
  %78 = sub nsw i32 %6, %8
  %79 = add nsw i32 %78, %77
  %80 = add nsw i32 %79, -1
  %81 = icmp ugt i32 %80, 253
  br i1 %81, label %85, label %82

82:                                               ; preds = %56
  %83 = shl i32 %78, 23
  %84 = add i32 %83, %75
  br label %91

85:                                               ; preds = %56
  %86 = and i32 %75, -2147483648
  %87 = icmp slt i32 %79, 255
  %88 = icmp slt i32 %79, 1
  %.19 = select i1 %88, i32 0, i32 %75
  %89 = icmp eq i32 %86, 0
  %. = select i1 %89, i32 2139095040, i32 2139095039
  %.0 = select i1 %87, i32 %.19, i32 %.
  %90 = or i32 %86, %.0
  br label %91

91:                                               ; preds = %82, %85, %52, %47, %42, %24
  %.sroa.018.0 = phi i32 [ %44, %42 ], [ %55, %52 ], [ %49, %47 ], [ %26, %24 ], [ %90, %85 ], [ %84, %82 ]
  %92 = bitcast i32 %.sroa.018.0 to float
  ret float %92
}

; Function Attrs: noinline
define weak dso_local float @__cuda_sm20_div_rz_ftz_f32(float %0, float %1) #0 {
  %3 = bitcast float %0 to i32
  %4 = bitcast float %1 to i32
  %5 = lshr i32 %3, 23
  %6 = and i32 %5, 255
  %7 = lshr i32 %4, 23
  %8 = and i32 %7, 255
  %9 = add nsw i32 %6, -1
  %10 = icmp ult i32 %9, 254
  %11 = add nsw i32 %8, -1
  %12 = icmp ult i32 %11, 254
  %13 = and i1 %12, %10
  br i1 %13, label %56, label %14

14:                                               ; preds = %2
  %15 = call float @llvm.nvvm.fabs.ftz.f32(float %0)
  %16 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %15, float 0.000000e+00)
  %17 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %18 = fcmp ugt float %16, %17
  br i1 %18, label %24, label %19

19:                                               ; preds = %14
  %20 = call float @llvm.nvvm.fabs.ftz.f32(float %1)
  %21 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %20, float 0.000000e+00)
  %22 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %23 = fcmp ugt float %21, %22
  br i1 %23, label %24, label %27

24:                                               ; preds = %19, %14
  %25 = call float @llvm.nvvm.add.ftz.f32(i32 1, float %0, float %1)
  %26 = bitcast float %25 to i32
  br label %90

27:                                               ; preds = %19
  %28 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %0, float 0.000000e+00)
  %29 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0.000000e+00, float 0.000000e+00)
  %30 = fcmp oeq float %28, %29
  %31 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %1, float 0.000000e+00)
  %32 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0.000000e+00, float 0.000000e+00)
  %33 = fcmp oeq float %31, %32
  %34 = and i1 %30, %33
  br i1 %34, label %42, label %35

35:                                               ; preds = %27
  %36 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %15, float 0.000000e+00)
  %37 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %38 = fcmp oeq float %36, %37
  %39 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %20, float 0.000000e+00)
  %40 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %41 = fcmp oeq float %39, %40
  %.not = xor i1 %38, true
  %.not21 = xor i1 %41, true
  %brmerge = or i1 %.not, %.not21
  br i1 %brmerge, label %45, label %42

42:                                               ; preds = %35, %27
  %43 = call float @llvm.nvvm.rsqrt.approx.ftz.f32(float 0xFFF8000000000000)
  %44 = bitcast float %43 to i32
  br label %90

45:                                               ; preds = %35
  %46 = or i1 %41, %30
  br i1 %46, label %47, label %50

47:                                               ; preds = %45
  %48 = xor i32 %3, %4
  %49 = and i32 %48, -2147483648
  br label %90

50:                                               ; preds = %45
  %51 = or i1 %38, %33
  br i1 %51, label %52, label %56

52:                                               ; preds = %50
  %53 = xor i32 %3, %4
  %54 = and i32 %53, -2147483648
  %55 = or i32 %54, 2139095040
  br label %90

56:                                               ; preds = %50, %2
  %57 = shl nuw nsw i32 %6, 23
  %58 = add nsw i32 %57, -1065353216
  %59 = shl nuw nsw i32 %8, 23
  %60 = add nsw i32 %59, -1065353216
  %61 = sub i32 %3, %58
  %62 = sub i32 %4, %60
  %63 = bitcast i32 %62 to float
  %64 = call float @llvm.nvvm.sub.ftz.f32(i32 1, float 0.000000e+00, float %63)
  %65 = bitcast i32 %62 to float
  %66 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %65)
  %67 = bitcast i32 %61 to float
  %68 = call float @llvm.nvvm.fma.f32(i32 1, float %67, float %66, float 0.000000e+00)
  %69 = bitcast i32 %61 to float
  %70 = call float @llvm.nvvm.fma.f32(i32 1, float %64, float %68, float %69)
  %71 = call float @llvm.nvvm.fma.f32(i32 1, float %70, float %66, float %68)
  %72 = bitcast i32 %61 to float
  %73 = call float @llvm.nvvm.fma.f32(i32 1, float %64, float %71, float %72)
  %74 = call float @llvm.nvvm.fma.f32(i32 4, float %73, float %66, float %71)
  %75 = bitcast float %74 to i32
  %76 = lshr i32 %75, 23
  %77 = and i32 %76, 255
  %78 = sub nsw i32 %6, %8
  %79 = add nsw i32 %78, %77
  %80 = add nsw i32 %79, -1
  %81 = icmp ugt i32 %80, 253
  br i1 %81, label %85, label %82

82:                                               ; preds = %56
  %83 = shl i32 %78, 23
  %84 = add i32 %83, %75
  br label %90

85:                                               ; preds = %56
  %86 = and i32 %75, -2147483648
  %87 = icmp slt i32 %79, 255
  %88 = icmp slt i32 %79, 1
  %. = select i1 %88, i32 0, i32 %75
  %.0 = select i1 %87, i32 %., i32 2139095039
  %89 = or i32 %86, %.0
  br label %90

90:                                               ; preds = %82, %85, %52, %47, %42, %24
  %.sroa.020.0 = phi i32 [ %44, %42 ], [ %55, %52 ], [ %49, %47 ], [ %26, %24 ], [ %89, %85 ], [ %84, %82 ]
  %91 = bitcast i32 %.sroa.020.0 to float
  ret float %91
}

; Function Attrs: noinline
define weak dso_local float @__cuda_sm3x_div_rn_noftz_f32_slowpath(float %0, float %1) #0 {
  %3 = bitcast float %0 to i32
  %4 = lshr i32 %3, 23
  %5 = and i32 %4, 255
  %6 = add nsw i32 %5, -1
  %7 = bitcast float %1 to i32
  %8 = lshr i32 %7, 23
  %9 = and i32 %8, 255
  %10 = add nsw i32 %9, -1
  %11 = icmp ugt i32 %6, 253
  %12 = icmp ugt i32 %10, 253
  %13 = or i1 %11, %12
  br i1 %13, label %14, label %55

14:                                               ; preds = %2
  %15 = call float @llvm.nvvm.fabs.ftz.f32(float %0)
  %16 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %15, float 0.000000e+00)
  %17 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %18 = fcmp ugt float %16, %17
  br i1 %18, label %138, label %19

19:                                               ; preds = %14
  %20 = call float @llvm.nvvm.fabs.ftz.f32(float %1)
  %21 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %20, float 0.000000e+00)
  %22 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %23 = fcmp ugt float %21, %22
  br i1 %23, label %138, label %24

24:                                               ; preds = %19
  %25 = or i32 %7, %3
  %26 = and i32 %25, 2147483647
  %27 = icmp eq i32 %26, 0
  br i1 %27, label %136, label %28

28:                                               ; preds = %24
  %29 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %15, float 0.000000e+00)
  %30 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %31 = fcmp oeq float %29, %30
  %32 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %20, float 0.000000e+00)
  %33 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %34 = fcmp oeq float %32, %33
  %35 = and i1 %31, %34
  br i1 %35, label %136, label %36

36:                                               ; preds = %28
  %37 = and i32 %3, 2147483647
  %38 = icmp eq i32 %37, 0
  %39 = or i1 %34, %38
  br i1 %39, label %132, label %40

40:                                               ; preds = %36
  %41 = and i32 %7, 2147483647
  %42 = icmp eq i32 %41, 0
  %43 = or i1 %31, %42
  br i1 %43, label %127, label %44

44:                                               ; preds = %40
  %45 = icmp eq i32 %5, 0
  br i1 %45, label %46, label %49

46:                                               ; preds = %44
  %47 = call float @llvm.nvvm.fma.f32(i32 1, float %0, float 0x43F0000000000000, float 0.000000e+00)
  %48 = bitcast float %47 to i32
  br label %49

49:                                               ; preds = %44, %46
  %.034 = phi i32 [ %48, %46 ], [ %3, %44 ]
  %.01 = phi i32 [ -64, %46 ], [ 0, %44 ]
  %50 = icmp eq i32 %9, 0
  br i1 %50, label %51, label %55

51:                                               ; preds = %49
  %52 = call float @llvm.nvvm.fma.f32(i32 1, float %1, float 0x43F0000000000000, float 0.000000e+00)
  %53 = bitcast float %52 to i32
  %54 = add nsw i32 %.01, 64
  br label %55

55:                                               ; preds = %49, %2, %51
  %.135 = phi i32 [ %.034, %51 ], [ %3, %2 ], [ %.034, %49 ]
  %.033 = phi i32 [ %53, %51 ], [ %7, %2 ], [ %7, %49 ]
  %.1 = phi i32 [ %54, %51 ], [ 0, %2 ], [ %.01, %49 ]
  %56 = add nsw i32 %5, -127
  %57 = shl i32 %56, 23
  %58 = sub nsw i32 %.135, %57
  %59 = shl nuw nsw i32 %9, 23
  %60 = add nsw i32 %59, -1065353216
  %61 = sub nsw i32 %.033, %60
  %62 = bitcast i32 %61 to float
  %63 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %62)
  %64 = bitcast i32 %61 to float
  %65 = call float @llvm.nvvm.sub.ftz.f32(i32 1, float 0.000000e+00, float %64)
  %66 = call float @llvm.nvvm.fma.f32(i32 1, float %65, float %63, float 1.000000e+00)
  %67 = call float @llvm.nvvm.fma.f32(i32 1, float %63, float %66, float %63)
  %68 = bitcast i32 %58 to float
  %69 = call float @llvm.nvvm.fma.f32(i32 1, float %68, float %67, float 0.000000e+00)
  %70 = bitcast i32 %58 to float
  %71 = call float @llvm.nvvm.fma.f32(i32 1, float %65, float %69, float %70)
  %72 = call float @llvm.nvvm.fma.f32(i32 1, float %71, float %67, float %69)
  %73 = bitcast i32 %58 to float
  %74 = call float @llvm.nvvm.fma.f32(i32 1, float %65, float %72, float %73)
  %75 = call float @llvm.nvvm.fma.f32(i32 1, float %74, float %67, float %72)
  %76 = bitcast float %75 to i32
  %77 = lshr i32 %76, 23
  %78 = and i32 %77, 255
  %79 = sub nsw i32 127, %9
  %80 = add nsw i32 %79, %56
  %81 = add nsw i32 %80, %.1
  %82 = add nsw i32 %81, %78
  %83 = add nsw i32 %82, -1
  %84 = icmp ult i32 %83, 254
  br i1 %84, label %122, label %85

85:                                               ; preds = %55
  %86 = icmp sgt i32 %82, 254
  br i1 %86, label %119, label %87

87:                                               ; preds = %85
  %88 = icmp slt i32 %82, 1
  br i1 %88, label %89, label %125

89:                                               ; preds = %87
  %90 = icmp slt i32 %82, -24
  %91 = and i32 %76, -2147483648
  br i1 %90, label %125, label %92

92:                                               ; preds = %89
  %93 = call float @llvm.nvvm.fma.f32(i32 3, float %74, float %67, float %72)
  %94 = call float @llvm.nvvm.fma.f32(i32 2, float %74, float %67, float %72)
  %95 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %93, float 0.000000e+00)
  %96 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %94, float 0.000000e+00)
  %97 = fcmp une float %95, %96
  %98 = call float @llvm.nvvm.fma.f32(i32 4, float %74, float %67, float %72)
  %99 = bitcast float %98 to i32
  %100 = and i32 %99, 8388607
  %101 = or i32 %100, 8388608
  %102 = add nsw i32 %82, 32
  %103 = shl i32 %101, %102
  %104 = icmp ne i32 %103, 0
  %105 = icmp eq i32 %82, 0
  %106 = sub nsw i32 0, %82
  %107 = select i1 %105, i32 0, i32 %106
  %108 = lshr i32 %101, %107
  %109 = icmp ne i32 %82, 0
  %110 = and i1 %104, %109
  %111 = or i1 %97, %110
  %112 = zext i1 %111 to i32
  %113 = lshr i32 %108, 1
  %114 = and i32 %113, 1
  %115 = or i32 %114, %112
  %116 = and i32 %115, %108
  %117 = add nuw nsw i32 %116, %113
  %118 = or i32 %117, %91
  br label %125

119:                                              ; preds = %85
  %120 = and i32 %76, -2147483648
  %121 = or i32 %120, 2139095040
  br label %125

122:                                              ; preds = %55
  %123 = shl i32 %81, 23
  %124 = add nsw i32 %123, %76
  br label %125

125:                                              ; preds = %89, %87, %122, %119, %92
  %.032 = phi i32 [ %124, %122 ], [ %121, %119 ], [ %118, %92 ], [ %76, %87 ], [ %91, %89 ]
  %126 = bitcast i32 %.032 to float
  br label %140

127:                                              ; preds = %40
  %128 = xor i32 %7, %3
  %129 = and i32 %128, -2147483648
  %130 = or i32 %129, 2139095040
  %131 = bitcast i32 %130 to float
  br label %140

132:                                              ; preds = %36
  %133 = xor i32 %7, %3
  %134 = and i32 %133, -2147483648
  %135 = bitcast i32 %134 to float
  br label %140

136:                                              ; preds = %28, %24
  %137 = call float @llvm.nvvm.rsqrt.approx.ftz.f32(float 0xFFF8000000000000)
  br label %140

138:                                              ; preds = %19, %14
  %139 = call float @llvm.nvvm.add.ftz.f32(i32 1, float %0, float %1)
  br label %140

140:                                              ; preds = %138, %136, %132, %127, %125
  %.0 = phi float [ %139, %138 ], [ %137, %136 ], [ %135, %132 ], [ %131, %127 ], [ %126, %125 ]
  ret float %.0
}

; Function Attrs: noinline
define weak dso_local float @__cuda_sm3x_div_rn_ftz_f32_slowpath(float %0, float %1) #0 {
  %3 = bitcast float %0 to i32
  %4 = lshr i32 %3, 23
  %5 = and i32 %4, 255
  %6 = add nsw i32 %5, -1
  %7 = bitcast float %1 to i32
  %8 = lshr i32 %7, 23
  %9 = and i32 %8, 255
  %10 = add nsw i32 %9, -1
  %11 = icmp ugt i32 %6, 253
  %12 = icmp ugt i32 %10, 253
  %13 = or i1 %11, %12
  br i1 %13, label %14, label %44

14:                                               ; preds = %2
  %15 = call float @llvm.nvvm.fabs.ftz.f32(float %0)
  %16 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %15, float 0.000000e+00)
  %17 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %18 = fcmp ugt float %16, %17
  br i1 %18, label %95, label %19

19:                                               ; preds = %14
  %20 = call float @llvm.nvvm.fabs.ftz.f32(float %1)
  %21 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %20, float 0.000000e+00)
  %22 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %23 = fcmp ugt float %21, %22
  br i1 %23, label %95, label %24

24:                                               ; preds = %19
  %25 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %0, float 0.000000e+00)
  %26 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0.000000e+00, float 0.000000e+00)
  %27 = fcmp oeq float %25, %26
  %28 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %1, float 0.000000e+00)
  %29 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0.000000e+00, float 0.000000e+00)
  %30 = fcmp oeq float %28, %29
  %31 = and i1 %27, %30
  br i1 %31, label %93, label %32

32:                                               ; preds = %24
  %33 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %15, float 0.000000e+00)
  %34 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %35 = fcmp oeq float %33, %34
  %36 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %20, float 0.000000e+00)
  %37 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %38 = fcmp oeq float %36, %37
  %39 = and i1 %35, %38
  br i1 %39, label %93, label %40

40:                                               ; preds = %32
  %41 = or i1 %38, %27
  br i1 %41, label %89, label %42

42:                                               ; preds = %40
  %43 = or i1 %35, %30
  br i1 %43, label %84, label %44

44:                                               ; preds = %42, %2
  %45 = add nsw i32 %5, -127
  %46 = shl i32 %45, 23
  %47 = sub nsw i32 %3, %46
  %48 = add nsw i32 %9, -127
  %49 = shl i32 %48, 23
  %50 = sub nsw i32 %7, %49
  %51 = bitcast i32 %50 to float
  %52 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %51)
  %53 = bitcast i32 %50 to float
  %54 = call float @llvm.nvvm.sub.ftz.f32(i32 1, float 0.000000e+00, float %53)
  %55 = call float @llvm.nvvm.fma.f32(i32 1, float %54, float %52, float 1.000000e+00)
  %56 = call float @llvm.nvvm.fma.f32(i32 1, float %52, float %55, float %52)
  %57 = bitcast i32 %47 to float
  %58 = call float @llvm.nvvm.fma.f32(i32 1, float %57, float %56, float 0.000000e+00)
  %59 = bitcast i32 %47 to float
  %60 = call float @llvm.nvvm.fma.f32(i32 1, float %54, float %58, float %59)
  %61 = call float @llvm.nvvm.fma.f32(i32 1, float %60, float %56, float %58)
  %62 = bitcast i32 %47 to float
  %63 = call float @llvm.nvvm.fma.f32(i32 1, float %54, float %61, float %62)
  %64 = call float @llvm.nvvm.fma.ftz.f32(i32 1, float %63, float %56, float %61)
  %65 = bitcast float %64 to i32
  %66 = lshr i32 %65, 23
  %67 = and i32 %66, 255
  %68 = sub nsw i32 %45, %48
  %69 = add nsw i32 %67, %68
  %70 = add nsw i32 %69, -1
  %71 = icmp ult i32 %70, 254
  br i1 %71, label %79, label %72

72:                                               ; preds = %44
  %73 = icmp sgt i32 %69, 254
  %74 = and i32 %65, -2147483648
  br i1 %73, label %77, label %75

75:                                               ; preds = %72
  %76 = icmp slt i32 %69, 1
  %. = select i1 %76, i32 %74, i32 %65
  br label %82

77:                                               ; preds = %72
  %78 = or i32 %74, 2139095040
  br label %82

79:                                               ; preds = %44
  %80 = shl i32 %68, 23
  %81 = add nsw i32 %80, %65
  br label %82

82:                                               ; preds = %79, %77, %75
  %.01 = phi i32 [ %81, %79 ], [ %78, %77 ], [ %., %75 ]
  %83 = bitcast i32 %.01 to float
  br label %97

84:                                               ; preds = %42
  %85 = xor i32 %7, %3
  %86 = and i32 %85, -2147483648
  %87 = or i32 %86, 2139095040
  %88 = bitcast i32 %87 to float
  br label %97

89:                                               ; preds = %40
  %90 = xor i32 %7, %3
  %91 = and i32 %90, -2147483648
  %92 = bitcast i32 %91 to float
  br label %97

93:                                               ; preds = %32, %24
  %94 = call float @llvm.nvvm.rsqrt.approx.ftz.f32(float 0xFFF8000000000000)
  br label %97

95:                                               ; preds = %19, %14
  %96 = call float @llvm.nvvm.add.ftz.f32(i32 1, float %0, float %1)
  br label %97

97:                                               ; preds = %95, %93, %89, %84, %82
  %.0 = phi float [ %96, %95 ], [ %94, %93 ], [ %92, %89 ], [ %88, %84 ], [ %83, %82 ]
  ret float %.0
}

; Unknown intrinsic
declare dso_local float @llvm.nvvm.fma.ftz.f32(i32, float, float, float) #2

; Function Attrs: noinline
define weak dso_local double @__cuda_sm20_div_f64_slowpath_v2(double %0, double %1) #0 {
  %3 = bitcast double %1 to i64
  %4 = and i64 %3, 4611686018427387904
  %5 = xor i64 %4, 6913025428013711360
  %6 = bitcast double %0 to i64
  %.sroa.0.4.extract.shift.i37 = lshr i64 %6, 32
  %.sroa.0.4.extract.trunc.i38 = trunc i64 %.sroa.0.4.extract.shift.i37 to i32
  %7 = and i32 %.sroa.0.4.extract.trunc.i38, 2139095040
  %8 = icmp ult i32 %7, 1048576000
  %9 = bitcast i64 %5 to double
  %10 = call double @llvm.nvvm.mul.f64(i32 1, double %9, double %1)
  %11 = select i1 %8, double 0x5FF0000000000000, double 0x1FF0000000000000
  %12 = call double @llvm.nvvm.mul.f64(i32 1, double %11, double %0)
  %13 = call double @llvm.nvvm.rcp.approx.ftz.f64(double %10)
  %14 = bitcast double %13 to i64
  %15 = bitcast double %13 to i64
  %.sroa.0.4.extract.shift.i27 = and i64 %15, -4294967296
  %16 = and i64 %14, 4294967294
  %.sroa.0.0.insert.ext.i22 = or i64 %16, 1
  %.sroa.0.4.insert.insert.i25 = or i64 %.sroa.0.4.extract.shift.i27, %.sroa.0.0.insert.ext.i22
  %17 = fsub double -0.000000e+00, %10
  %18 = bitcast i64 %.sroa.0.4.insert.insert.i25 to double
  %19 = call double @llvm.nvvm.fma.f64(i32 1, double %17, double %18, double 1.000000e+00)
  %20 = call double @llvm.nvvm.fma.f64(i32 1, double %19, double %19, double %19)
  %21 = bitcast i64 %.sroa.0.4.insert.insert.i25 to double
  %22 = bitcast i64 %.sroa.0.4.insert.insert.i25 to double
  %23 = call double @llvm.nvvm.fma.f64(i32 1, double %20, double %21, double %22)
  %24 = call double @llvm.nvvm.fma.f64(i32 1, double %17, double %23, double 1.000000e+00)
  %25 = call double @llvm.nvvm.fma.f64(i32 1, double %24, double %23, double %23)
  %26 = call double @llvm.nvvm.mul.f64(i32 1, double %12, double %25)
  %27 = call double @llvm.nvvm.fma.f64(i32 1, double %17, double %26, double %12)
  %28 = call double @llvm.nvvm.fma.f64(i32 1, double %27, double %25, double %26)
  %29 = call double @llvm.nvvm.fabs.f64(double %28)
  %30 = fcmp ule double %29, 0.000000e+00
  br i1 %30, label %31, label %35

31:                                               ; preds = %2
  %32 = fcmp oeq double %28, 0.000000e+00
  br i1 %32, label %33, label %84

33:                                               ; preds = %31
  %34 = call double @llvm.nvvm.mul.f64(i32 1, double %0, double %1)
  br label %93

35:                                               ; preds = %2
  %36 = icmp ugt i32 %7, 1048575999
  %37 = select i1 %36, i64 6913025428013711360, i64 2301339409586323456
  %38 = bitcast i64 %5 to double
  %39 = call double @llvm.nvvm.mul.f64(i32 1, double %28, double %38)
  %40 = bitcast i64 %37 to double
  %41 = call double @llvm.nvvm.mul.f64(i32 1, double %39, double %40)
  %42 = bitcast i64 %37 to double
  %43 = call double @llvm.nvvm.mul.f64(i32 1, double %28, double %42)
  %44 = bitcast i64 %5 to double
  %45 = call double @llvm.nvvm.mul.f64(i32 1, double %44, double %43)
  %46 = fsub double -0.000000e+00, %0
  %47 = call double @llvm.nvvm.fma.f64(i32 1, double %41, double %1, double %46)
  %48 = call double @llvm.nvvm.fma.f64(i32 1, double %45, double %1, double %46)
  %49 = call double @llvm.nvvm.fabs.f64(double %47)
  %50 = call double @llvm.nvvm.fabs.f64(double %48)
  %51 = fcmp ogt double %49, %50
  %. = select i1 %51, double %45, double %41
  %52 = bitcast double %. to i64
  %.sroa.0.4.extract.shift.i16 = lshr i64 %52, 32
  %.sroa.0.4.extract.trunc.i17 = trunc i64 %.sroa.0.4.extract.shift.i16 to i32
  %53 = bitcast i32 %.sroa.0.4.extract.trunc.i17 to float
  %54 = call float @llvm.nvvm.fabs.f32(float %53)
  %55 = fcmp ugt float %54, 0x37E0000000000000
  br i1 %55, label %93, label %56

56:                                               ; preds = %35
  %57 = bitcast double %. to i64
  %58 = and i64 %57, 4294967294
  %.sroa.0.4.insert.shift.i10 = and i64 %52, -4294967296
  %.sroa.0.4.insert.insert.i11 = or i64 %.sroa.0.4.insert.shift.i10, %58
  %59 = and i64 %57, 4294967294
  %.sroa.0.0.insert.ext.i4 = or i64 %59, 1
  %.sroa.0.4.insert.shift.i6 = and i64 %52, -4294967296
  %.sroa.0.4.insert.insert.i7 = or i64 %.sroa.0.4.insert.shift.i6, %.sroa.0.0.insert.ext.i4
  %60 = bitcast i32 %.sroa.0.4.extract.trunc.i38 to float
  %61 = call float @llvm.nvvm.fabs.f32(float %60)
  %62 = fcmp olt float %61, 0x3880000000000000
  %.sroa.0.4.insert.ext.i = select i1 %62, i64 6363586273474510848, i64 4607182418800017408
  %63 = bitcast i64 %.sroa.0.4.insert.ext.i to double
  %64 = call double @llvm.nvvm.mul.f64(i32 1, double %63, double %1)
  %65 = bitcast i64 %.sroa.0.4.insert.ext.i to double
  %66 = call double @llvm.nvvm.mul.f64(i32 1, double %65, double %0)
  %67 = fsub double -0.000000e+00, %66
  %68 = bitcast i64 %.sroa.0.4.insert.insert.i11 to double
  %69 = call double @llvm.nvvm.fma.f64(i32 1, double %68, double %64, double %67)
  %70 = bitcast i64 %.sroa.0.4.insert.insert.i7 to double
  %71 = call double @llvm.nvvm.fma.f64(i32 1, double %70, double %64, double %67)
  %72 = call double @llvm.nvvm.fabs.f64(double %69)
  %73 = call double @llvm.nvvm.fabs.f64(double %71)
  %74 = fcmp ogt double %72, %73
  %.sroa.0.4.insert.insert.i7..sroa.0.4.insert.insert.i11 = select i1 %74, i64 %.sroa.0.4.insert.insert.i7, i64 %.sroa.0.4.insert.insert.i11
  %75 = bitcast i64 %.sroa.0.4.insert.insert.i7..sroa.0.4.insert.insert.i11 to double
  %76 = add i64 %.sroa.0.4.insert.insert.i7..sroa.0.4.insert.insert.i11, -1
  %77 = bitcast i64 %76 to double
  %78 = call double @llvm.nvvm.fma.f64(i32 1, double %75, double %64, double %67)
  %79 = call double @llvm.nvvm.fma.f64(i32 1, double %77, double %64, double %67)
  %80 = call double @llvm.nvvm.fabs.f64(double %78)
  %81 = call double @llvm.nvvm.fabs.f64(double %79)
  %82 = fcmp ogt double %80, %81
  %83 = select i1 %82, double %77, double %75
  br label %93

84:                                               ; preds = %31
  %85 = call double @llvm.nvvm.rcp.approx.ftz.f64(double %1)
  %86 = call double @llvm.nvvm.fabs.f64(double %85)
  %87 = fcmp ogt double %86, 0.000000e+00
  br i1 %87, label %91, label %88

88:                                               ; preds = %84
  %89 = call double @llvm.nvvm.fabs.f64(double %1)
  %90 = fcmp une double %89, 0x7FF0000000000000
  %.63 = select i1 %90, double %1, double %85
  br label %91

91:                                               ; preds = %84, %88
  %.043 = phi double [ %.63, %88 ], [ %85, %84 ]
  %92 = call double @llvm.nvvm.mul.f64(i32 1, double %.043, double %0)
  br label %93

93:                                               ; preds = %56, %35, %91, %33
  %.0 = phi double [ %34, %33 ], [ %92, %91 ], [ %., %35 ], [ %83, %56 ]
  ret double %.0
}

; Unknown intrinsic
declare dso_local double @llvm.nvvm.mul.f64(i32, double, double) #2

; Unknown intrinsic
declare dso_local double @llvm.nvvm.rcp.approx.ftz.f64(double) #2

; Unknown intrinsic
declare dso_local double @llvm.nvvm.fma.f64(i32, double, double, double) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare dso_local double @llvm.nvvm.fabs.f64(double) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare dso_local float @llvm.nvvm.fabs.f32(float) #1

; Function Attrs: noinline
define weak dso_local double @__cuda_sm20_div_rz_f64(double %0, double %1) #0 {
  %3 = bitcast double %0 to i64
  %.sroa.0.0.extract.trunc.i = trunc i64 %3 to i32
  %4 = bitcast double %0 to i64
  %.sroa.0.4.extract.shift.i54 = lshr i64 %4, 32
  %.sroa.0.4.extract.trunc.i55 = trunc i64 %.sroa.0.4.extract.shift.i54 to i32
  %5 = bitcast double %1 to i64
  %6 = bitcast double %1 to i64
  %.sroa.0.4.extract.shift.i48 = lshr i64 %6, 32
  %.sroa.0.4.extract.trunc.i49 = trunc i64 %.sroa.0.4.extract.shift.i48 to i32
  %7 = lshr i64 %4, 52
  %8 = trunc i64 %7 to i32
  %9 = and i32 %8, 2047
  %10 = lshr i64 %6, 52
  %11 = trunc i64 %10 to i32
  %12 = and i32 %11, 2047
  %13 = add nsw i32 %9, -1
  %14 = add nsw i32 %12, -1
  %15 = icmp ult i32 %13, 2046
  %16 = icmp ult i32 %14, 2046
  %17 = and i1 %16, %15
  br i1 %17, label %59, label %18

18:                                               ; preds = %2
  %19 = call double @llvm.nvvm.fabs.f64(double %0)
  %20 = fcmp ugt double %19, 0x7FF0000000000000
  br i1 %20, label %21, label %23

21:                                               ; preds = %18
  %.sroa.0.0.insert.ext.i43 = and i64 %3, 4294967295
  %22 = and i64 %4, -2251804108652544
  %.sroa.0.4.insert.shift.i45 = or i64 %22, %.sroa.0.0.insert.ext.i43
  %.sroa.0.4.insert.insert.i46 = or i64 %.sroa.0.4.insert.shift.i45, 2251799813685248
  br label %121

23:                                               ; preds = %18
  %24 = call double @llvm.nvvm.fabs.f64(double %1)
  %25 = fcmp ugt double %24, 0x7FF0000000000000
  br i1 %25, label %26, label %28

26:                                               ; preds = %23
  %.sroa.0.0.insert.ext.i39 = and i64 %5, 4294967295
  %27 = and i64 %6, -2251804108652544
  %.sroa.0.4.insert.shift.i41 = or i64 %27, %.sroa.0.0.insert.ext.i39
  %.sroa.0.4.insert.insert.i42 = or i64 %.sroa.0.4.insert.shift.i41, 2251799813685248
  br label %121

28:                                               ; preds = %23
  %29 = fcmp oeq double %0, 0.000000e+00
  %30 = fcmp oeq double %1, 0.000000e+00
  %31 = and i1 %29, %30
  br i1 %31, label %121, label %32

32:                                               ; preds = %28
  %33 = fcmp oeq double %19, 0x7FF0000000000000
  %34 = fcmp oeq double %24, 0x7FF0000000000000
  %.not = xor i1 %33, true
  %.not80 = xor i1 %34, true
  %brmerge = or i1 %.not, %.not80
  br i1 %brmerge, label %35, label %121

35:                                               ; preds = %32
  %36 = or i1 %34, %29
  br i1 %36, label %37, label %40

37:                                               ; preds = %35
  %38 = xor i64 %.sroa.0.4.extract.shift.i48, %.sroa.0.4.extract.shift.i54
  %39 = shl nuw i64 %38, 32
  %.sroa.0.4.insert.shift.i33 = and i64 %39, -9223372036854775808
  br label %121

40:                                               ; preds = %35
  %41 = or i1 %33, %30
  br i1 %41, label %42, label %46

42:                                               ; preds = %40
  %43 = xor i64 %.sroa.0.4.extract.shift.i48, %.sroa.0.4.extract.shift.i54
  %44 = shl nuw i64 %43, 32
  %45 = and i64 %44, -9223372036854775808
  %.sroa.0.4.insert.shift.i29 = or i64 %45, 9218868437227405312
  br label %121

46:                                               ; preds = %40
  %47 = icmp eq i32 %9, 0
  br i1 %47, label %48, label %52

48:                                               ; preds = %46
  %49 = call double @llvm.nvvm.mul.f64(i32 1, double %0, double 0x4350000000000000)
  %50 = bitcast double %49 to i64
  %.sroa.0.0.extract.trunc.i24 = trunc i64 %50 to i32
  %51 = bitcast double %49 to i64
  %.sroa.0.4.extract.shift.i22 = lshr i64 %51, 32
  %.sroa.0.4.extract.trunc.i23 = trunc i64 %.sroa.0.4.extract.shift.i22 to i32
  br label %52

52:                                               ; preds = %46, %48
  %.061 = phi i32 [ %.sroa.0.0.extract.trunc.i24, %48 ], [ %.sroa.0.0.extract.trunc.i, %46 ]
  %.059 = phi i32 [ %.sroa.0.4.extract.trunc.i23, %48 ], [ %.sroa.0.4.extract.trunc.i55, %46 ]
  %.056 = phi i32 [ -54, %48 ], [ 0, %46 ]
  %53 = icmp eq i32 %12, 0
  br i1 %53, label %54, label %59

54:                                               ; preds = %52
  %55 = call double @llvm.nvvm.mul.f64(i32 1, double %1, double 0x4350000000000000)
  %56 = bitcast double %55 to i64
  %57 = bitcast double %55 to i64
  %.sroa.0.4.extract.shift.i16 = lshr i64 %57, 32
  %.sroa.0.4.extract.trunc.i17 = trunc i64 %.sroa.0.4.extract.shift.i16 to i32
  %58 = add nsw i32 %.056, 54
  br label %59

59:                                               ; preds = %2, %52, %54
  %.162 = phi i32 [ %.061, %54 ], [ %.061, %52 ], [ %.sroa.0.0.extract.trunc.i, %2 ]
  %.160 = phi i32 [ %.059, %54 ], [ %.059, %52 ], [ %.sroa.0.4.extract.trunc.i55, %2 ]
  %.058 = phi i64 [ %56, %54 ], [ %5, %52 ], [ %5, %2 ]
  %.057 = phi i32 [ %.sroa.0.4.extract.trunc.i17, %54 ], [ %.sroa.0.4.extract.trunc.i49, %52 ], [ %.sroa.0.4.extract.trunc.i49, %2 ]
  %.1 = phi i32 [ %58, %54 ], [ %.056, %52 ], [ 0, %2 ]
  %60 = shl nuw nsw i32 %9, 20
  %61 = add nsw i32 %60, -1072693248
  %62 = sub i32 %.160, %61
  %63 = shl nuw nsw i32 %12, 20
  %64 = add nsw i32 %63, -1072693248
  %65 = sub i32 %.057, %64
  %.sroa.0.0.insert.ext.i11 = and i64 %.058, 4294967295
  %.sroa.0.4.insert.ext.i12 = zext i32 %65 to i64
  %.sroa.0.4.insert.shift.i13 = shl nuw i64 %.sroa.0.4.insert.ext.i12, 32
  %.sroa.0.4.insert.insert.i14 = or i64 %.sroa.0.4.insert.shift.i13, %.sroa.0.0.insert.ext.i11
  %66 = bitcast i64 %.sroa.0.4.insert.insert.i14 to double
  %67 = call double @llvm.nvvm.rcp.approx.ftz.f64(double %66)
  %68 = bitcast i64 %.sroa.0.4.insert.insert.i14 to double
  %69 = fsub double -0.000000e+00, %68
  %70 = bitcast double %67 to i64
  %71 = or i64 %70, 1
  %72 = bitcast i64 %71 to double
  %73 = call double @llvm.nvvm.fma.f64(i32 1, double %69, double %72, double 1.000000e+00)
  %74 = bitcast i64 %71 to double
  %75 = bitcast i64 %71 to double
  %76 = call double @llvm.nvvm.fma.f64(i32 1, double %73, double %74, double %75)
  %77 = call double @llvm.nvvm.mul.f64(i32 1, double %73, double %73)
  %78 = call double @llvm.nvvm.fma.f64(i32 1, double %77, double %76, double %76)
  %79 = call double @llvm.nvvm.fma.f64(i32 1, double %69, double %78, double 1.000000e+00)
  %80 = call double @llvm.nvvm.fma.f64(i32 1, double %79, double %78, double %78)
  %.sroa.0.0.insert.ext.i7 = zext i32 %.162 to i64
  %.sroa.0.4.insert.ext.i8 = zext i32 %62 to i64
  %.sroa.0.4.insert.shift.i9 = shl nuw i64 %.sroa.0.4.insert.ext.i8, 32
  %.sroa.0.4.insert.insert.i10 = or i64 %.sroa.0.4.insert.shift.i9, %.sroa.0.0.insert.ext.i7
  %81 = bitcast i64 %.sroa.0.4.insert.insert.i10 to double
  %82 = bitcast i64 %71 to double
  %83 = call double @llvm.nvvm.mul.f64(i32 1, double %81, double %82)
  %84 = bitcast i64 %.sroa.0.4.insert.insert.i10 to double
  %85 = call double @llvm.nvvm.fma.f64(i32 1, double %69, double %83, double %84)
  %86 = call double @llvm.nvvm.fma.f64(i32 1, double %85, double %78, double %83)
  %87 = bitcast i64 %.sroa.0.4.insert.insert.i10 to double
  %88 = call double @llvm.nvvm.fma.f64(i32 1, double %69, double %86, double %87)
  %89 = call double @llvm.nvvm.fma.f64(i32 4, double %88, double %80, double %86)
  %90 = bitcast double %89 to i64
  %.sroa.0.4.extract.shift.i5 = lshr i64 %90, 32
  %91 = lshr i64 %90, 52
  %92 = trunc i64 %91 to i32
  %93 = and i32 %92, 2047
  %94 = sub nsw i32 %9, %12
  %95 = add nsw i32 %94, %93
  %96 = add nsw i32 %95, %.1
  %97 = add nsw i32 %96, -1
  %98 = icmp ugt i32 %97, 2045
  br i1 %98, label %104, label %99

99:                                               ; preds = %59
  %.sroa.0.4.extract.trunc.i6 = trunc i64 %.sroa.0.4.extract.shift.i5 to i32
  %100 = sub nsw i32 %96, %93
  %101 = shl i32 %100, 20
  %102 = add i32 %101, %.sroa.0.4.extract.trunc.i6
  %103 = bitcast double %89 to i64
  %.sroa.0.0.insert.ext.i = and i64 %103, 4294967295
  %.sroa.0.4.insert.ext.i = zext i32 %102 to i64
  %.sroa.0.4.insert.shift.i = shl nuw i64 %.sroa.0.4.insert.ext.i, 32
  %.sroa.0.4.insert.insert.i = or i64 %.sroa.0.4.insert.shift.i, %.sroa.0.0.insert.ext.i
  br label %121

104:                                              ; preds = %59
  %105 = icmp slt i32 %96, 2047
  br i1 %105, label %106, label %115

106:                                              ; preds = %104
  %107 = icmp sgt i32 %96, -54
  br i1 %107, label %108, label %115

108:                                              ; preds = %106
  %109 = bitcast double %89 to i64
  %110 = and i64 %109, 4503599627370495
  %111 = or i64 %110, 4503599627370496
  %112 = sub nsw i32 1, %96
  %113 = zext i32 %112 to i64
  %114 = lshr i64 %111, %113
  br label %115

115:                                              ; preds = %108, %104, %106
  %.0 = phi i64 [ %114, %108 ], [ 9218868437227405311, %104 ], [ 0, %106 ]
  %116 = trunc i64 %.sroa.0.4.extract.shift.i5 to i32
  %117 = and i32 %116, -2147483648
  %118 = call i64 @llvm.nvvm.cvt.i64.i32(i32 4, i32 %117)
  %119 = shl i64 %118, 32
  %120 = or i64 %.0, %119
  br label %121

121:                                              ; preds = %32, %99, %115, %28, %42, %37, %26, %21
  %.sroa.075.0 = phi i64 [ %.sroa.0.4.insert.shift.i29, %42 ], [ %.sroa.0.4.insert.shift.i33, %37 ], [ %.sroa.0.4.insert.insert.i42, %26 ], [ %.sroa.0.4.insert.insert.i46, %21 ], [ -2251799813685248, %28 ], [ %120, %115 ], [ %.sroa.0.4.insert.insert.i, %99 ], [ -2251799813685248, %32 ]
  %122 = bitcast i64 %.sroa.075.0 to double
  ret double %122
}

; Unknown intrinsic
declare dso_local i64 @llvm.nvvm.cvt.i64.i32(i32, i32) #2

; Function Attrs: noinline
define weak dso_local double @__cuda_sm20_div_ru_f64(double %0, double %1) #0 {
  %3 = bitcast double %0 to i64
  %.sroa.0.0.extract.trunc.i = trunc i64 %3 to i32
  %4 = bitcast double %0 to i64
  %.sroa.0.4.extract.shift.i54 = lshr i64 %4, 32
  %.sroa.0.4.extract.trunc.i55 = trunc i64 %.sroa.0.4.extract.shift.i54 to i32
  %5 = bitcast double %1 to i64
  %6 = bitcast double %1 to i64
  %.sroa.0.4.extract.shift.i48 = lshr i64 %6, 32
  %.sroa.0.4.extract.trunc.i49 = trunc i64 %.sroa.0.4.extract.shift.i48 to i32
  %7 = lshr i64 %4, 52
  %8 = trunc i64 %7 to i32
  %9 = and i32 %8, 2047
  %10 = lshr i64 %6, 52
  %11 = trunc i64 %10 to i32
  %12 = and i32 %11, 2047
  %13 = add nsw i32 %9, -1
  %14 = add nsw i32 %12, -1
  %15 = icmp ult i32 %13, 2046
  %16 = icmp ult i32 %14, 2046
  %17 = and i1 %16, %15
  br i1 %17, label %59, label %18

18:                                               ; preds = %2
  %19 = call double @llvm.nvvm.fabs.f64(double %0)
  %20 = fcmp ugt double %19, 0x7FF0000000000000
  br i1 %20, label %21, label %23

21:                                               ; preds = %18
  %.sroa.0.0.insert.ext.i43 = and i64 %3, 4294967295
  %22 = and i64 %4, -2251804108652544
  %.sroa.0.4.insert.shift.i45 = or i64 %22, %.sroa.0.0.insert.ext.i43
  %.sroa.0.4.insert.insert.i46 = or i64 %.sroa.0.4.insert.shift.i45, 2251799813685248
  br label %141

23:                                               ; preds = %18
  %24 = call double @llvm.nvvm.fabs.f64(double %1)
  %25 = fcmp ugt double %24, 0x7FF0000000000000
  br i1 %25, label %26, label %28

26:                                               ; preds = %23
  %.sroa.0.0.insert.ext.i39 = and i64 %5, 4294967295
  %27 = and i64 %6, -2251804108652544
  %.sroa.0.4.insert.shift.i41 = or i64 %27, %.sroa.0.0.insert.ext.i39
  %.sroa.0.4.insert.insert.i42 = or i64 %.sroa.0.4.insert.shift.i41, 2251799813685248
  br label %141

28:                                               ; preds = %23
  %29 = fcmp oeq double %0, 0.000000e+00
  %30 = fcmp oeq double %1, 0.000000e+00
  %31 = and i1 %29, %30
  br i1 %31, label %141, label %32

32:                                               ; preds = %28
  %33 = fcmp oeq double %19, 0x7FF0000000000000
  %34 = fcmp oeq double %24, 0x7FF0000000000000
  %.not = xor i1 %33, true
  %.not80 = xor i1 %34, true
  %brmerge = or i1 %.not, %.not80
  br i1 %brmerge, label %35, label %141

35:                                               ; preds = %32
  %36 = or i1 %34, %29
  br i1 %36, label %37, label %40

37:                                               ; preds = %35
  %38 = xor i64 %.sroa.0.4.extract.shift.i48, %.sroa.0.4.extract.shift.i54
  %39 = shl nuw i64 %38, 32
  %.sroa.0.4.insert.shift.i33 = and i64 %39, -9223372036854775808
  br label %141

40:                                               ; preds = %35
  %41 = or i1 %33, %30
  br i1 %41, label %42, label %46

42:                                               ; preds = %40
  %43 = xor i64 %.sroa.0.4.extract.shift.i48, %.sroa.0.4.extract.shift.i54
  %44 = shl nuw i64 %43, 32
  %45 = and i64 %44, -9223372036854775808
  %.sroa.0.4.insert.shift.i29 = or i64 %45, 9218868437227405312
  br label %141

46:                                               ; preds = %40
  %47 = icmp eq i32 %9, 0
  br i1 %47, label %48, label %52

48:                                               ; preds = %46
  %49 = call double @llvm.nvvm.mul.f64(i32 1, double %0, double 0x4350000000000000)
  %50 = bitcast double %49 to i64
  %.sroa.0.0.extract.trunc.i24 = trunc i64 %50 to i32
  %51 = bitcast double %49 to i64
  %.sroa.0.4.extract.shift.i22 = lshr i64 %51, 32
  %.sroa.0.4.extract.trunc.i23 = trunc i64 %.sroa.0.4.extract.shift.i22 to i32
  br label %52

52:                                               ; preds = %46, %48
  %.061 = phi i32 [ %.sroa.0.0.extract.trunc.i24, %48 ], [ %.sroa.0.0.extract.trunc.i, %46 ]
  %.059 = phi i32 [ %.sroa.0.4.extract.trunc.i23, %48 ], [ %.sroa.0.4.extract.trunc.i55, %46 ]
  %.056 = phi i32 [ -54, %48 ], [ 0, %46 ]
  %53 = icmp eq i32 %12, 0
  br i1 %53, label %54, label %59

54:                                               ; preds = %52
  %55 = call double @llvm.nvvm.mul.f64(i32 1, double %1, double 0x4350000000000000)
  %56 = bitcast double %55 to i64
  %57 = bitcast double %55 to i64
  %.sroa.0.4.extract.shift.i16 = lshr i64 %57, 32
  %.sroa.0.4.extract.trunc.i17 = trunc i64 %.sroa.0.4.extract.shift.i16 to i32
  %58 = add nsw i32 %.056, 54
  br label %59

59:                                               ; preds = %2, %52, %54
  %.162 = phi i32 [ %.061, %54 ], [ %.061, %52 ], [ %.sroa.0.0.extract.trunc.i, %2 ]
  %.160 = phi i32 [ %.059, %54 ], [ %.059, %52 ], [ %.sroa.0.4.extract.trunc.i55, %2 ]
  %.058 = phi i64 [ %56, %54 ], [ %5, %52 ], [ %5, %2 ]
  %.057 = phi i32 [ %.sroa.0.4.extract.trunc.i17, %54 ], [ %.sroa.0.4.extract.trunc.i49, %52 ], [ %.sroa.0.4.extract.trunc.i49, %2 ]
  %.1 = phi i32 [ %58, %54 ], [ %.056, %52 ], [ 0, %2 ]
  %60 = shl nuw nsw i32 %9, 20
  %61 = add nsw i32 %60, -1072693248
  %62 = sub i32 %.160, %61
  %63 = shl nuw nsw i32 %12, 20
  %64 = add nsw i32 %63, -1072693248
  %65 = sub i32 %.057, %64
  %.sroa.0.0.insert.ext.i11 = and i64 %.058, 4294967295
  %.sroa.0.4.insert.ext.i12 = zext i32 %65 to i64
  %.sroa.0.4.insert.shift.i13 = shl nuw i64 %.sroa.0.4.insert.ext.i12, 32
  %.sroa.0.4.insert.insert.i14 = or i64 %.sroa.0.4.insert.shift.i13, %.sroa.0.0.insert.ext.i11
  %66 = bitcast i64 %.sroa.0.4.insert.insert.i14 to double
  %67 = call double @llvm.nvvm.rcp.approx.ftz.f64(double %66)
  %68 = bitcast i64 %.sroa.0.4.insert.insert.i14 to double
  %69 = fsub double -0.000000e+00, %68
  %70 = bitcast double %67 to i64
  %71 = or i64 %70, 1
  %72 = bitcast i64 %71 to double
  %73 = call double @llvm.nvvm.fma.f64(i32 1, double %69, double %72, double 1.000000e+00)
  %74 = bitcast i64 %71 to double
  %75 = bitcast i64 %71 to double
  %76 = call double @llvm.nvvm.fma.f64(i32 1, double %73, double %74, double %75)
  %77 = call double @llvm.nvvm.mul.f64(i32 1, double %73, double %73)
  %78 = call double @llvm.nvvm.fma.f64(i32 1, double %77, double %76, double %76)
  %79 = call double @llvm.nvvm.fma.f64(i32 1, double %69, double %78, double 1.000000e+00)
  %80 = call double @llvm.nvvm.fma.f64(i32 1, double %79, double %78, double %78)
  %.sroa.0.0.insert.ext.i7 = zext i32 %.162 to i64
  %.sroa.0.4.insert.ext.i8 = zext i32 %62 to i64
  %.sroa.0.4.insert.shift.i9 = shl nuw i64 %.sroa.0.4.insert.ext.i8, 32
  %.sroa.0.4.insert.insert.i10 = or i64 %.sroa.0.4.insert.shift.i9, %.sroa.0.0.insert.ext.i7
  %81 = bitcast i64 %.sroa.0.4.insert.insert.i10 to double
  %82 = bitcast i64 %71 to double
  %83 = call double @llvm.nvvm.mul.f64(i32 1, double %81, double %82)
  %84 = bitcast i64 %.sroa.0.4.insert.insert.i10 to double
  %85 = call double @llvm.nvvm.fma.f64(i32 1, double %69, double %83, double %84)
  %86 = call double @llvm.nvvm.fma.f64(i32 1, double %85, double %78, double %83)
  %87 = bitcast i64 %.sroa.0.4.insert.insert.i10 to double
  %88 = call double @llvm.nvvm.fma.f64(i32 1, double %69, double %86, double %87)
  %89 = call double @llvm.nvvm.fma.f64(i32 3, double %88, double %80, double %86)
  %90 = bitcast double %89 to i64
  %.sroa.0.4.extract.shift.i5 = lshr i64 %90, 32
  %91 = lshr i64 %90, 52
  %92 = trunc i64 %91 to i32
  %93 = and i32 %92, 2047
  %94 = sub nsw i32 %9, %12
  %95 = add nsw i32 %94, %93
  %96 = add nsw i32 %95, %.1
  %97 = add nsw i32 %96, -1
  %98 = icmp ugt i32 %97, 2045
  %99 = trunc i64 %.sroa.0.4.extract.shift.i5 to i32
  br i1 %98, label %105, label %100

100:                                              ; preds = %59
  %101 = sub nsw i32 %96, %93
  %102 = shl i32 %101, 20
  %103 = add i32 %102, %99
  %104 = bitcast double %89 to i64
  %.sroa.0.0.insert.ext.i = and i64 %104, 4294967295
  %.sroa.0.4.insert.ext.i = zext i32 %103 to i64
  %.sroa.0.4.insert.shift.i = shl nuw i64 %.sroa.0.4.insert.ext.i, 32
  %.sroa.0.4.insert.insert.i = or i64 %.sroa.0.4.insert.shift.i, %.sroa.0.0.insert.ext.i
  br label %141

105:                                              ; preds = %59
  %106 = and i32 %99, -2147483648
  %107 = icmp slt i32 %96, 2047
  br i1 %107, label %110, label %108

108:                                              ; preds = %105
  %109 = icmp eq i32 %106, 0
  %. = select i1 %109, i64 9218868437227405312, i64 9218868437227405311
  br label %137

110:                                              ; preds = %105
  %111 = lshr i64 %90, 63
  %112 = trunc i64 %111 to i32
  %113 = xor i32 %112, 1
  %114 = icmp sgt i32 %96, -54
  br i1 %114, label %117, label %115

115:                                              ; preds = %110
  %116 = call i64 @llvm.nvvm.cvt.i64.i32(i32 4, i32 %113)
  br label %137

117:                                              ; preds = %110
  %118 = call double @llvm.nvvm.fma.f64(i32 2, double %88, double %80, double %86)
  %119 = fcmp one double %89, %118
  %120 = call double @llvm.nvvm.fma.f64(i32 4, double %88, double %80, double %86)
  %121 = bitcast double %120 to i64
  %122 = and i64 %121, 4503599627370495
  %123 = or i64 %122, 4503599627370496
  %124 = add nsw i32 %96, 63
  %125 = zext i32 %124 to i64
  %126 = shl i64 %123, %125
  %127 = icmp ne i64 %126, 0
  %128 = or i1 %119, %127
  %129 = sub nsw i32 1, %96
  %130 = zext i32 %129 to i64
  %131 = lshr i64 %123, %130
  %132 = zext i1 %128 to i32
  %133 = and i32 %113, %132
  %134 = icmp eq i32 %133, 1
  %135 = add i64 %131, 1
  %136 = select i1 %134, i64 %135, i64 %131
  br label %137

137:                                              ; preds = %108, %115, %117
  %.0 = phi i64 [ %116, %115 ], [ %., %108 ], [ %136, %117 ]
  %138 = call i64 @llvm.nvvm.cvt.i64.i32(i32 4, i32 %106)
  %139 = shl i64 %138, 32
  %140 = or i64 %.0, %139
  br label %141

141:                                              ; preds = %32, %100, %137, %28, %42, %37, %26, %21
  %.sroa.075.0 = phi i64 [ %.sroa.0.4.insert.shift.i29, %42 ], [ %.sroa.0.4.insert.shift.i33, %37 ], [ %.sroa.0.4.insert.insert.i42, %26 ], [ %.sroa.0.4.insert.insert.i46, %21 ], [ -2251799813685248, %28 ], [ %140, %137 ], [ %.sroa.0.4.insert.insert.i, %100 ], [ -2251799813685248, %32 ]
  %142 = bitcast i64 %.sroa.075.0 to double
  ret double %142
}

; Function Attrs: noinline
define weak dso_local double @__cuda_sm20_div_rd_f64(double %0, double %1) #0 {
  %3 = bitcast double %0 to i64
  %.sroa.0.0.extract.trunc.i = trunc i64 %3 to i32
  %4 = bitcast double %0 to i64
  %.sroa.0.4.extract.shift.i54 = lshr i64 %4, 32
  %.sroa.0.4.extract.trunc.i55 = trunc i64 %.sroa.0.4.extract.shift.i54 to i32
  %5 = bitcast double %1 to i64
  %6 = bitcast double %1 to i64
  %.sroa.0.4.extract.shift.i48 = lshr i64 %6, 32
  %.sroa.0.4.extract.trunc.i49 = trunc i64 %.sroa.0.4.extract.shift.i48 to i32
  %7 = lshr i64 %4, 52
  %8 = trunc i64 %7 to i32
  %9 = and i32 %8, 2047
  %10 = lshr i64 %6, 52
  %11 = trunc i64 %10 to i32
  %12 = and i32 %11, 2047
  %13 = add nsw i32 %9, -1
  %14 = add nsw i32 %12, -1
  %15 = icmp ult i32 %13, 2046
  %16 = icmp ult i32 %14, 2046
  %17 = and i1 %16, %15
  br i1 %17, label %59, label %18

18:                                               ; preds = %2
  %19 = call double @llvm.nvvm.fabs.f64(double %0)
  %20 = fcmp ugt double %19, 0x7FF0000000000000
  br i1 %20, label %21, label %23

21:                                               ; preds = %18
  %.sroa.0.0.insert.ext.i43 = and i64 %3, 4294967295
  %22 = and i64 %4, -2251804108652544
  %.sroa.0.4.insert.shift.i45 = or i64 %22, %.sroa.0.0.insert.ext.i43
  %.sroa.0.4.insert.insert.i46 = or i64 %.sroa.0.4.insert.shift.i45, 2251799813685248
  br label %139

23:                                               ; preds = %18
  %24 = call double @llvm.nvvm.fabs.f64(double %1)
  %25 = fcmp ugt double %24, 0x7FF0000000000000
  br i1 %25, label %26, label %28

26:                                               ; preds = %23
  %.sroa.0.0.insert.ext.i39 = and i64 %5, 4294967295
  %27 = and i64 %6, -2251804108652544
  %.sroa.0.4.insert.shift.i41 = or i64 %27, %.sroa.0.0.insert.ext.i39
  %.sroa.0.4.insert.insert.i42 = or i64 %.sroa.0.4.insert.shift.i41, 2251799813685248
  br label %139

28:                                               ; preds = %23
  %29 = fcmp oeq double %0, 0.000000e+00
  %30 = fcmp oeq double %1, 0.000000e+00
  %31 = and i1 %29, %30
  br i1 %31, label %139, label %32

32:                                               ; preds = %28
  %33 = fcmp oeq double %19, 0x7FF0000000000000
  %34 = fcmp oeq double %24, 0x7FF0000000000000
  %.not = xor i1 %33, true
  %.not81 = xor i1 %34, true
  %brmerge = or i1 %.not, %.not81
  br i1 %brmerge, label %35, label %139

35:                                               ; preds = %32
  %36 = or i1 %34, %29
  br i1 %36, label %37, label %40

37:                                               ; preds = %35
  %38 = xor i64 %.sroa.0.4.extract.shift.i48, %.sroa.0.4.extract.shift.i54
  %39 = shl nuw i64 %38, 32
  %.sroa.0.4.insert.shift.i33 = and i64 %39, -9223372036854775808
  br label %139

40:                                               ; preds = %35
  %41 = or i1 %33, %30
  br i1 %41, label %42, label %46

42:                                               ; preds = %40
  %43 = xor i64 %.sroa.0.4.extract.shift.i48, %.sroa.0.4.extract.shift.i54
  %44 = shl nuw i64 %43, 32
  %45 = and i64 %44, -9223372036854775808
  %.sroa.0.4.insert.shift.i29 = or i64 %45, 9218868437227405312
  br label %139

46:                                               ; preds = %40
  %47 = icmp eq i32 %9, 0
  br i1 %47, label %48, label %52

48:                                               ; preds = %46
  %49 = call double @llvm.nvvm.mul.f64(i32 1, double %0, double 0x4350000000000000)
  %50 = bitcast double %49 to i64
  %.sroa.0.0.extract.trunc.i24 = trunc i64 %50 to i32
  %51 = bitcast double %49 to i64
  %.sroa.0.4.extract.shift.i22 = lshr i64 %51, 32
  %.sroa.0.4.extract.trunc.i23 = trunc i64 %.sroa.0.4.extract.shift.i22 to i32
  br label %52

52:                                               ; preds = %46, %48
  %.061 = phi i32 [ %.sroa.0.0.extract.trunc.i24, %48 ], [ %.sroa.0.0.extract.trunc.i, %46 ]
  %.059 = phi i32 [ %.sroa.0.4.extract.trunc.i23, %48 ], [ %.sroa.0.4.extract.trunc.i55, %46 ]
  %.056 = phi i32 [ -54, %48 ], [ 0, %46 ]
  %53 = icmp eq i32 %12, 0
  br i1 %53, label %54, label %59

54:                                               ; preds = %52
  %55 = call double @llvm.nvvm.mul.f64(i32 1, double %1, double 0x4350000000000000)
  %56 = bitcast double %55 to i64
  %57 = bitcast double %55 to i64
  %.sroa.0.4.extract.shift.i16 = lshr i64 %57, 32
  %.sroa.0.4.extract.trunc.i17 = trunc i64 %.sroa.0.4.extract.shift.i16 to i32
  %58 = add nsw i32 %.056, 54
  br label %59

59:                                               ; preds = %2, %52, %54
  %.162 = phi i32 [ %.061, %54 ], [ %.061, %52 ], [ %.sroa.0.0.extract.trunc.i, %2 ]
  %.160 = phi i32 [ %.059, %54 ], [ %.059, %52 ], [ %.sroa.0.4.extract.trunc.i55, %2 ]
  %.058 = phi i64 [ %56, %54 ], [ %5, %52 ], [ %5, %2 ]
  %.057 = phi i32 [ %.sroa.0.4.extract.trunc.i17, %54 ], [ %.sroa.0.4.extract.trunc.i49, %52 ], [ %.sroa.0.4.extract.trunc.i49, %2 ]
  %.1 = phi i32 [ %58, %54 ], [ %.056, %52 ], [ 0, %2 ]
  %60 = shl nuw nsw i32 %9, 20
  %61 = add nsw i32 %60, -1072693248
  %62 = sub i32 %.160, %61
  %63 = shl nuw nsw i32 %12, 20
  %64 = add nsw i32 %63, -1072693248
  %65 = sub i32 %.057, %64
  %.sroa.0.0.insert.ext.i11 = and i64 %.058, 4294967295
  %.sroa.0.4.insert.ext.i12 = zext i32 %65 to i64
  %.sroa.0.4.insert.shift.i13 = shl nuw i64 %.sroa.0.4.insert.ext.i12, 32
  %.sroa.0.4.insert.insert.i14 = or i64 %.sroa.0.4.insert.shift.i13, %.sroa.0.0.insert.ext.i11
  %66 = bitcast i64 %.sroa.0.4.insert.insert.i14 to double
  %67 = call double @llvm.nvvm.rcp.approx.ftz.f64(double %66)
  %68 = bitcast i64 %.sroa.0.4.insert.insert.i14 to double
  %69 = fsub double -0.000000e+00, %68
  %70 = bitcast double %67 to i64
  %71 = or i64 %70, 1
  %72 = bitcast i64 %71 to double
  %73 = call double @llvm.nvvm.fma.f64(i32 1, double %69, double %72, double 1.000000e+00)
  %74 = bitcast i64 %71 to double
  %75 = bitcast i64 %71 to double
  %76 = call double @llvm.nvvm.fma.f64(i32 1, double %73, double %74, double %75)
  %77 = call double @llvm.nvvm.mul.f64(i32 1, double %73, double %73)
  %78 = call double @llvm.nvvm.fma.f64(i32 1, double %77, double %76, double %76)
  %79 = call double @llvm.nvvm.fma.f64(i32 1, double %69, double %78, double 1.000000e+00)
  %80 = call double @llvm.nvvm.fma.f64(i32 1, double %79, double %78, double %78)
  %.sroa.0.0.insert.ext.i7 = zext i32 %.162 to i64
  %.sroa.0.4.insert.ext.i8 = zext i32 %62 to i64
  %.sroa.0.4.insert.shift.i9 = shl nuw i64 %.sroa.0.4.insert.ext.i8, 32
  %.sroa.0.4.insert.insert.i10 = or i64 %.sroa.0.4.insert.shift.i9, %.sroa.0.0.insert.ext.i7
  %81 = bitcast i64 %.sroa.0.4.insert.insert.i10 to double
  %82 = bitcast i64 %71 to double
  %83 = call double @llvm.nvvm.mul.f64(i32 1, double %81, double %82)
  %84 = bitcast i64 %.sroa.0.4.insert.insert.i10 to double
  %85 = call double @llvm.nvvm.fma.f64(i32 1, double %69, double %83, double %84)
  %86 = call double @llvm.nvvm.fma.f64(i32 1, double %85, double %78, double %83)
  %87 = bitcast i64 %.sroa.0.4.insert.insert.i10 to double
  %88 = call double @llvm.nvvm.fma.f64(i32 1, double %69, double %86, double %87)
  %89 = call double @llvm.nvvm.fma.f64(i32 2, double %88, double %80, double %86)
  %90 = bitcast double %89 to i64
  %.sroa.0.4.extract.shift.i5 = lshr i64 %90, 32
  %91 = lshr i64 %90, 52
  %92 = trunc i64 %91 to i32
  %93 = and i32 %92, 2047
  %94 = sub nsw i32 %9, %12
  %95 = add nsw i32 %94, %93
  %96 = add nsw i32 %95, %.1
  %97 = add nsw i32 %96, -1
  %98 = icmp ugt i32 %97, 2045
  %99 = trunc i64 %.sroa.0.4.extract.shift.i5 to i32
  br i1 %98, label %105, label %100

100:                                              ; preds = %59
  %101 = sub nsw i32 %96, %93
  %102 = shl i32 %101, 20
  %103 = add i32 %102, %99
  %104 = bitcast double %89 to i64
  %.sroa.0.0.insert.ext.i = and i64 %104, 4294967295
  %.sroa.0.4.insert.ext.i = zext i32 %103 to i64
  %.sroa.0.4.insert.shift.i = shl nuw i64 %.sroa.0.4.insert.ext.i, 32
  %.sroa.0.4.insert.insert.i = or i64 %.sroa.0.4.insert.shift.i, %.sroa.0.0.insert.ext.i
  br label %139

105:                                              ; preds = %59
  %106 = and i32 %99, -2147483648
  %.lobit80 = lshr i64 %90, 63
  %107 = icmp slt i32 %96, 2047
  br i1 %107, label %110, label %108

108:                                              ; preds = %105
  %109 = icmp eq i64 %.lobit80, 0
  %. = select i1 %109, i64 9218868437227405311, i64 9218868437227405312
  br label %135

110:                                              ; preds = %105
  %111 = trunc i64 %.lobit80 to i32
  %112 = icmp sgt i32 %96, -54
  br i1 %112, label %115, label %113

113:                                              ; preds = %110
  %114 = call i64 @llvm.nvvm.cvt.i64.i32(i32 4, i32 %111)
  br label %135

115:                                              ; preds = %110
  %116 = call double @llvm.nvvm.fma.f64(i32 3, double %88, double %80, double %86)
  %117 = fcmp one double %89, %116
  %118 = call double @llvm.nvvm.fma.f64(i32 4, double %88, double %80, double %86)
  %119 = bitcast double %118 to i64
  %120 = and i64 %119, 4503599627370495
  %121 = or i64 %120, 4503599627370496
  %122 = add nsw i32 %96, 63
  %123 = zext i32 %122 to i64
  %124 = shl i64 %121, %123
  %125 = icmp ne i64 %124, 0
  %126 = or i1 %117, %125
  %127 = sub nsw i32 1, %96
  %128 = zext i32 %127 to i64
  %129 = lshr i64 %121, %128
  %130 = zext i1 %126 to i32
  %131 = and i32 %111, %130
  %132 = icmp eq i32 %131, 1
  %133 = add i64 %129, 1
  %134 = select i1 %132, i64 %133, i64 %129
  br label %135

135:                                              ; preds = %108, %113, %115
  %.0 = phi i64 [ %114, %113 ], [ %., %108 ], [ %134, %115 ]
  %136 = call i64 @llvm.nvvm.cvt.i64.i32(i32 4, i32 %106)
  %137 = shl i64 %136, 32
  %138 = or i64 %.0, %137
  br label %139

139:                                              ; preds = %32, %100, %135, %28, %42, %37, %26, %21
  %.sroa.075.0 = phi i64 [ %.sroa.0.4.insert.shift.i29, %42 ], [ %.sroa.0.4.insert.shift.i33, %37 ], [ %.sroa.0.4.insert.insert.i42, %26 ], [ %.sroa.0.4.insert.insert.i46, %21 ], [ -2251799813685248, %28 ], [ %138, %135 ], [ %.sroa.0.4.insert.insert.i, %100 ], [ -2251799813685248, %32 ]
  %140 = bitcast i64 %.sroa.075.0 to double
  ret double %140
}

; Function Attrs: noinline
define weak dso_local float @__cuda_sm20_rcp_rn_f32_slowpath(float %0) #0 {
  %2 = bitcast float %0 to i32
  %3 = lshr i32 %2, 23
  %4 = and i32 %3, 255
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %6, label %19

6:                                                ; preds = %1
  %.mask = and i32 %2, 2147483647
  %7 = icmp eq i32 %.mask, 0
  br i1 %7, label %8, label %11

8:                                                ; preds = %6
  %9 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %0)
  %10 = bitcast float %9 to i32
  br label %63

11:                                               ; preds = %6
  %12 = call float @llvm.nvvm.fma.f32(i32 1, float %0, float 0x43F0000000000000, float 0.000000e+00)
  %13 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %12)
  %14 = call float @llvm.nvvm.fma.f32(i32 1, float %12, float %13, float -1.000000e+00)
  %15 = call float @llvm.nvvm.sub.ftz.f32(i32 1, float 0.000000e+00, float %14)
  %16 = call float @llvm.nvvm.fma.f32(i32 1, float %13, float %15, float %13)
  %17 = call float @llvm.nvvm.fma.f32(i32 1, float %16, float 0x43F0000000000000, float 0.000000e+00)
  %18 = bitcast float %17 to i32
  br label %63

19:                                               ; preds = %1
  %20 = add nsw i32 %4, -253
  %21 = icmp ugt i32 %20, 1
  br i1 %21, label %60, label %22

22:                                               ; preds = %19
  %23 = and i32 %2, 8388607
  %24 = or i32 %23, 1065353216
  %25 = bitcast i32 %24 to float
  %26 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %25)
  %27 = bitcast i32 %24 to float
  %28 = call float @llvm.nvvm.fma.f32(i32 1, float %27, float %26, float -1.000000e+00)
  %29 = call float @llvm.nvvm.sub.ftz.f32(i32 1, float 0.000000e+00, float %28)
  %30 = call float @llvm.nvvm.fma.f32(i32 2, float %26, float %29, float %26)
  %31 = bitcast float %30 to i32
  %32 = and i32 %31, 8388607
  %33 = or i32 %32, 8388608
  %34 = call float @llvm.nvvm.fma.f32(i32 3, float %26, float %29, float %26)
  %35 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %30, float 0.000000e+00)
  %36 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %34, float 0.000000e+00)
  %37 = fcmp une float %35, %36
  %38 = zext i1 %37 to i32
  %39 = and i32 %20, %33
  %40 = or i32 %39, %38
  %41 = shl i32 3, %20
  %42 = and i32 %41, %33
  %43 = lshr i32 %42, %20
  %44 = add nsw i32 %4, -252
  %45 = lshr i32 %33, %44
  %46 = icmp ne i32 %40, 0
  %47 = zext i1 %46 to i32
  %48 = lshr i32 %43, 1
  %49 = and i32 %48, 1
  %50 = or i32 %49, %47
  %51 = and i32 %43, %50
  %52 = icmp eq i32 %51, 1
  %53 = add i32 %45, 1
  %54 = select i1 %52, i32 %53, i32 %45
  %55 = icmp eq i32 %23, 0
  %56 = shl i32 %54, 1
  %57 = select i1 %55, i32 %56, i32 %54
  %58 = and i32 %2, -2147483648
  %59 = or i32 %57, %58
  br label %63

60:                                               ; preds = %19
  %61 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %0)
  %62 = bitcast float %61 to i32
  br label %63

63:                                               ; preds = %60, %22, %11, %8
  %.sroa.02.0 = phi i32 [ %62, %60 ], [ %59, %22 ], [ %18, %11 ], [ %10, %8 ]
  %64 = bitcast i32 %.sroa.02.0 to float
  ret float %64
}

; Function Attrs: noinline
define weak dso_local float @__cuda_sm20_rcp_rd_f32_slowpath(float %0) #0 {
  %2 = bitcast float %0 to i32
  %3 = lshr i32 %2, 23
  %4 = and i32 %3, 255
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %6, label %27

6:                                                ; preds = %1
  %7 = shl i32 %2, 1
  %8 = icmp sgt i32 %7, 4194304
  br i1 %8, label %19, label %9

9:                                                ; preds = %6
  %10 = and i32 %2, -2147483648
  %11 = icmp ne i32 %7, 0
  %12 = zext i1 %11 to i32
  %13 = lshr i32 %2, 31
  %14 = xor i32 %13, 1
  %15 = and i32 %14, %12
  %16 = icmp eq i32 %15, 0
  br i1 %16, label %17, label %70

17:                                               ; preds = %9
  %18 = or i32 %10, 2139095040
  br label %70

19:                                               ; preds = %6
  %20 = call float @llvm.nvvm.fma.f32(i32 1, float %0, float 0x43F0000000000000, float 0.000000e+00)
  %21 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %20)
  %22 = call float @llvm.nvvm.fma.f32(i32 1, float %20, float %21, float -1.000000e+00)
  %23 = call float @llvm.nvvm.sub.ftz.f32(i32 1, float 0.000000e+00, float %22)
  %24 = call float @llvm.nvvm.fma.f32(i32 2, float %21, float %23, float %21)
  %25 = call float @llvm.nvvm.fma.f32(i32 1, float %24, float 0x43F0000000000000, float 0.000000e+00)
  %26 = bitcast float %25 to i32
  br label %70

27:                                               ; preds = %1
  %28 = add nsw i32 %4, -253
  %29 = icmp ugt i32 %28, 1
  br i1 %29, label %67, label %30

30:                                               ; preds = %27
  %31 = and i32 %2, 8388607
  %32 = or i32 %31, 1065353216
  %33 = bitcast i32 %32 to float
  %34 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %33)
  %35 = bitcast i32 %32 to float
  %36 = call float @llvm.nvvm.fma.f32(i32 1, float %35, float %34, float -1.000000e+00)
  %37 = call float @llvm.nvvm.sub.ftz.f32(i32 1, float 0.000000e+00, float %36)
  %38 = call float @llvm.nvvm.fma.f32(i32 2, float %34, float %37, float %34)
  %39 = bitcast float %38 to i32
  %40 = and i32 %39, 8388607
  %41 = or i32 %40, 8388608
  %42 = call float @llvm.nvvm.fma.f32(i32 3, float %34, float %37, float %34)
  %43 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %38, float 0.000000e+00)
  %44 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %42, float 0.000000e+00)
  %45 = fcmp une float %43, %44
  %46 = zext i1 %45 to i32
  %47 = and i32 %28, %41
  %48 = or i32 %47, %46
  %49 = shl i32 1, %28
  %50 = and i32 %41, %49
  %51 = lshr i32 %50, %28
  %52 = or i32 %48, %51
  %53 = add nsw i32 %4, -252
  %54 = lshr i32 %41, %53
  %55 = and i32 %2, -2147483648
  %56 = lshr i32 %2, 31
  %57 = icmp ne i32 %52, 0
  %58 = zext i1 %57 to i32
  %59 = and i32 %56, %58
  %60 = icmp eq i32 %59, 1
  %61 = add i32 %54, 1
  %62 = select i1 %60, i32 %61, i32 %54
  %63 = icmp eq i32 %31, 0
  %64 = shl i32 %62, 1
  %65 = select i1 %63, i32 %64, i32 %62
  %66 = or i32 %65, %55
  br label %70

67:                                               ; preds = %27
  %68 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %0)
  %69 = bitcast float %68 to i32
  br label %70

70:                                               ; preds = %9, %67, %30, %19, %17
  %.sroa.02.0 = phi i32 [ %69, %67 ], [ %66, %30 ], [ %26, %19 ], [ %18, %17 ], [ 2139095039, %9 ]
  %71 = bitcast i32 %.sroa.02.0 to float
  ret float %71
}

; Function Attrs: noinline
define weak dso_local float @__cuda_sm20_rcp_ru_f32_slowpath(float %0) #0 {
  %2 = bitcast float %0 to i32
  %3 = lshr i32 %2, 23
  %4 = and i32 %3, 255
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %6, label %28

6:                                                ; preds = %1
  %7 = shl i32 %2, 1
  %8 = icmp sgt i32 %7, 4194304
  br i1 %8, label %20, label %9

9:                                                ; preds = %6
  %10 = and i32 %2, -2147483648
  %11 = icmp ne i32 %7, 0
  %12 = zext i1 %11 to i32
  %13 = lshr i32 %2, 31
  %14 = and i32 %13, %12
  %15 = icmp eq i32 %14, 0
  br i1 %15, label %18, label %16

16:                                               ; preds = %9
  %17 = or i32 %10, 2139095039
  br label %72

18:                                               ; preds = %9
  %19 = or i32 %10, 2139095040
  br label %72

20:                                               ; preds = %6
  %21 = call float @llvm.nvvm.fma.f32(i32 1, float %0, float 0x43F0000000000000, float 0.000000e+00)
  %22 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %21)
  %23 = call float @llvm.nvvm.fma.f32(i32 1, float %21, float %22, float -1.000000e+00)
  %24 = call float @llvm.nvvm.sub.ftz.f32(i32 1, float 0.000000e+00, float %23)
  %25 = call float @llvm.nvvm.fma.f32(i32 3, float %22, float %24, float %22)
  %26 = call float @llvm.nvvm.fma.f32(i32 1, float %25, float 0x43F0000000000000, float 0.000000e+00)
  %27 = bitcast float %26 to i32
  br label %72

28:                                               ; preds = %1
  %29 = add nsw i32 %4, -253
  %30 = icmp ugt i32 %29, 1
  br i1 %30, label %69, label %31

31:                                               ; preds = %28
  %32 = and i32 %2, 8388607
  %33 = or i32 %32, 1065353216
  %34 = bitcast i32 %33 to float
  %35 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %34)
  %36 = bitcast i32 %33 to float
  %37 = call float @llvm.nvvm.fma.f32(i32 1, float %36, float %35, float -1.000000e+00)
  %38 = call float @llvm.nvvm.sub.ftz.f32(i32 1, float 0.000000e+00, float %37)
  %39 = call float @llvm.nvvm.fma.f32(i32 2, float %35, float %38, float %35)
  %40 = bitcast float %39 to i32
  %41 = and i32 %40, 8388607
  %42 = or i32 %41, 8388608
  %43 = call float @llvm.nvvm.fma.f32(i32 3, float %35, float %38, float %35)
  %44 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %39, float 0.000000e+00)
  %45 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %43, float 0.000000e+00)
  %46 = fcmp une float %44, %45
  %47 = zext i1 %46 to i32
  %48 = and i32 %29, %42
  %49 = or i32 %48, %47
  %50 = shl i32 1, %29
  %51 = and i32 %42, %50
  %52 = lshr i32 %51, %29
  %53 = or i32 %49, %52
  %54 = add nsw i32 %4, -252
  %55 = lshr i32 %42, %54
  %56 = and i32 %2, -2147483648
  %57 = lshr i32 %2, 31
  %58 = xor i32 %57, 1
  %59 = icmp ne i32 %53, 0
  %60 = zext i1 %59 to i32
  %61 = and i32 %58, %60
  %62 = icmp eq i32 %61, 1
  %63 = add i32 %55, 1
  %64 = select i1 %62, i32 %63, i32 %55
  %65 = icmp eq i32 %32, 0
  %66 = shl i32 %64, 1
  %67 = select i1 %65, i32 %66, i32 %64
  %68 = or i32 %67, %56
  br label %72

69:                                               ; preds = %28
  %70 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %0)
  %71 = bitcast float %70 to i32
  br label %72

72:                                               ; preds = %69, %31, %20, %18, %16
  %.sroa.02.0 = phi i32 [ %71, %69 ], [ %68, %31 ], [ %27, %20 ], [ %19, %18 ], [ %17, %16 ]
  %73 = bitcast i32 %.sroa.02.0 to float
  ret float %73
}

; Function Attrs: noinline
define weak dso_local float @__cuda_sm20_rcp_rz_f32_slowpath(float %0) #0 {
  %2 = bitcast float %0 to i32
  %3 = lshr i32 %2, 23
  %4 = and i32 %3, 255
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %6, label %25

6:                                                ; preds = %1
  %7 = shl i32 %2, 1
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %9, label %12

9:                                                ; preds = %6
  %10 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %0)
  %11 = bitcast float %10 to i32
  br label %50

12:                                               ; preds = %6
  %13 = icmp sgt i32 %7, 4194304
  br i1 %13, label %17, label %14

14:                                               ; preds = %12
  %15 = and i32 %2, -2147483648
  %16 = or i32 %15, 2139095039
  br label %50

17:                                               ; preds = %12
  %18 = call float @llvm.nvvm.fma.f32(i32 1, float %0, float 0x43F0000000000000, float 0.000000e+00)
  %19 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %18)
  %20 = call float @llvm.nvvm.fma.f32(i32 1, float %18, float %19, float -1.000000e+00)
  %21 = call float @llvm.nvvm.sub.ftz.f32(i32 1, float 0.000000e+00, float %20)
  %22 = call float @llvm.nvvm.fma.f32(i32 4, float %19, float %21, float %19)
  %23 = call float @llvm.nvvm.fma.f32(i32 1, float %22, float 0x43F0000000000000, float 0.000000e+00)
  %24 = bitcast float %23 to i32
  br label %50

25:                                               ; preds = %1
  %26 = add nsw i32 %4, -253
  %27 = icmp ugt i32 %26, 1
  br i1 %27, label %47, label %28

28:                                               ; preds = %25
  %29 = and i32 %2, 8388607
  %30 = or i32 %29, 1065353216
  %31 = bitcast i32 %30 to float
  %32 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %31)
  %33 = bitcast i32 %30 to float
  %34 = call float @llvm.nvvm.fma.f32(i32 4, float %33, float %32, float -1.000000e+00)
  %35 = call float @llvm.nvvm.sub.ftz.f32(i32 1, float 0.000000e+00, float %34)
  %36 = call float @llvm.nvvm.fma.f32(i32 4, float %32, float %35, float %32)
  %37 = bitcast float %36 to i32
  %38 = and i32 %37, 8388607
  %39 = or i32 %38, 8388608
  %40 = add nsw i32 %4, -252
  %41 = lshr i32 %39, %40
  %42 = icmp eq i32 %29, 0
  %43 = shl i32 %41, 1
  %44 = select i1 %42, i32 %43, i32 %41
  %45 = and i32 %2, -2147483648
  %46 = or i32 %44, %45
  br label %50

47:                                               ; preds = %25
  %48 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %0)
  %49 = bitcast float %48 to i32
  br label %50

50:                                               ; preds = %47, %28, %17, %14, %9
  %.sroa.02.0 = phi i32 [ %49, %47 ], [ %46, %28 ], [ %24, %17 ], [ %16, %14 ], [ %11, %9 ]
  %51 = bitcast i32 %.sroa.02.0 to float
  ret float %51
}

; Function Attrs: noinline
define weak dso_local float @__cuda_sm20_rcp_rn_ftz_f32_slowpath(float %0) #0 {
  %2 = bitcast float %0 to i32
  %3 = add i32 %2, 25165824
  %4 = and i32 %3, 2139095040
  %5 = icmp eq i32 %4, 25165824
  br i1 %5, label %6, label %9

6:                                                ; preds = %1
  %7 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %0)
  %8 = bitcast float %7 to i32
  br label %21

9:                                                ; preds = %1
  %10 = or i32 %4, 8388608
  %11 = icmp eq i32 %10, 8388608
  br i1 %11, label %12, label %18

12:                                               ; preds = %9
  %13 = and i32 %2, 8388607
  %14 = or i32 %4, %13
  %15 = icmp eq i32 %14, 0
  %16 = and i32 %2, -2147483648
  %17 = or i32 %16, 8388608
  %spec.select = select i1 %15, i32 %17, i32 %16
  br label %21

18:                                               ; preds = %9
  %19 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %0)
  %20 = bitcast float %19 to i32
  br label %21

21:                                               ; preds = %12, %18, %6
  %.sroa.0.0 = phi i32 [ %20, %18 ], [ %8, %6 ], [ %spec.select, %12 ]
  %22 = bitcast i32 %.sroa.0.0 to float
  ret float %22
}

; Function Attrs: noinline
define weak dso_local float @__cuda_sm20_rcp_rd_ftz_f32_slowpath(float %0) #0 {
  %2 = bitcast float %0 to i32
  %3 = add i32 %2, 25165824
  %4 = and i32 %3, 2139095040
  %5 = icmp eq i32 %4, 25165824
  br i1 %5, label %6, label %9

6:                                                ; preds = %1
  %7 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %0)
  %8 = bitcast float %7 to i32
  br label %21

9:                                                ; preds = %1
  %10 = or i32 %4, 8388608
  %11 = icmp eq i32 %10, 8388608
  br i1 %11, label %12, label %18

12:                                               ; preds = %9
  %13 = and i32 %2, 8388607
  %14 = or i32 %4, %13
  %15 = icmp eq i32 %14, 0
  %16 = and i32 %2, -2147483648
  %17 = or i32 %16, 8388608
  %spec.select = select i1 %15, i32 %17, i32 %16
  br label %21

18:                                               ; preds = %9
  %19 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %0)
  %20 = bitcast float %19 to i32
  br label %21

21:                                               ; preds = %12, %18, %6
  %.sroa.0.0 = phi i32 [ %20, %18 ], [ %8, %6 ], [ %spec.select, %12 ]
  %22 = bitcast i32 %.sroa.0.0 to float
  ret float %22
}

; Function Attrs: noinline
define weak dso_local float @__cuda_sm20_rcp_ru_ftz_f32_slowpath(float %0) #0 {
  %2 = bitcast float %0 to i32
  %3 = add i32 %2, 25165824
  %4 = and i32 %3, 2139095040
  %5 = icmp eq i32 %4, 25165824
  br i1 %5, label %6, label %9

6:                                                ; preds = %1
  %7 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %0)
  %8 = bitcast float %7 to i32
  br label %21

9:                                                ; preds = %1
  %10 = or i32 %4, 8388608
  %11 = icmp eq i32 %10, 8388608
  br i1 %11, label %12, label %18

12:                                               ; preds = %9
  %13 = and i32 %2, 8388607
  %14 = or i32 %4, %13
  %15 = icmp eq i32 %14, 0
  %16 = and i32 %2, -2147483648
  %17 = or i32 %16, 8388608
  %spec.select = select i1 %15, i32 %17, i32 %16
  br label %21

18:                                               ; preds = %9
  %19 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %0)
  %20 = bitcast float %19 to i32
  br label %21

21:                                               ; preds = %12, %18, %6
  %.sroa.0.0 = phi i32 [ %20, %18 ], [ %8, %6 ], [ %spec.select, %12 ]
  %22 = bitcast i32 %.sroa.0.0 to float
  ret float %22
}

; Function Attrs: noinline
define weak dso_local float @__cuda_sm20_rcp_rz_ftz_f32_slowpath(float %0) #0 {
  %2 = bitcast float %0 to i32
  %3 = add i32 %2, 25165824
  %4 = and i32 %3, 2139095040
  %5 = icmp eq i32 %4, 25165824
  br i1 %5, label %6, label %9

6:                                                ; preds = %1
  %7 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %0)
  %8 = bitcast float %7 to i32
  br label %21

9:                                                ; preds = %1
  %10 = or i32 %4, 8388608
  %11 = icmp eq i32 %10, 8388608
  br i1 %11, label %12, label %18

12:                                               ; preds = %9
  %13 = and i32 %2, 8388607
  %14 = or i32 %4, %13
  %15 = icmp eq i32 %14, 0
  %16 = and i32 %2, -2147483648
  %17 = or i32 %16, 8388608
  %spec.select = select i1 %15, i32 %17, i32 %16
  br label %21

18:                                               ; preds = %9
  %19 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %0)
  %20 = bitcast float %19 to i32
  br label %21

21:                                               ; preds = %12, %18, %6
  %.sroa.0.0 = phi i32 [ %20, %18 ], [ %8, %6 ], [ %spec.select, %12 ]
  %22 = bitcast i32 %.sroa.0.0 to float
  ret float %22
}

; Function Attrs: noinline
define weak dso_local double @__cuda_sm20_dblrcp_rn_slowpath_v3(double %0, i32 %1) #0 {
  %3 = bitcast double %0 to i64
  %.sroa.0.4.extract.shift.i = lshr i64 %3, 32
  %.sroa.0.4.extract.trunc.i = trunc i64 %.sroa.0.4.extract.shift.i to i32
  %4 = and i32 %.sroa.0.4.extract.trunc.i, 2147483647
  %5 = bitcast double %0 to i64
  %6 = call double @llvm.nvvm.fabs.f64(double %0)
  %7 = fcmp ugt double %6, 0x7FF0000000000000
  br i1 %7, label %55, label %8

8:                                                ; preds = %2
  %9 = add nsw i32 %4, -1
  %10 = icmp ult i32 %9, 2146435071
  br i1 %10, label %13, label %11

11:                                               ; preds = %8
  %12 = and i64 %3, -4294967296
  %.sroa.0.4.insert.shift.i35 = xor i64 %12, 9218868437227405312
  br label %57

13:                                               ; preds = %8
  %14 = icmp ult i32 %4, 16777217
  br i1 %14, label %38, label %15

15:                                               ; preds = %13
  %.sroa.0.0.insert.ext.i29 = and i64 %5, 4294967295
  %16 = add i64 %3, -4602678819172646912
  %.sroa.0.4.insert.ext.i30 = and i64 %16, -4294967296
  %.sroa.0.4.insert.insert.i32 = or i64 %.sroa.0.4.insert.ext.i30, %.sroa.0.0.insert.ext.i29
  %17 = bitcast i64 %.sroa.0.4.insert.insert.i32 to double
  %18 = call double @llvm.nvvm.rcp.approx.ftz.f64(double %17)
  %19 = bitcast double %18 to i64
  %.sroa.0.0.extract.trunc.i26 = trunc i64 %19 to i32
  %20 = bitcast double %18 to i64
  %.sroa.0.4.extract.shift.i24 = and i64 %20, -4294967296
  %21 = or i32 %.sroa.0.0.extract.trunc.i26, %1
  %.sroa.0.0.insert.ext.i19 = zext i32 %21 to i64
  %.sroa.0.4.insert.insert.i22 = or i64 %.sroa.0.4.extract.shift.i24, %.sroa.0.0.insert.ext.i19
  %22 = bitcast i64 %.sroa.0.4.insert.insert.i32 to double
  %23 = fsub double -0.000000e+00, %22
  %24 = bitcast i64 %.sroa.0.4.insert.insert.i22 to double
  %25 = call double @llvm.nvvm.fma.f64(i32 1, double %23, double %24, double 1.000000e+00)
  %26 = call double @llvm.nvvm.fma.f64(i32 1, double %25, double %25, double %25)
  %27 = bitcast i64 %.sroa.0.4.insert.insert.i22 to double
  %28 = bitcast i64 %.sroa.0.4.insert.insert.i22 to double
  %29 = call double @llvm.nvvm.fma.f64(i32 1, double %26, double %27, double %28)
  %30 = call double @llvm.nvvm.fma.f64(i32 1, double %23, double %29, double 1.000000e+00)
  %31 = call double @llvm.nvvm.fma.f64(i32 1, double %30, double %29, double %29)
  %32 = call double @llvm.nvvm.mul.f64(i32 1, double %31, double 0x10000000000000)
  %33 = fsub double -0.000000e+00, %0
  %34 = call double @llvm.nvvm.fma.f64(i32 1, double %33, double %32, double 1.000000e+00)
  %35 = call double @llvm.nvvm.fma.f64(i32 1, double %34, double %34, double %34)
  %36 = call double @llvm.nvvm.fma.f64(i32 1, double %35, double %32, double %32)
  %37 = bitcast double %36 to i64
  br label %57

38:                                               ; preds = %13
  %39 = call double @llvm.nvvm.mul.f64(i32 1, double 0x4690000000000000, double %0)
  %40 = call double @llvm.nvvm.rcp.approx.ftz.f64(double %39)
  %41 = bitcast double %40 to i64
  %.sroa.0.0.extract.trunc.i8 = trunc i64 %41 to i32
  %42 = bitcast double %40 to i64
  %.sroa.0.4.extract.shift.i6 = and i64 %42, -4294967296
  %43 = or i32 %.sroa.0.0.extract.trunc.i8, %1
  %.sroa.0.0.insert.ext.i1 = zext i32 %43 to i64
  %.sroa.0.4.insert.insert.i4 = or i64 %.sroa.0.4.extract.shift.i6, %.sroa.0.0.insert.ext.i1
  %44 = fsub double -0.000000e+00, %39
  %45 = bitcast i64 %.sroa.0.4.insert.insert.i4 to double
  %46 = call double @llvm.nvvm.fma.f64(i32 1, double %44, double %45, double 1.000000e+00)
  %47 = call double @llvm.nvvm.fma.f64(i32 1, double %46, double %46, double %46)
  %48 = bitcast i64 %.sroa.0.4.insert.insert.i4 to double
  %49 = bitcast i64 %.sroa.0.4.insert.insert.i4 to double
  %50 = call double @llvm.nvvm.fma.f64(i32 1, double %47, double %48, double %49)
  %51 = call double @llvm.nvvm.fma.f64(i32 1, double %44, double %50, double 1.000000e+00)
  %52 = call double @llvm.nvvm.fma.f64(i32 1, double %51, double %50, double %50)
  %53 = call double @llvm.nvvm.mul.f64(i32 1, double %52, double 0x4690000000000000)
  %54 = bitcast double %53 to i64
  br label %57

55:                                               ; preds = %2
  %.sroa.0.0.insert.ext.i = and i64 %5, 4294967295
  %56 = and i64 %3, -2251804108652544
  %.sroa.0.4.insert.shift.i = or i64 %56, %.sroa.0.0.insert.ext.i
  %.sroa.0.4.insert.insert.i = or i64 %.sroa.0.4.insert.shift.i, 2251799813685248
  br label %57

57:                                               ; preds = %55, %38, %15, %11
  %.sroa.0.0 = phi i64 [ %.sroa.0.4.insert.insert.i, %55 ], [ %54, %38 ], [ %37, %15 ], [ %.sroa.0.4.insert.shift.i35, %11 ]
  %58 = bitcast i64 %.sroa.0.0 to double
  ret double %58
}

; Function Attrs: noinline
define weak dso_local double @__cuda_sm20_rcp_rd_f64(double %0) #0 {
  %2 = bitcast double %0 to i64
  %3 = bitcast double %0 to i64
  %.sroa.0.4.extract.shift.i38 = lshr i64 %3, 32
  %.sroa.0.4.extract.trunc.i39 = trunc i64 %.sroa.0.4.extract.shift.i38 to i32
  %4 = lshr i64 %3, 52
  %5 = trunc i64 %4 to i32
  %6 = and i32 %5, 2047
  %7 = add nsw i32 %6, -1
  %8 = icmp ult i32 %7, 2046
  br i1 %8, label %27, label %9

9:                                                ; preds = %1
  %10 = call double @llvm.nvvm.fabs.f64(double %0)
  %11 = fcmp ugt double %10, 0x7FF0000000000000
  br i1 %11, label %12, label %14

12:                                               ; preds = %9
  %.sroa.0.0.insert.ext.i33 = and i64 %2, 4294967295
  %13 = and i64 %3, -2251804108652544
  %.sroa.0.4.insert.shift.i35 = or i64 %13, %.sroa.0.0.insert.ext.i33
  %.sroa.0.4.insert.insert.i36 = or i64 %.sroa.0.4.insert.shift.i35, 2251799813685248
  br label %99

14:                                               ; preds = %9
  %15 = fcmp oeq double %10, 0x7FF0000000000000
  br i1 %15, label %16, label %17

16:                                               ; preds = %14
  %.sroa.0.4.insert.shift.i31 = and i64 %3, -9223372036854775808
  br label %99

17:                                               ; preds = %14
  %18 = fcmp oeq double %0, 0.000000e+00
  br i1 %18, label %19, label %21

19:                                               ; preds = %17
  %20 = and i64 %3, -9223372036854775808
  %.sroa.0.4.insert.shift.i27 = or i64 %20, 9218868437227405312
  br label %99

21:                                               ; preds = %17
  %22 = icmp eq i32 %6, 0
  br i1 %22, label %23, label %27

23:                                               ; preds = %21
  %24 = call double @llvm.nvvm.mul.f64(i32 1, double %0, double 0x4350000000000000)
  %25 = bitcast double %24 to i64
  %26 = bitcast double %24 to i64
  %.sroa.0.4.extract.shift.i20 = lshr i64 %26, 32
  %.sroa.0.4.extract.trunc.i21 = trunc i64 %.sroa.0.4.extract.shift.i20 to i32
  br label %27

27:                                               ; preds = %1, %21, %23
  %.041 = phi i64 [ %25, %23 ], [ %2, %21 ], [ %2, %1 ]
  %.040 = phi i32 [ %.sroa.0.4.extract.trunc.i21, %23 ], [ %.sroa.0.4.extract.trunc.i39, %21 ], [ %.sroa.0.4.extract.trunc.i39, %1 ]
  %.0 = phi i32 [ 54, %23 ], [ 0, %21 ], [ 0, %1 ]
  %28 = shl nuw nsw i32 %6, 20
  %29 = add nsw i32 %28, -1072693248
  %30 = sub i32 %.040, %29
  %.sroa.0.0.insert.ext.i15 = and i64 %.041, 4294967295
  %.sroa.0.4.insert.ext.i16 = zext i32 %30 to i64
  %.sroa.0.4.insert.shift.i17 = shl nuw i64 %.sroa.0.4.insert.ext.i16, 32
  %.sroa.0.4.insert.insert.i18 = or i64 %.sroa.0.4.insert.shift.i17, %.sroa.0.0.insert.ext.i15
  %31 = bitcast i64 %.sroa.0.4.insert.insert.i18 to double
  %32 = call double @llvm.nvvm.rcp.approx.ftz.f64(double %31)
  %33 = bitcast i64 %.sroa.0.4.insert.insert.i18 to double
  %34 = fsub double -0.000000e+00, %33
  %35 = bitcast double %32 to i64
  %36 = or i64 %35, 1
  %37 = bitcast i64 %36 to double
  %38 = call double @llvm.nvvm.fma.f64(i32 1, double %34, double %37, double 1.000000e+00)
  %39 = bitcast i64 %36 to double
  %40 = bitcast i64 %36 to double
  %41 = call double @llvm.nvvm.fma.f64(i32 1, double %38, double %39, double %40)
  %42 = call double @llvm.nvvm.mul.f64(i32 1, double %38, double %38)
  %43 = call double @llvm.nvvm.fma.f64(i32 1, double %42, double %41, double %41)
  %44 = call double @llvm.nvvm.fma.f64(i32 1, double %34, double %43, double 1.000000e+00)
  %45 = call double @llvm.nvvm.fma.f64(i32 1, double %44, double %43, double %43)
  %46 = bitcast i64 %36 to double
  %47 = call double @llvm.nvvm.fma.f64(i32 1, double %38, double %43, double %46)
  %48 = call double @llvm.nvvm.fma.f64(i32 1, double %34, double %47, double 1.000000e+00)
  %49 = call double @llvm.nvvm.fma.f64(i32 2, double %48, double %45, double %47)
  %50 = bitcast double %49 to i64
  %.sroa.0.4.extract.shift.i13 = lshr i64 %50, 32
  %51 = lshr i64 %50, 52
  %52 = trunc i64 %51 to i32
  %53 = and i32 %52, 2047
  %54 = sub nsw i32 %53, %6
  %55 = add nsw i32 %54, %.0
  %56 = add nsw i32 %55, 1022
  %57 = icmp ugt i32 %56, 2045
  %58 = trunc i64 %.sroa.0.4.extract.shift.i13 to i32
  br i1 %57, label %65, label %59

59:                                               ; preds = %27
  %60 = sub nsw i32 %55, %53
  %61 = shl i32 %60, 20
  %62 = add i32 %61, 1072693248
  %63 = add i32 %62, %58
  %64 = bitcast double %49 to i64
  %.sroa.0.0.insert.ext.i5 = and i64 %64, 4294967295
  %.sroa.0.4.insert.ext.i6 = zext i32 %63 to i64
  %.sroa.0.4.insert.shift.i7 = shl nuw i64 %.sroa.0.4.insert.ext.i6, 32
  %.sroa.0.4.insert.insert.i8 = or i64 %.sroa.0.4.insert.shift.i7, %.sroa.0.0.insert.ext.i5
  br label %99

65:                                               ; preds = %27
  %66 = and i32 %58, -2147483648
  %.lobit55 = lshr i64 %50, 63
  %67 = icmp slt i32 %55, 1024
  br i1 %67, label %72, label %68

68:                                               ; preds = %65
  %69 = icmp eq i64 %.lobit55, 0
  %. = select i1 %69, i64 9218868432932438016, i64 -4503599627370496
  %70 = icmp eq i32 %66, 0
  %71 = select i1 %70, i64 4294967295, i64 0
  %.sroa.0.4.insert.insert.i4 = or i64 %., %71
  br label %99

72:                                               ; preds = %65
  %73 = trunc i64 %.lobit55 to i32
  %74 = icmp sgt i32 %55, -1077
  br i1 %74, label %76, label %75

75:                                               ; preds = %72
  %.sroa.0.4.insert.ext.i = zext i32 %66 to i64
  %.sroa.0.4.insert.shift.i = shl nuw i64 %.sroa.0.4.insert.ext.i, 32
  %.sroa.0.4.insert.insert.i = or i64 %.sroa.0.4.insert.shift.i, %.lobit55
  br label %99

76:                                               ; preds = %72
  %77 = call double @llvm.nvvm.fma.f64(i32 3, double %48, double %45, double %47)
  %78 = fcmp one double %49, %77
  %79 = call double @llvm.nvvm.fma.f64(i32 4, double %48, double %45, double %47)
  %80 = bitcast double %79 to i64
  %81 = and i64 %80, 4503599627370495
  %82 = or i64 %81, 4503599627370496
  %83 = add nsw i32 %55, 1086
  %84 = zext i32 %83 to i64
  %85 = shl i64 %82, %84
  %86 = icmp ne i64 %85, 0
  %87 = or i1 %78, %86
  %88 = sub nsw i32 -1022, %55
  %89 = zext i32 %88 to i64
  %90 = lshr i64 %82, %89
  %91 = zext i1 %87 to i32
  %92 = and i32 %73, %91
  %93 = icmp eq i32 %92, 1
  %94 = add i64 %90, 1
  %95 = select i1 %93, i64 %94, i64 %90
  %96 = call i64 @llvm.nvvm.cvt.i64.i32(i32 4, i32 %66)
  %97 = shl i64 %96, 32
  %98 = or i64 %95, %97
  br label %99

99:                                               ; preds = %59, %68, %75, %76, %19, %16, %12
  %.sroa.051.0 = phi i64 [ %.sroa.0.4.insert.shift.i31, %16 ], [ %.sroa.0.4.insert.shift.i27, %19 ], [ %.sroa.0.4.insert.insert.i36, %12 ], [ %98, %76 ], [ %.sroa.0.4.insert.insert.i, %75 ], [ %.sroa.0.4.insert.insert.i4, %68 ], [ %.sroa.0.4.insert.insert.i8, %59 ]
  %100 = bitcast i64 %.sroa.051.0 to double
  ret double %100
}

; Function Attrs: noinline
define weak dso_local double @__cuda_sm20_rcp_ru_f64(double %0) #0 {
  %2 = bitcast double %0 to i64
  %3 = bitcast double %0 to i64
  %.sroa.0.4.extract.shift.i38 = lshr i64 %3, 32
  %.sroa.0.4.extract.trunc.i39 = trunc i64 %.sroa.0.4.extract.shift.i38 to i32
  %4 = lshr i64 %3, 52
  %5 = trunc i64 %4 to i32
  %6 = and i32 %5, 2047
  %7 = add nsw i32 %6, -1
  %8 = icmp ult i32 %7, 2046
  br i1 %8, label %27, label %9

9:                                                ; preds = %1
  %10 = call double @llvm.nvvm.fabs.f64(double %0)
  %11 = fcmp ugt double %10, 0x7FF0000000000000
  br i1 %11, label %12, label %14

12:                                               ; preds = %9
  %.sroa.0.0.insert.ext.i33 = and i64 %2, 4294967295
  %13 = and i64 %3, -2251804108652544
  %.sroa.0.4.insert.shift.i35 = or i64 %13, %.sroa.0.0.insert.ext.i33
  %.sroa.0.4.insert.insert.i36 = or i64 %.sroa.0.4.insert.shift.i35, 2251799813685248
  br label %99

14:                                               ; preds = %9
  %15 = fcmp oeq double %10, 0x7FF0000000000000
  br i1 %15, label %16, label %17

16:                                               ; preds = %14
  %.sroa.0.4.insert.shift.i31 = and i64 %3, -9223372036854775808
  br label %99

17:                                               ; preds = %14
  %18 = fcmp oeq double %0, 0.000000e+00
  br i1 %18, label %19, label %21

19:                                               ; preds = %17
  %20 = and i64 %3, -9223372036854775808
  %.sroa.0.4.insert.shift.i27 = or i64 %20, 9218868437227405312
  br label %99

21:                                               ; preds = %17
  %22 = icmp eq i32 %6, 0
  br i1 %22, label %23, label %27

23:                                               ; preds = %21
  %24 = call double @llvm.nvvm.mul.f64(i32 1, double %0, double 0x4350000000000000)
  %25 = bitcast double %24 to i64
  %26 = bitcast double %24 to i64
  %.sroa.0.4.extract.shift.i20 = lshr i64 %26, 32
  %.sroa.0.4.extract.trunc.i21 = trunc i64 %.sroa.0.4.extract.shift.i20 to i32
  br label %27

27:                                               ; preds = %1, %21, %23
  %.041 = phi i64 [ %25, %23 ], [ %2, %21 ], [ %2, %1 ]
  %.040 = phi i32 [ %.sroa.0.4.extract.trunc.i21, %23 ], [ %.sroa.0.4.extract.trunc.i39, %21 ], [ %.sroa.0.4.extract.trunc.i39, %1 ]
  %.0 = phi i32 [ 54, %23 ], [ 0, %21 ], [ 0, %1 ]
  %28 = shl nuw nsw i32 %6, 20
  %29 = add nsw i32 %28, -1072693248
  %30 = sub i32 %.040, %29
  %.sroa.0.0.insert.ext.i15 = and i64 %.041, 4294967295
  %.sroa.0.4.insert.ext.i16 = zext i32 %30 to i64
  %.sroa.0.4.insert.shift.i17 = shl nuw i64 %.sroa.0.4.insert.ext.i16, 32
  %.sroa.0.4.insert.insert.i18 = or i64 %.sroa.0.4.insert.shift.i17, %.sroa.0.0.insert.ext.i15
  %31 = bitcast i64 %.sroa.0.4.insert.insert.i18 to double
  %32 = call double @llvm.nvvm.rcp.approx.ftz.f64(double %31)
  %33 = bitcast i64 %.sroa.0.4.insert.insert.i18 to double
  %34 = fsub double -0.000000e+00, %33
  %35 = bitcast double %32 to i64
  %36 = or i64 %35, 1
  %37 = bitcast i64 %36 to double
  %38 = call double @llvm.nvvm.fma.f64(i32 1, double %34, double %37, double 1.000000e+00)
  %39 = bitcast i64 %36 to double
  %40 = bitcast i64 %36 to double
  %41 = call double @llvm.nvvm.fma.f64(i32 1, double %38, double %39, double %40)
  %42 = call double @llvm.nvvm.mul.f64(i32 1, double %38, double %38)
  %43 = call double @llvm.nvvm.fma.f64(i32 1, double %42, double %41, double %41)
  %44 = call double @llvm.nvvm.fma.f64(i32 1, double %34, double %43, double 1.000000e+00)
  %45 = call double @llvm.nvvm.fma.f64(i32 1, double %44, double %43, double %43)
  %46 = bitcast i64 %36 to double
  %47 = call double @llvm.nvvm.fma.f64(i32 1, double %38, double %43, double %46)
  %48 = call double @llvm.nvvm.fma.f64(i32 1, double %34, double %47, double 1.000000e+00)
  %49 = call double @llvm.nvvm.fma.f64(i32 3, double %48, double %45, double %47)
  %50 = bitcast double %49 to i64
  %.sroa.0.4.extract.shift.i13 = lshr i64 %50, 32
  %51 = lshr i64 %50, 52
  %52 = trunc i64 %51 to i32
  %53 = and i32 %52, 2047
  %54 = sub nsw i32 %53, %6
  %55 = add nsw i32 %54, %.0
  %56 = add nsw i32 %55, 1022
  %57 = icmp ugt i32 %56, 2045
  %58 = trunc i64 %.sroa.0.4.extract.shift.i13 to i32
  br i1 %57, label %65, label %59

59:                                               ; preds = %27
  %60 = sub nsw i32 %55, %53
  %61 = shl i32 %60, 20
  %62 = add i32 %61, 1072693248
  %63 = add i32 %62, %58
  %64 = bitcast double %49 to i64
  %.sroa.0.0.insert.ext.i5 = and i64 %64, 4294967295
  %.sroa.0.4.insert.ext.i6 = zext i32 %63 to i64
  %.sroa.0.4.insert.shift.i7 = shl nuw i64 %.sroa.0.4.insert.ext.i6, 32
  %.sroa.0.4.insert.insert.i8 = or i64 %.sroa.0.4.insert.shift.i7, %.sroa.0.0.insert.ext.i5
  br label %99

65:                                               ; preds = %27
  %66 = and i32 %58, -2147483648
  %67 = icmp slt i32 %55, 1024
  %.lobit56 = lshr i64 %50, 63
  br i1 %67, label %72, label %68

68:                                               ; preds = %65
  %69 = icmp eq i64 %.lobit56, 0
  %. = select i1 %69, i64 9218868437227405312, i64 -4503603922337792
  %70 = icmp eq i64 %.lobit56, 0
  %71 = select i1 %70, i64 0, i64 4294967295
  %.sroa.0.4.insert.insert.i4 = or i64 %., %71
  br label %99

72:                                               ; preds = %65
  %.lobit = trunc i64 %.lobit56 to i32
  %73 = xor i32 %.lobit, 1
  %74 = icmp sgt i32 %55, -1077
  br i1 %74, label %76, label %75

75:                                               ; preds = %72
  %.sroa.0.0.insert.ext.i = zext i32 %73 to i64
  %.sroa.0.4.insert.ext.i = zext i32 %66 to i64
  %.sroa.0.4.insert.shift.i = shl nuw i64 %.sroa.0.4.insert.ext.i, 32
  %.sroa.0.4.insert.insert.i = or i64 %.sroa.0.4.insert.shift.i, %.sroa.0.0.insert.ext.i
  br label %99

76:                                               ; preds = %72
  %77 = call double @llvm.nvvm.fma.f64(i32 2, double %48, double %45, double %47)
  %78 = fcmp one double %49, %77
  %79 = call double @llvm.nvvm.fma.f64(i32 4, double %48, double %45, double %47)
  %80 = bitcast double %79 to i64
  %81 = and i64 %80, 4503599627370495
  %82 = or i64 %81, 4503599627370496
  %83 = add nsw i32 %55, 1086
  %84 = zext i32 %83 to i64
  %85 = shl i64 %82, %84
  %86 = icmp ne i64 %85, 0
  %87 = or i1 %78, %86
  %88 = sub nsw i32 -1022, %55
  %89 = zext i32 %88 to i64
  %90 = lshr i64 %82, %89
  %91 = zext i1 %87 to i32
  %92 = and i32 %73, %91
  %93 = icmp eq i32 %92, 1
  %94 = add i64 %90, 1
  %95 = select i1 %93, i64 %94, i64 %90
  %96 = call i64 @llvm.nvvm.cvt.i64.i32(i32 4, i32 %66)
  %97 = shl i64 %96, 32
  %98 = or i64 %95, %97
  br label %99

99:                                               ; preds = %59, %68, %75, %76, %19, %16, %12
  %.sroa.051.0 = phi i64 [ %.sroa.0.4.insert.shift.i31, %16 ], [ %.sroa.0.4.insert.shift.i27, %19 ], [ %.sroa.0.4.insert.insert.i36, %12 ], [ %98, %76 ], [ %.sroa.0.4.insert.insert.i, %75 ], [ %.sroa.0.4.insert.insert.i4, %68 ], [ %.sroa.0.4.insert.insert.i8, %59 ]
  %100 = bitcast i64 %.sroa.051.0 to double
  ret double %100
}

; Function Attrs: noinline
define weak dso_local double @__cuda_sm20_rcp_rz_f64(double %0) #0 {
  %2 = bitcast double %0 to i64
  %3 = bitcast double %0 to i64
  %.sroa.0.4.extract.shift.i30 = lshr i64 %3, 32
  %.sroa.0.4.extract.trunc.i31 = trunc i64 %.sroa.0.4.extract.shift.i30 to i32
  %4 = lshr i64 %3, 52
  %5 = trunc i64 %4 to i32
  %6 = and i32 %5, 2047
  %7 = add nsw i32 %6, -1
  %8 = icmp ult i32 %7, 2046
  br i1 %8, label %27, label %9

9:                                                ; preds = %1
  %10 = call double @llvm.nvvm.fabs.f64(double %0)
  %11 = fcmp ugt double %10, 0x7FF0000000000000
  br i1 %11, label %12, label %14

12:                                               ; preds = %9
  %.sroa.0.0.insert.ext.i25 = and i64 %2, 4294967295
  %13 = and i64 %3, -2251804108652544
  %.sroa.0.4.insert.shift.i27 = or i64 %13, %.sroa.0.0.insert.ext.i25
  %.sroa.0.4.insert.insert.i28 = or i64 %.sroa.0.4.insert.shift.i27, 2251799813685248
  br label %81

14:                                               ; preds = %9
  %15 = fcmp oeq double %10, 0x7FF0000000000000
  br i1 %15, label %16, label %17

16:                                               ; preds = %14
  %.sroa.0.4.insert.shift.i23 = and i64 %3, -9223372036854775808
  br label %81

17:                                               ; preds = %14
  %18 = fcmp oeq double %0, 0.000000e+00
  br i1 %18, label %19, label %21

19:                                               ; preds = %17
  %20 = and i64 %3, -9223372036854775808
  %.sroa.0.4.insert.shift.i19 = or i64 %20, 9218868437227405312
  br label %81

21:                                               ; preds = %17
  %22 = icmp eq i32 %6, 0
  br i1 %22, label %23, label %27

23:                                               ; preds = %21
  %24 = call double @llvm.nvvm.mul.f64(i32 1, double %0, double 0x4350000000000000)
  %25 = bitcast double %24 to i64
  %26 = bitcast double %24 to i64
  %.sroa.0.4.extract.shift.i12 = lshr i64 %26, 32
  %.sroa.0.4.extract.trunc.i13 = trunc i64 %.sroa.0.4.extract.shift.i12 to i32
  br label %27

27:                                               ; preds = %1, %21, %23
  %.034 = phi i64 [ %25, %23 ], [ %2, %21 ], [ %2, %1 ]
  %.033 = phi i32 [ %.sroa.0.4.extract.trunc.i13, %23 ], [ %.sroa.0.4.extract.trunc.i31, %21 ], [ %.sroa.0.4.extract.trunc.i31, %1 ]
  %.032 = phi i32 [ 54, %23 ], [ 0, %21 ], [ 0, %1 ]
  %28 = shl nuw nsw i32 %6, 20
  %29 = add nsw i32 %28, -1072693248
  %30 = sub i32 %.033, %29
  %.sroa.0.0.insert.ext.i7 = and i64 %.034, 4294967295
  %.sroa.0.4.insert.ext.i8 = zext i32 %30 to i64
  %.sroa.0.4.insert.shift.i9 = shl nuw i64 %.sroa.0.4.insert.ext.i8, 32
  %.sroa.0.4.insert.insert.i10 = or i64 %.sroa.0.4.insert.shift.i9, %.sroa.0.0.insert.ext.i7
  %31 = bitcast i64 %.sroa.0.4.insert.insert.i10 to double
  %32 = call double @llvm.nvvm.rcp.approx.ftz.f64(double %31)
  %33 = bitcast i64 %.sroa.0.4.insert.insert.i10 to double
  %34 = fsub double -0.000000e+00, %33
  %35 = bitcast double %32 to i64
  %36 = or i64 %35, 1
  %37 = bitcast i64 %36 to double
  %38 = call double @llvm.nvvm.fma.f64(i32 1, double %34, double %37, double 1.000000e+00)
  %39 = bitcast i64 %36 to double
  %40 = bitcast i64 %36 to double
  %41 = call double @llvm.nvvm.fma.f64(i32 1, double %38, double %39, double %40)
  %42 = call double @llvm.nvvm.mul.f64(i32 1, double %38, double %38)
  %43 = call double @llvm.nvvm.fma.f64(i32 1, double %42, double %41, double %41)
  %44 = call double @llvm.nvvm.fma.f64(i32 1, double %34, double %43, double 1.000000e+00)
  %45 = call double @llvm.nvvm.fma.f64(i32 1, double %44, double %43, double %43)
  %46 = bitcast i64 %36 to double
  %47 = call double @llvm.nvvm.fma.f64(i32 1, double %38, double %43, double %46)
  %48 = call double @llvm.nvvm.fma.f64(i32 1, double %34, double %47, double 1.000000e+00)
  %49 = call double @llvm.nvvm.fma.f64(i32 4, double %48, double %45, double %47)
  %50 = bitcast double %49 to i64
  %.sroa.0.4.extract.shift.i5 = lshr i64 %50, 32
  %51 = lshr i64 %50, 52
  %52 = trunc i64 %51 to i32
  %53 = and i32 %52, 2047
  %54 = sub nsw i32 %53, %6
  %55 = add nsw i32 %54, %.032
  %56 = add nsw i32 %55, 1022
  %57 = icmp ugt i32 %56, 2045
  br i1 %57, label %64, label %58

58:                                               ; preds = %27
  %.sroa.0.4.extract.trunc.i6 = trunc i64 %.sroa.0.4.extract.shift.i5 to i32
  %59 = sub nsw i32 %55, %53
  %60 = shl i32 %59, 20
  %61 = add i32 %60, 1072693248
  %62 = add i32 %61, %.sroa.0.4.extract.trunc.i6
  %63 = bitcast double %49 to i64
  %.sroa.0.0.insert.ext.i = and i64 %63, 4294967295
  %.sroa.0.4.insert.ext.i = zext i32 %62 to i64
  %.sroa.0.4.insert.shift.i = shl nuw i64 %.sroa.0.4.insert.ext.i, 32
  %.sroa.0.4.insert.insert.i = or i64 %.sroa.0.4.insert.shift.i, %.sroa.0.0.insert.ext.i
  br label %81

64:                                               ; preds = %27
  %65 = icmp slt i32 %55, 1024
  br i1 %65, label %66, label %75

66:                                               ; preds = %64
  %67 = icmp sgt i32 %55, -1077
  br i1 %67, label %68, label %75

68:                                               ; preds = %66
  %69 = bitcast double %49 to i64
  %70 = and i64 %69, 4503599627370495
  %71 = or i64 %70, 4503599627370496
  %72 = sub nsw i32 -1022, %55
  %73 = zext i32 %72 to i64
  %74 = lshr i64 %71, %73
  br label %75

75:                                               ; preds = %68, %64, %66
  %.0 = phi i64 [ %74, %68 ], [ 9218868437227405311, %64 ], [ 0, %66 ]
  %76 = trunc i64 %.sroa.0.4.extract.shift.i5 to i32
  %77 = and i32 %76, -2147483648
  %78 = call i64 @llvm.nvvm.cvt.i64.i32(i32 4, i32 %77)
  %79 = shl i64 %78, 32
  %80 = or i64 %.0, %79
  br label %81

81:                                               ; preds = %58, %75, %19, %16, %12
  %.sroa.044.0 = phi i64 [ %.sroa.0.4.insert.shift.i23, %16 ], [ %.sroa.0.4.insert.shift.i19, %19 ], [ %.sroa.0.4.insert.insert.i28, %12 ], [ %80, %75 ], [ %.sroa.0.4.insert.insert.i, %58 ]
  %82 = bitcast i64 %.sroa.044.0 to double
  ret double %82
}

; Function Attrs: noinline
define weak dso_local float @__cuda_sm20_sqrt_rn_f32_slowpath(float %0) #0 {
  %2 = bitcast float %0 to i32
  %3 = and i32 %2, 2147483647
  %4 = icmp eq i32 %3, 0
  br i1 %4, label %29, label %5

5:                                                ; preds = %1
  %6 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %0, float 0.000000e+00)
  %7 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0.000000e+00, float 0.000000e+00)
  %8 = fcmp olt float %6, %7
  br i1 %8, label %29, label %9

9:                                                ; preds = %5
  %10 = call float @llvm.nvvm.fabs.ftz.f32(float %0)
  %11 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %10, float 0.000000e+00)
  %12 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %13 = fcmp ugt float %11, %12
  br i1 %13, label %14, label %16

14:                                               ; preds = %9
  %15 = call float @llvm.nvvm.add.ftz.f32(i32 1, float %0, float 1.000000e+00)
  br label %29

16:                                               ; preds = %9
  %17 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %10, float 0.000000e+00)
  %18 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %19 = fcmp oeq float %17, %18
  br i1 %19, label %29, label %20

20:                                               ; preds = %16
  %21 = call float @llvm.nvvm.fma.f32(i32 1, float %0, float 0x43F0000000000000, float 0.000000e+00)
  %22 = call float @llvm.nvvm.rsqrt.approx.ftz.f32(float %21)
  %23 = call float @llvm.nvvm.mul.ftz.f32(i32 1, float %22, float %21)
  %24 = call float @llvm.nvvm.sub.ftz.f32(i32 1, float 0.000000e+00, float %23)
  %25 = call float @llvm.nvvm.fma.f32(i32 1, float %24, float %23, float %21)
  %26 = call float @llvm.nvvm.mul.ftz.f32(i32 1, float %22, float 5.000000e-01)
  %27 = call float @llvm.nvvm.fma.f32(i32 1, float %25, float %26, float %23)
  %28 = call float @llvm.nvvm.mul.ftz.f32(i32 1, float %27, float 0x3DF0000000000000)
  br label %29

29:                                               ; preds = %16, %5, %1, %20, %14
  %.0 = phi float [ %28, %20 ], [ %15, %14 ], [ %0, %1 ], [ 0x7FFFFFFFE0000000, %5 ], [ %0, %16 ]
  ret float %.0
}

; Unknown intrinsic
declare dso_local float @llvm.nvvm.mul.ftz.f32(i32, float, float) #2

; Function Attrs: noinline
define weak dso_local float @__cuda_sm20_sqrt_rd_f32_slowpath(float %0) #0 {
  %2 = bitcast float %0 to i32
  %3 = and i32 %2, 2147483647
  %4 = icmp eq i32 %3, 0
  br i1 %4, label %32, label %5

5:                                                ; preds = %1
  %6 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %0, float 0.000000e+00)
  %7 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0.000000e+00, float 0.000000e+00)
  %8 = fcmp olt float %6, %7
  br i1 %8, label %32, label %9

9:                                                ; preds = %5
  %10 = call float @llvm.nvvm.fabs.ftz.f32(float %0)
  %11 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %10, float 0.000000e+00)
  %12 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %13 = fcmp ugt float %11, %12
  br i1 %13, label %14, label %16

14:                                               ; preds = %9
  %15 = call float @llvm.nvvm.add.ftz.f32(i32 1, float %0, float 1.000000e+00)
  br label %32

16:                                               ; preds = %9
  %17 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %10, float 0.000000e+00)
  %18 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %19 = fcmp oeq float %17, %18
  br i1 %19, label %32, label %20

20:                                               ; preds = %16
  %21 = call float @llvm.nvvm.fma.f32(i32 1, float %0, float 0x43F0000000000000, float 0.000000e+00)
  %22 = call float @llvm.nvvm.rsqrt.approx.ftz.f32(float %21)
  %23 = call float @llvm.nvvm.mul.ftz.f32(i32 1, float %22, float %21)
  %24 = call float @llvm.nvvm.mul.ftz.f32(i32 1, float %22, float 5.000000e-01)
  %25 = call float @llvm.nvvm.sub.ftz.f32(i32 1, float 0.000000e+00, float %23)
  %26 = call float @llvm.nvvm.fma.f32(i32 1, float %25, float %24, float 5.000000e-01)
  %27 = call float @llvm.nvvm.fma.f32(i32 1, float %23, float %26, float %23)
  %28 = call float @llvm.nvvm.sub.ftz.f32(i32 1, float 0.000000e+00, float %27)
  %29 = call float @llvm.nvvm.fma.f32(i32 1, float %28, float %27, float %21)
  %30 = call float @llvm.nvvm.fma.f32(i32 2, float %29, float %24, float %27)
  %31 = call float @llvm.nvvm.mul.ftz.f32(i32 1, float %30, float 0x3DF0000000000000)
  br label %32

32:                                               ; preds = %16, %5, %1, %20, %14
  %.0 = phi float [ %31, %20 ], [ %15, %14 ], [ %0, %1 ], [ 0x7FFFFFFFE0000000, %5 ], [ %0, %16 ]
  ret float %.0
}

; Function Attrs: noinline
define weak dso_local float @__cuda_sm20_sqrt_ru_f32_slowpath(float %0) #0 {
  %2 = bitcast float %0 to i32
  %3 = and i32 %2, 2147483647
  %4 = icmp eq i32 %3, 0
  br i1 %4, label %29, label %5

5:                                                ; preds = %1
  %6 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %0, float 0.000000e+00)
  %7 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0.000000e+00, float 0.000000e+00)
  %8 = fcmp olt float %6, %7
  br i1 %8, label %29, label %9

9:                                                ; preds = %5
  %10 = call float @llvm.nvvm.fabs.ftz.f32(float %0)
  %11 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %10, float 0.000000e+00)
  %12 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %13 = fcmp ugt float %11, %12
  br i1 %13, label %14, label %16

14:                                               ; preds = %9
  %15 = call float @llvm.nvvm.add.ftz.f32(i32 1, float %0, float 1.000000e+00)
  br label %29

16:                                               ; preds = %9
  %17 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %10, float 0.000000e+00)
  %18 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %19 = fcmp oeq float %17, %18
  br i1 %19, label %29, label %20

20:                                               ; preds = %16
  %21 = call float @llvm.nvvm.fma.f32(i32 1, float %0, float 0x43F0000000000000, float 0.000000e+00)
  %22 = call float @llvm.nvvm.rsqrt.approx.ftz.f32(float %21)
  %23 = call float @llvm.nvvm.mul.ftz.f32(i32 1, float %22, float %21)
  %24 = call float @llvm.nvvm.sub.ftz.f32(i32 1, float 0.000000e+00, float %23)
  %25 = call float @llvm.nvvm.fma.f32(i32 1, float %24, float %23, float %21)
  %26 = call float @llvm.nvvm.mul.ftz.f32(i32 1, float %22, float 5.000000e-01)
  %27 = call float @llvm.nvvm.fma.f32(i32 3, float %25, float %26, float %23)
  %28 = call float @llvm.nvvm.mul.ftz.f32(i32 1, float %27, float 0x3DF0000000000000)
  br label %29

29:                                               ; preds = %16, %5, %1, %20, %14
  %.0 = phi float [ %28, %20 ], [ %15, %14 ], [ %0, %1 ], [ 0x7FFFFFFFE0000000, %5 ], [ %0, %16 ]
  ret float %.0
}

; Function Attrs: noinline
define weak dso_local float @__cuda_sm20_sqrt_rz_f32_slowpath(float %0) #0 {
  %2 = bitcast float %0 to i32
  %3 = and i32 %2, 2147483647
  %4 = icmp eq i32 %3, 0
  br i1 %4, label %32, label %5

5:                                                ; preds = %1
  %6 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %0, float 0.000000e+00)
  %7 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0.000000e+00, float 0.000000e+00)
  %8 = fcmp olt float %6, %7
  br i1 %8, label %32, label %9

9:                                                ; preds = %5
  %10 = call float @llvm.nvvm.fabs.ftz.f32(float %0)
  %11 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %10, float 0.000000e+00)
  %12 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %13 = fcmp ugt float %11, %12
  br i1 %13, label %14, label %16

14:                                               ; preds = %9
  %15 = call float @llvm.nvvm.add.ftz.f32(i32 1, float %0, float 1.000000e+00)
  br label %32

16:                                               ; preds = %9
  %17 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %10, float 0.000000e+00)
  %18 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %19 = fcmp oeq float %17, %18
  br i1 %19, label %32, label %20

20:                                               ; preds = %16
  %21 = call float @llvm.nvvm.fma.f32(i32 1, float %0, float 0x43F0000000000000, float 0.000000e+00)
  %22 = call float @llvm.nvvm.rsqrt.approx.ftz.f32(float %21)
  %23 = call float @llvm.nvvm.mul.ftz.f32(i32 1, float %22, float %21)
  %24 = call float @llvm.nvvm.mul.ftz.f32(i32 1, float %22, float 5.000000e-01)
  %25 = call float @llvm.nvvm.sub.ftz.f32(i32 1, float 0.000000e+00, float %23)
  %26 = call float @llvm.nvvm.fma.f32(i32 1, float %25, float %24, float 5.000000e-01)
  %27 = call float @llvm.nvvm.fma.f32(i32 1, float %23, float %26, float %23)
  %28 = call float @llvm.nvvm.sub.ftz.f32(i32 1, float 0.000000e+00, float %27)
  %29 = call float @llvm.nvvm.fma.f32(i32 1, float %28, float %27, float %21)
  %30 = call float @llvm.nvvm.fma.f32(i32 4, float %29, float %24, float %27)
  %31 = call float @llvm.nvvm.mul.ftz.f32(i32 1, float %30, float 0x3DF0000000000000)
  br label %32

32:                                               ; preds = %16, %5, %1, %20, %14
  %.0 = phi float [ %31, %20 ], [ %15, %14 ], [ %0, %1 ], [ 0x7FFFFFFFE0000000, %5 ], [ %0, %16 ]
  ret float %.0
}

; Function Attrs: noinline
define weak dso_local float @__cuda_sm20_sqrt_rn_ftz_f32_slowpath(float %0) #0 {
  %2 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %0, float 0.000000e+00)
  %3 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0.000000e+00, float 0.000000e+00)
  %4 = fcmp oeq float %2, %3
  br i1 %4, label %5, label %7

5:                                                ; preds = %1
  %6 = call float @llvm.nvvm.mul.ftz.f32(i32 1, float %0, float 1.000000e+00)
  br label %31

7:                                                ; preds = %1
  %8 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %0, float 0.000000e+00)
  %9 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0.000000e+00, float 0.000000e+00)
  %10 = fcmp olt float %8, %9
  br i1 %10, label %31, label %11

11:                                               ; preds = %7
  %12 = call float @llvm.nvvm.fabs.ftz.f32(float %0)
  %13 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %12, float 0.000000e+00)
  %14 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %15 = fcmp ugt float %13, %14
  br i1 %15, label %16, label %18

16:                                               ; preds = %11
  %17 = call float @llvm.nvvm.add.ftz.f32(i32 1, float %0, float 1.000000e+00)
  br label %31

18:                                               ; preds = %11
  %19 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %12, float 0.000000e+00)
  %20 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %21 = fcmp oeq float %19, %20
  br i1 %21, label %31, label %22

22:                                               ; preds = %18
  %23 = call float @llvm.nvvm.mul.ftz.f32(i32 1, float %0, float 0x43F0000000000000)
  %24 = call float @llvm.nvvm.rsqrt.approx.ftz.f32(float %23)
  %25 = call float @llvm.nvvm.mul.ftz.f32(i32 1, float %24, float %23)
  %26 = call float @llvm.nvvm.sub.ftz.f32(i32 1, float 0.000000e+00, float %25)
  %27 = call float @llvm.nvvm.fma.f32(i32 1, float %26, float %25, float %23)
  %28 = call float @llvm.nvvm.mul.ftz.f32(i32 1, float %24, float 5.000000e-01)
  %29 = call float @llvm.nvvm.fma.f32(i32 1, float %27, float %28, float %25)
  %30 = call float @llvm.nvvm.mul.ftz.f32(i32 1, float %29, float 0x3DF0000000000000)
  br label %31

31:                                               ; preds = %18, %7, %22, %16, %5
  %.0 = phi float [ %6, %5 ], [ %30, %22 ], [ %17, %16 ], [ 0x7FFFFFFFE0000000, %7 ], [ %0, %18 ]
  ret float %.0
}

; Function Attrs: noinline
define weak dso_local float @__cuda_sm20_sqrt_rd_ftz_f32_slowpath(float %0) #0 {
  %2 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %0, float 0.000000e+00)
  %3 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0.000000e+00, float 0.000000e+00)
  %4 = fcmp oeq float %2, %3
  br i1 %4, label %5, label %7

5:                                                ; preds = %1
  %6 = call float @llvm.nvvm.mul.ftz.f32(i32 1, float %0, float 1.000000e+00)
  br label %34

7:                                                ; preds = %1
  %8 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %0, float 0.000000e+00)
  %9 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0.000000e+00, float 0.000000e+00)
  %10 = fcmp olt float %8, %9
  br i1 %10, label %34, label %11

11:                                               ; preds = %7
  %12 = call float @llvm.nvvm.fabs.ftz.f32(float %0)
  %13 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %12, float 0.000000e+00)
  %14 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %15 = fcmp ugt float %13, %14
  br i1 %15, label %16, label %18

16:                                               ; preds = %11
  %17 = call float @llvm.nvvm.add.ftz.f32(i32 1, float %0, float 1.000000e+00)
  br label %34

18:                                               ; preds = %11
  %19 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %12, float 0.000000e+00)
  %20 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %21 = fcmp oeq float %19, %20
  br i1 %21, label %34, label %22

22:                                               ; preds = %18
  %23 = call float @llvm.nvvm.mul.ftz.f32(i32 1, float %0, float 0x43F0000000000000)
  %24 = call float @llvm.nvvm.rsqrt.approx.ftz.f32(float %23)
  %25 = call float @llvm.nvvm.mul.ftz.f32(i32 1, float %24, float %23)
  %26 = call float @llvm.nvvm.mul.ftz.f32(i32 1, float %24, float 5.000000e-01)
  %27 = call float @llvm.nvvm.sub.ftz.f32(i32 1, float 0.000000e+00, float %25)
  %28 = call float @llvm.nvvm.fma.f32(i32 1, float %27, float %26, float 5.000000e-01)
  %29 = call float @llvm.nvvm.fma.f32(i32 1, float %25, float %28, float %25)
  %30 = call float @llvm.nvvm.sub.ftz.f32(i32 1, float 0.000000e+00, float %29)
  %31 = call float @llvm.nvvm.fma.f32(i32 1, float %30, float %29, float %23)
  %32 = call float @llvm.nvvm.fma.f32(i32 2, float %31, float %26, float %29)
  %33 = call float @llvm.nvvm.mul.ftz.f32(i32 1, float %32, float 0x3DF0000000000000)
  br label %34

34:                                               ; preds = %18, %7, %22, %16, %5
  %.0 = phi float [ %6, %5 ], [ %33, %22 ], [ %17, %16 ], [ 0x7FFFFFFFE0000000, %7 ], [ %0, %18 ]
  ret float %.0
}

; Function Attrs: noinline
define weak dso_local float @__cuda_sm20_sqrt_ru_ftz_f32_slowpath(float %0) #0 {
  %2 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %0, float 0.000000e+00)
  %3 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0.000000e+00, float 0.000000e+00)
  %4 = fcmp oeq float %2, %3
  br i1 %4, label %5, label %7

5:                                                ; preds = %1
  %6 = call float @llvm.nvvm.mul.ftz.f32(i32 1, float %0, float 1.000000e+00)
  br label %31

7:                                                ; preds = %1
  %8 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %0, float 0.000000e+00)
  %9 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0.000000e+00, float 0.000000e+00)
  %10 = fcmp olt float %8, %9
  br i1 %10, label %31, label %11

11:                                               ; preds = %7
  %12 = call float @llvm.nvvm.fabs.ftz.f32(float %0)
  %13 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %12, float 0.000000e+00)
  %14 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %15 = fcmp ugt float %13, %14
  br i1 %15, label %16, label %18

16:                                               ; preds = %11
  %17 = call float @llvm.nvvm.add.ftz.f32(i32 1, float %0, float 1.000000e+00)
  br label %31

18:                                               ; preds = %11
  %19 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %12, float 0.000000e+00)
  %20 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %21 = fcmp oeq float %19, %20
  br i1 %21, label %31, label %22

22:                                               ; preds = %18
  %23 = call float @llvm.nvvm.mul.ftz.f32(i32 1, float %0, float 0x43F0000000000000)
  %24 = call float @llvm.nvvm.rsqrt.approx.ftz.f32(float %23)
  %25 = call float @llvm.nvvm.mul.ftz.f32(i32 1, float %24, float %23)
  %26 = call float @llvm.nvvm.sub.ftz.f32(i32 1, float 0.000000e+00, float %25)
  %27 = call float @llvm.nvvm.fma.f32(i32 1, float %26, float %25, float %23)
  %28 = call float @llvm.nvvm.mul.ftz.f32(i32 1, float %24, float 5.000000e-01)
  %29 = call float @llvm.nvvm.fma.f32(i32 3, float %27, float %28, float %25)
  %30 = call float @llvm.nvvm.mul.ftz.f32(i32 1, float %29, float 0x3DF0000000000000)
  br label %31

31:                                               ; preds = %18, %7, %22, %16, %5
  %.0 = phi float [ %6, %5 ], [ %30, %22 ], [ %17, %16 ], [ 0x7FFFFFFFE0000000, %7 ], [ %0, %18 ]
  ret float %.0
}

; Function Attrs: noinline
define weak dso_local float @__cuda_sm20_sqrt_rz_ftz_f32_slowpath(float %0) #0 {
  %2 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %0, float 0.000000e+00)
  %3 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0.000000e+00, float 0.000000e+00)
  %4 = fcmp oeq float %2, %3
  br i1 %4, label %5, label %7

5:                                                ; preds = %1
  %6 = call float @llvm.nvvm.mul.ftz.f32(i32 1, float %0, float 1.000000e+00)
  br label %34

7:                                                ; preds = %1
  %8 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %0, float 0.000000e+00)
  %9 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0.000000e+00, float 0.000000e+00)
  %10 = fcmp olt float %8, %9
  br i1 %10, label %34, label %11

11:                                               ; preds = %7
  %12 = call float @llvm.nvvm.fabs.ftz.f32(float %0)
  %13 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %12, float 0.000000e+00)
  %14 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %15 = fcmp ugt float %13, %14
  br i1 %15, label %16, label %18

16:                                               ; preds = %11
  %17 = call float @llvm.nvvm.add.ftz.f32(i32 1, float %0, float 1.000000e+00)
  br label %34

18:                                               ; preds = %11
  %19 = call float @llvm.nvvm.add.ftz.f32(i32 2, float %12, float 0.000000e+00)
  %20 = call float @llvm.nvvm.add.ftz.f32(i32 2, float 0x7FF0000000000000, float 0.000000e+00)
  %21 = fcmp oeq float %19, %20
  br i1 %21, label %34, label %22

22:                                               ; preds = %18
  %23 = call float @llvm.nvvm.mul.ftz.f32(i32 1, float %0, float 0x43F0000000000000)
  %24 = call float @llvm.nvvm.rsqrt.approx.ftz.f32(float %23)
  %25 = call float @llvm.nvvm.mul.ftz.f32(i32 1, float %24, float %23)
  %26 = call float @llvm.nvvm.mul.ftz.f32(i32 1, float %24, float 5.000000e-01)
  %27 = call float @llvm.nvvm.sub.ftz.f32(i32 1, float 0.000000e+00, float %25)
  %28 = call float @llvm.nvvm.fma.f32(i32 1, float %27, float %26, float 5.000000e-01)
  %29 = call float @llvm.nvvm.fma.f32(i32 1, float %25, float %28, float %25)
  %30 = call float @llvm.nvvm.sub.ftz.f32(i32 1, float 0.000000e+00, float %29)
  %31 = call float @llvm.nvvm.fma.f32(i32 1, float %30, float %29, float %23)
  %32 = call float @llvm.nvvm.fma.f32(i32 4, float %31, float %26, float %29)
  %33 = call float @llvm.nvvm.mul.ftz.f32(i32 1, float %32, float 0x3DF0000000000000)
  br label %34

34:                                               ; preds = %18, %7, %22, %16, %5
  %.0 = phi float [ %6, %5 ], [ %33, %22 ], [ %17, %16 ], [ 0x7FFFFFFFE0000000, %7 ], [ %0, %18 ]
  ret float %.0
}

; Function Attrs: noinline
define weak dso_local double @__cuda_sm20_dsqrt_rz_f64(double %0) #0 {
  %2 = bitcast double %0 to i64
  %3 = bitcast double %0 to i64
  %4 = lshr i64 %2, 52
  %5 = trunc i64 %4 to i32
  %6 = and i32 %5, 2047
  %7 = add nsw i32 %6, -1
  %.lobit = ashr i64 %2, 63
  %8 = trunc i64 %.lobit to i32
  %9 = sub i32 0, %8
  %10 = icmp ugt i32 %7, 2045
  %11 = zext i1 %10 to i32
  %12 = or i32 %9, %11
  %13 = icmp eq i32 %12, 0
  br i1 %13, label %31, label %14

14:                                               ; preds = %1
  %15 = fcmp olt double %0, 0.000000e+00
  br i1 %15, label %72, label %16

16:                                               ; preds = %14
  %17 = fcmp oeq double %0, 0.000000e+00
  br i1 %17, label %21, label %18

18:                                               ; preds = %16
  %19 = call double @llvm.nvvm.fabs.f64(double %0)
  %20 = fcmp oeq double %19, 0x7FF0000000000000
  br i1 %20, label %21, label %23

21:                                               ; preds = %18, %16
  %22 = bitcast double %0 to i64
  br label %72

23:                                               ; preds = %18
  %24 = fcmp ugt double %19, 0x7FF0000000000000
  br i1 %24, label %25, label %27

25:                                               ; preds = %23
  %.sroa.0.0.insert.ext.i31 = and i64 %3, 4294967295
  %26 = and i64 %2, -2251804108652544
  %.sroa.0.4.insert.shift.i33 = or i64 %26, %.sroa.0.0.insert.ext.i31
  %.sroa.0.4.insert.insert.i34 = or i64 %.sroa.0.4.insert.shift.i33, 2251799813685248
  br label %72

27:                                               ; preds = %23
  %28 = call double @llvm.nvvm.mul.f64(i32 1, double %0, double 0x4350000000000000)
  %29 = bitcast double %28 to i64
  %30 = bitcast double %28 to i64
  br label %31

31:                                               ; preds = %1, %27
  %.043.in.in = phi i64 [ %29, %27 ], [ %2, %1 ]
  %.042.in = phi i64 [ %30, %27 ], [ %3, %1 ]
  %.0 = phi i32 [ 54, %27 ], [ 0, %1 ]
  %.043.in = lshr i64 %.043.in.in, 32
  %.043 = trunc i64 %.043.in to i32
  %32 = and i32 %7, -2
  %33 = shl nsw i32 %32, 20
  %34 = add i32 %33, -1071644672
  %35 = sub i32 %.043, %34
  %.sroa.0.0.insert.ext.i21 = and i64 %.042.in, 4294967295
  %.sroa.0.4.insert.ext.i22 = zext i32 %35 to i64
  %.sroa.0.4.insert.shift.i23 = shl nuw i64 %.sroa.0.4.insert.ext.i22, 32
  %.sroa.0.4.insert.insert.i24 = or i64 %.sroa.0.4.insert.shift.i23, %.sroa.0.0.insert.ext.i21
  %36 = bitcast i64 %.sroa.0.4.insert.insert.i24 to double
  %37 = call float @llvm.nvvm.cvt.f32.f64(i32 4, double %36)
  %38 = add i32 %35, -1048576
  %.sroa.0.0.insert.ext.i17 = and i64 %.042.in, 4294967295
  %.sroa.0.4.insert.ext.i18 = zext i32 %38 to i64
  %.sroa.0.4.insert.shift.i19 = shl nuw i64 %.sroa.0.4.insert.ext.i18, 32
  %.sroa.0.4.insert.insert.i20 = or i64 %.sroa.0.4.insert.shift.i19, %.sroa.0.0.insert.ext.i17
  %39 = call float @llvm.nvvm.rsqrt.approx.ftz.f32(float %37)
  %40 = call double @llvm.nvvm.cvt.f64.f32(i32 4, float %39)
  %41 = call double @llvm.nvvm.mul.f64(i32 1, double %40, double %40)
  %42 = fsub double -0.000000e+00, %41
  %43 = bitcast i64 %.sroa.0.4.insert.insert.i20 to double
  %44 = call double @llvm.nvvm.fma.f64(i32 1, double %42, double %43, double 5.000000e-01)
  %45 = call double @llvm.nvvm.fma.f64(i32 1, double %44, double %40, double %40)
  %46 = bitcast i64 %.sroa.0.4.insert.insert.i24 to double
  %47 = call double @llvm.nvvm.mul.f64(i32 1, double %45, double %46)
  %48 = bitcast i64 %.sroa.0.4.insert.insert.i20 to double
  %49 = call double @llvm.nvvm.mul.f64(i32 1, double %48, double %45)
  %50 = bitcast double %45 to i64
  %51 = bitcast double %45 to i64
  %52 = fsub double -0.000000e+00, %47
  %53 = bitcast i64 %.sroa.0.4.insert.insert.i24 to double
  %54 = call double @llvm.nvvm.fma.f64(i32 1, double %52, double %47, double %53)
  %.sroa.0.0.insert.ext.i7 = and i64 %50, 4294967295
  %.sroa.0.4.extract.shift.i1258 = add i64 %51, -4503599627370496
  %.sroa.0.4.insert.ext.i8 = and i64 %.sroa.0.4.extract.shift.i1258, -4294967296
  %.sroa.0.4.insert.insert.i10 = or i64 %.sroa.0.4.insert.ext.i8, %.sroa.0.0.insert.ext.i7
  %55 = bitcast i64 %.sroa.0.4.insert.insert.i10 to double
  %56 = call double @llvm.nvvm.fma.f64(i32 1, double %54, double %55, double %47)
  %57 = fsub double -0.000000e+00, %56
  %58 = bitcast i64 %.sroa.0.4.insert.insert.i24 to double
  %59 = call double @llvm.nvvm.fma.f64(i32 1, double %57, double %56, double %58)
  %60 = fsub double -0.000000e+00, %49
  %61 = call double @llvm.nvvm.fma.f64(i32 1, double %60, double %45, double 5.000000e-01)
  %62 = bitcast i64 %.sroa.0.4.insert.insert.i10 to double
  %63 = bitcast i64 %.sroa.0.4.insert.insert.i10 to double
  %64 = call double @llvm.nvvm.fma.f64(i32 1, double %61, double %62, double %63)
  %65 = call double @llvm.nvvm.fma.f64(i32 4, double %59, double %64, double %56)
  %66 = bitcast double %65 to i64
  %.sroa.0.4.extract.shift.i5 = lshr i64 %66, 32
  %.sroa.0.4.extract.trunc.i6 = trunc i64 %.sroa.0.4.extract.shift.i5 to i32
  %67 = sub nsw i32 %32, %.0
  %68 = shl nsw i32 %67, 19
  %69 = add i32 %68, -535822336
  %70 = add i32 %69, %.sroa.0.4.extract.trunc.i6
  %71 = bitcast double %65 to i64
  %.sroa.0.0.insert.ext.i = and i64 %71, 4294967295
  %.sroa.0.4.insert.ext.i = zext i32 %70 to i64
  %.sroa.0.4.insert.shift.i = shl nuw i64 %.sroa.0.4.insert.ext.i, 32
  %.sroa.0.4.insert.insert.i = or i64 %.sroa.0.4.insert.shift.i, %.sroa.0.0.insert.ext.i
  br label %72

72:                                               ; preds = %14, %31, %25, %21
  %.sroa.055.0 = phi i64 [ %.sroa.0.4.insert.insert.i, %31 ], [ %22, %21 ], [ %.sroa.0.4.insert.insert.i34, %25 ], [ -2251799813685248, %14 ]
  %73 = bitcast i64 %.sroa.055.0 to double
  ret double %73
}

; Unknown intrinsic
declare dso_local float @llvm.nvvm.cvt.f32.f64(i32, double) #2

; Unknown intrinsic
declare dso_local double @llvm.nvvm.cvt.f64.f32(i32, float) #2

; Function Attrs: noinline
define weak dso_local double @__cuda_sm20_dsqrt_ru_f64(double %0) #0 {
  %2 = bitcast double %0 to i64
  %3 = bitcast double %0 to i64
  %4 = lshr i64 %2, 52
  %5 = trunc i64 %4 to i32
  %6 = and i32 %5, 2047
  %7 = add nsw i32 %6, -1
  %.lobit = ashr i64 %2, 63
  %8 = trunc i64 %.lobit to i32
  %9 = sub i32 0, %8
  %10 = icmp ugt i32 %7, 2045
  %11 = zext i1 %10 to i32
  %12 = or i32 %9, %11
  %13 = icmp eq i32 %12, 0
  br i1 %13, label %31, label %14

14:                                               ; preds = %1
  %15 = fcmp olt double %0, 0.000000e+00
  br i1 %15, label %72, label %16

16:                                               ; preds = %14
  %17 = fcmp oeq double %0, 0.000000e+00
  br i1 %17, label %21, label %18

18:                                               ; preds = %16
  %19 = call double @llvm.nvvm.fabs.f64(double %0)
  %20 = fcmp oeq double %19, 0x7FF0000000000000
  br i1 %20, label %21, label %23

21:                                               ; preds = %18, %16
  %22 = bitcast double %0 to i64
  br label %72

23:                                               ; preds = %18
  %24 = fcmp ugt double %19, 0x7FF0000000000000
  br i1 %24, label %25, label %27

25:                                               ; preds = %23
  %.sroa.0.0.insert.ext.i31 = and i64 %3, 4294967295
  %26 = and i64 %2, -2251804108652544
  %.sroa.0.4.insert.shift.i33 = or i64 %26, %.sroa.0.0.insert.ext.i31
  %.sroa.0.4.insert.insert.i34 = or i64 %.sroa.0.4.insert.shift.i33, 2251799813685248
  br label %72

27:                                               ; preds = %23
  %28 = call double @llvm.nvvm.mul.f64(i32 1, double %0, double 0x4350000000000000)
  %29 = bitcast double %28 to i64
  %30 = bitcast double %28 to i64
  br label %31

31:                                               ; preds = %1, %27
  %.043.in.in = phi i64 [ %29, %27 ], [ %2, %1 ]
  %.042.in = phi i64 [ %30, %27 ], [ %3, %1 ]
  %.0 = phi i32 [ 54, %27 ], [ 0, %1 ]
  %.043.in = lshr i64 %.043.in.in, 32
  %.043 = trunc i64 %.043.in to i32
  %32 = and i32 %7, -2
  %33 = shl nsw i32 %32, 20
  %34 = add i32 %33, -1071644672
  %35 = sub i32 %.043, %34
  %.sroa.0.0.insert.ext.i21 = and i64 %.042.in, 4294967295
  %.sroa.0.4.insert.ext.i22 = zext i32 %35 to i64
  %.sroa.0.4.insert.shift.i23 = shl nuw i64 %.sroa.0.4.insert.ext.i22, 32
  %.sroa.0.4.insert.insert.i24 = or i64 %.sroa.0.4.insert.shift.i23, %.sroa.0.0.insert.ext.i21
  %36 = bitcast i64 %.sroa.0.4.insert.insert.i24 to double
  %37 = call float @llvm.nvvm.cvt.f32.f64(i32 4, double %36)
  %38 = add i32 %35, -1048576
  %.sroa.0.0.insert.ext.i17 = and i64 %.042.in, 4294967295
  %.sroa.0.4.insert.ext.i18 = zext i32 %38 to i64
  %.sroa.0.4.insert.shift.i19 = shl nuw i64 %.sroa.0.4.insert.ext.i18, 32
  %.sroa.0.4.insert.insert.i20 = or i64 %.sroa.0.4.insert.shift.i19, %.sroa.0.0.insert.ext.i17
  %39 = call float @llvm.nvvm.rsqrt.approx.ftz.f32(float %37)
  %40 = call double @llvm.nvvm.cvt.f64.f32(i32 4, float %39)
  %41 = call double @llvm.nvvm.mul.f64(i32 1, double %40, double %40)
  %42 = fsub double -0.000000e+00, %41
  %43 = bitcast i64 %.sroa.0.4.insert.insert.i20 to double
  %44 = call double @llvm.nvvm.fma.f64(i32 1, double %42, double %43, double 5.000000e-01)
  %45 = call double @llvm.nvvm.fma.f64(i32 1, double %44, double %40, double %40)
  %46 = bitcast i64 %.sroa.0.4.insert.insert.i24 to double
  %47 = call double @llvm.nvvm.mul.f64(i32 1, double %45, double %46)
  %48 = bitcast i64 %.sroa.0.4.insert.insert.i20 to double
  %49 = call double @llvm.nvvm.mul.f64(i32 1, double %48, double %45)
  %50 = bitcast double %45 to i64
  %51 = bitcast double %45 to i64
  %52 = fsub double -0.000000e+00, %47
  %53 = bitcast i64 %.sroa.0.4.insert.insert.i24 to double
  %54 = call double @llvm.nvvm.fma.f64(i32 1, double %52, double %47, double %53)
  %.sroa.0.0.insert.ext.i7 = and i64 %50, 4294967295
  %.sroa.0.4.extract.shift.i1258 = add i64 %51, -4503599627370496
  %.sroa.0.4.insert.ext.i8 = and i64 %.sroa.0.4.extract.shift.i1258, -4294967296
  %.sroa.0.4.insert.insert.i10 = or i64 %.sroa.0.4.insert.ext.i8, %.sroa.0.0.insert.ext.i7
  %55 = bitcast i64 %.sroa.0.4.insert.insert.i10 to double
  %56 = call double @llvm.nvvm.fma.f64(i32 1, double %54, double %55, double %47)
  %57 = fsub double -0.000000e+00, %56
  %58 = bitcast i64 %.sroa.0.4.insert.insert.i24 to double
  %59 = call double @llvm.nvvm.fma.f64(i32 1, double %57, double %56, double %58)
  %60 = fsub double -0.000000e+00, %49
  %61 = call double @llvm.nvvm.fma.f64(i32 1, double %60, double %45, double 5.000000e-01)
  %62 = bitcast i64 %.sroa.0.4.insert.insert.i10 to double
  %63 = bitcast i64 %.sroa.0.4.insert.insert.i10 to double
  %64 = call double @llvm.nvvm.fma.f64(i32 1, double %61, double %62, double %63)
  %65 = call double @llvm.nvvm.fma.f64(i32 3, double %59, double %64, double %56)
  %66 = bitcast double %65 to i64
  %.sroa.0.4.extract.shift.i5 = lshr i64 %66, 32
  %.sroa.0.4.extract.trunc.i6 = trunc i64 %.sroa.0.4.extract.shift.i5 to i32
  %67 = sub nsw i32 %32, %.0
  %68 = shl nsw i32 %67, 19
  %69 = add i32 %68, -535822336
  %70 = add i32 %69, %.sroa.0.4.extract.trunc.i6
  %71 = bitcast double %65 to i64
  %.sroa.0.0.insert.ext.i = and i64 %71, 4294967295
  %.sroa.0.4.insert.ext.i = zext i32 %70 to i64
  %.sroa.0.4.insert.shift.i = shl nuw i64 %.sroa.0.4.insert.ext.i, 32
  %.sroa.0.4.insert.insert.i = or i64 %.sroa.0.4.insert.shift.i, %.sroa.0.0.insert.ext.i
  br label %72

72:                                               ; preds = %14, %31, %25, %21
  %.sroa.055.0 = phi i64 [ %.sroa.0.4.insert.insert.i, %31 ], [ %22, %21 ], [ %.sroa.0.4.insert.insert.i34, %25 ], [ -2251799813685248, %14 ]
  %73 = bitcast i64 %.sroa.055.0 to double
  ret double %73
}

; Function Attrs: noinline
define weak dso_local double @__cuda_sm20_dsqrt_rd_f64(double %0) #0 {
  %2 = bitcast double %0 to i64
  %3 = bitcast double %0 to i64
  %4 = lshr i64 %2, 52
  %5 = trunc i64 %4 to i32
  %6 = and i32 %5, 2047
  %7 = add nsw i32 %6, -1
  %.lobit = ashr i64 %2, 63
  %8 = trunc i64 %.lobit to i32
  %9 = sub i32 0, %8
  %10 = icmp ugt i32 %7, 2045
  %11 = zext i1 %10 to i32
  %12 = or i32 %9, %11
  %13 = icmp eq i32 %12, 0
  br i1 %13, label %31, label %14

14:                                               ; preds = %1
  %15 = fcmp olt double %0, 0.000000e+00
  br i1 %15, label %72, label %16

16:                                               ; preds = %14
  %17 = fcmp oeq double %0, 0.000000e+00
  br i1 %17, label %21, label %18

18:                                               ; preds = %16
  %19 = call double @llvm.nvvm.fabs.f64(double %0)
  %20 = fcmp oeq double %19, 0x7FF0000000000000
  br i1 %20, label %21, label %23

21:                                               ; preds = %18, %16
  %22 = bitcast double %0 to i64
  br label %72

23:                                               ; preds = %18
  %24 = fcmp ugt double %19, 0x7FF0000000000000
  br i1 %24, label %25, label %27

25:                                               ; preds = %23
  %.sroa.0.0.insert.ext.i31 = and i64 %3, 4294967295
  %26 = and i64 %2, -2251804108652544
  %.sroa.0.4.insert.shift.i33 = or i64 %26, %.sroa.0.0.insert.ext.i31
  %.sroa.0.4.insert.insert.i34 = or i64 %.sroa.0.4.insert.shift.i33, 2251799813685248
  br label %72

27:                                               ; preds = %23
  %28 = call double @llvm.nvvm.mul.f64(i32 1, double %0, double 0x4350000000000000)
  %29 = bitcast double %28 to i64
  %30 = bitcast double %28 to i64
  br label %31

31:                                               ; preds = %1, %27
  %.043.in.in = phi i64 [ %29, %27 ], [ %2, %1 ]
  %.042.in = phi i64 [ %30, %27 ], [ %3, %1 ]
  %.0 = phi i32 [ 54, %27 ], [ 0, %1 ]
  %.043.in = lshr i64 %.043.in.in, 32
  %.043 = trunc i64 %.043.in to i32
  %32 = and i32 %7, -2
  %33 = shl nsw i32 %32, 20
  %34 = add i32 %33, -1071644672
  %35 = sub i32 %.043, %34
  %.sroa.0.0.insert.ext.i21 = and i64 %.042.in, 4294967295
  %.sroa.0.4.insert.ext.i22 = zext i32 %35 to i64
  %.sroa.0.4.insert.shift.i23 = shl nuw i64 %.sroa.0.4.insert.ext.i22, 32
  %.sroa.0.4.insert.insert.i24 = or i64 %.sroa.0.4.insert.shift.i23, %.sroa.0.0.insert.ext.i21
  %36 = bitcast i64 %.sroa.0.4.insert.insert.i24 to double
  %37 = call float @llvm.nvvm.cvt.f32.f64(i32 4, double %36)
  %38 = add i32 %35, -1048576
  %.sroa.0.0.insert.ext.i17 = and i64 %.042.in, 4294967295
  %.sroa.0.4.insert.ext.i18 = zext i32 %38 to i64
  %.sroa.0.4.insert.shift.i19 = shl nuw i64 %.sroa.0.4.insert.ext.i18, 32
  %.sroa.0.4.insert.insert.i20 = or i64 %.sroa.0.4.insert.shift.i19, %.sroa.0.0.insert.ext.i17
  %39 = call float @llvm.nvvm.rsqrt.approx.ftz.f32(float %37)
  %40 = call double @llvm.nvvm.cvt.f64.f32(i32 4, float %39)
  %41 = call double @llvm.nvvm.mul.f64(i32 1, double %40, double %40)
  %42 = fsub double -0.000000e+00, %41
  %43 = bitcast i64 %.sroa.0.4.insert.insert.i20 to double
  %44 = call double @llvm.nvvm.fma.f64(i32 1, double %42, double %43, double 5.000000e-01)
  %45 = call double @llvm.nvvm.fma.f64(i32 1, double %44, double %40, double %40)
  %46 = bitcast i64 %.sroa.0.4.insert.insert.i24 to double
  %47 = call double @llvm.nvvm.mul.f64(i32 1, double %45, double %46)
  %48 = bitcast i64 %.sroa.0.4.insert.insert.i20 to double
  %49 = call double @llvm.nvvm.mul.f64(i32 1, double %48, double %45)
  %50 = bitcast double %45 to i64
  %51 = bitcast double %45 to i64
  %52 = fsub double -0.000000e+00, %47
  %53 = bitcast i64 %.sroa.0.4.insert.insert.i24 to double
  %54 = call double @llvm.nvvm.fma.f64(i32 1, double %52, double %47, double %53)
  %.sroa.0.0.insert.ext.i7 = and i64 %50, 4294967295
  %.sroa.0.4.extract.shift.i1258 = add i64 %51, -4503599627370496
  %.sroa.0.4.insert.ext.i8 = and i64 %.sroa.0.4.extract.shift.i1258, -4294967296
  %.sroa.0.4.insert.insert.i10 = or i64 %.sroa.0.4.insert.ext.i8, %.sroa.0.0.insert.ext.i7
  %55 = bitcast i64 %.sroa.0.4.insert.insert.i10 to double
  %56 = call double @llvm.nvvm.fma.f64(i32 1, double %54, double %55, double %47)
  %57 = fsub double -0.000000e+00, %56
  %58 = bitcast i64 %.sroa.0.4.insert.insert.i24 to double
  %59 = call double @llvm.nvvm.fma.f64(i32 1, double %57, double %56, double %58)
  %60 = fsub double -0.000000e+00, %49
  %61 = call double @llvm.nvvm.fma.f64(i32 1, double %60, double %45, double 5.000000e-01)
  %62 = bitcast i64 %.sroa.0.4.insert.insert.i10 to double
  %63 = bitcast i64 %.sroa.0.4.insert.insert.i10 to double
  %64 = call double @llvm.nvvm.fma.f64(i32 1, double %61, double %62, double %63)
  %65 = call double @llvm.nvvm.fma.f64(i32 2, double %59, double %64, double %56)
  %66 = bitcast double %65 to i64
  %.sroa.0.4.extract.shift.i5 = lshr i64 %66, 32
  %.sroa.0.4.extract.trunc.i6 = trunc i64 %.sroa.0.4.extract.shift.i5 to i32
  %67 = sub nsw i32 %32, %.0
  %68 = shl nsw i32 %67, 19
  %69 = add i32 %68, -535822336
  %70 = add i32 %69, %.sroa.0.4.extract.trunc.i6
  %71 = bitcast double %65 to i64
  %.sroa.0.0.insert.ext.i = and i64 %71, 4294967295
  %.sroa.0.4.insert.ext.i = zext i32 %70 to i64
  %.sroa.0.4.insert.shift.i = shl nuw i64 %.sroa.0.4.insert.ext.i, 32
  %.sroa.0.4.insert.insert.i = or i64 %.sroa.0.4.insert.shift.i, %.sroa.0.0.insert.ext.i
  br label %72

72:                                               ; preds = %14, %31, %25, %21
  %.sroa.055.0 = phi i64 [ %.sroa.0.4.insert.insert.i, %31 ], [ %22, %21 ], [ %.sroa.0.4.insert.insert.i34, %25 ], [ -2251799813685248, %14 ]
  %73 = bitcast i64 %.sroa.055.0 to double
  ret double %73
}

; Function Attrs: noinline
define weak dso_local double @__cuda_sm20_dsqrt_rn_f64_mediumpath_v1(double %0, i32 %1, double %2, double %3, double %4) #0 {
  %6 = icmp ult i32 %1, -54525952
  br i1 %6, label %7, label %17

7:                                                ; preds = %5
  %8 = fcmp ueq double %0, 0.000000e+00
  br i1 %8, label %14, label %9

9:                                                ; preds = %7
  %10 = bitcast double %0 to i64
  %.sroa.0.4.extract.shift.i33 = lshr i64 %10, 32
  %.sroa.0.4.extract.trunc.i34 = trunc i64 %.sroa.0.4.extract.shift.i33 to i32
  %11 = icmp slt i32 %.sroa.0.4.extract.trunc.i34, 0
  br i1 %11, label %28, label %12

12:                                               ; preds = %9
  %13 = icmp sgt i32 %.sroa.0.4.extract.trunc.i34, 2146435071
  br i1 %13, label %14, label %30

14:                                               ; preds = %12, %7
  %15 = call double @llvm.nvvm.add.f64(i32 1, double %0, double %0)
  %16 = bitcast double %15 to i64
  br label %28

17:                                               ; preds = %5
  %18 = call double @llvm.nvvm.fma.f64(i32 2, double %2, double %3, double %4)
  %19 = bitcast double %18 to i64
  %20 = add nsw i64 %19, 1
  %21 = fsub double -0.000000e+00, %18
  %22 = bitcast i64 %20 to double
  %23 = call double @llvm.nvvm.fma.f64(i32 3, double %21, double %22, double %0)
  %24 = fcmp ogt double %23, 0.000000e+00
  %25 = bitcast i64 %20 to double
  %26 = select i1 %24, double %25, double %18
  %27 = bitcast double %26 to i64
  br label %28

28:                                               ; preds = %9, %30, %17, %14
  %.sroa.0.0 = phi i64 [ %16, %14 ], [ %.sroa.0.4.insert.insert.i, %30 ], [ %27, %17 ], [ -2251799813685248, %9 ]
  %29 = bitcast i64 %.sroa.0.0 to double
  ret double %29

30:                                               ; preds = %12
  %31 = call double @llvm.nvvm.mul.f64(i32 1, double 0x4690000000000000, double %0)
  %32 = call double @llvm.nvvm.rsqrt.approx.ftz.f64(double %31)
  %33 = bitcast double %32 to i64
  %34 = bitcast double %32 to i64
  %.sroa.0.4.extract.shift.i19 = and i64 %34, -4294967296
  %.sroa.0.0.insert.ext.i14 = and i64 %33, 4294967295
  %.sroa.0.4.insert.insert.i17 = or i64 %.sroa.0.4.extract.shift.i19, %.sroa.0.0.insert.ext.i14
  %35 = bitcast i64 %.sroa.0.4.insert.insert.i17 to double
  %36 = bitcast i64 %.sroa.0.4.insert.insert.i17 to double
  %37 = call double @llvm.nvvm.mul.f64(i32 1, double %35, double %36)
  %38 = fsub double -0.000000e+00, %37
  %39 = call double @llvm.nvvm.fma.f64(i32 1, double %31, double %38, double 1.000000e+00)
  %40 = call double @llvm.nvvm.fma.f64(i32 1, double 3.750000e-01, double %39, double 5.000000e-01)
  %41 = bitcast i64 %.sroa.0.4.insert.insert.i17 to double
  %42 = call double @llvm.nvvm.mul.f64(i32 1, double %39, double %41)
  %43 = bitcast i64 %.sroa.0.4.insert.insert.i17 to double
  %44 = call double @llvm.nvvm.fma.f64(i32 1, double %40, double %42, double %43)
  %45 = call double @llvm.nvvm.mul.f64(i32 1, double %31, double %44)
  %46 = bitcast double %44 to i64
  %47 = bitcast double %44 to i64
  %.sroa.0.0.insert.ext.i4 = and i64 %46, 4294967295
  %.sroa.0.4.extract.shift.i943 = add i64 %47, -4503599627370496
  %.sroa.0.4.insert.ext.i5 = and i64 %.sroa.0.4.extract.shift.i943, -4294967296
  %.sroa.0.4.insert.insert.i7 = or i64 %.sroa.0.4.insert.ext.i5, %.sroa.0.0.insert.ext.i4
  %48 = fsub double -0.000000e+00, %45
  %49 = call double @llvm.nvvm.fma.f64(i32 1, double %45, double %48, double %31)
  %50 = bitcast i64 %.sroa.0.4.insert.insert.i7 to double
  %51 = call double @llvm.nvvm.fma.f64(i32 1, double %49, double %50, double %45)
  %52 = bitcast double %51 to i64
  %53 = bitcast double %51 to i64
  %.sroa.0.0.insert.ext.i = and i64 %52, 4294967295
  %.sroa.0.4.extract.shift.i44 = add i64 %53, -238690780250636288
  %.sroa.0.4.insert.ext.i = and i64 %.sroa.0.4.extract.shift.i44, -4294967296
  %.sroa.0.4.insert.insert.i = or i64 %.sroa.0.4.insert.ext.i, %.sroa.0.0.insert.ext.i
  br label %28
}

; Unknown intrinsic
declare dso_local double @llvm.nvvm.add.f64(i32, double, double) #2

; Unknown intrinsic
declare dso_local double @llvm.nvvm.rsqrt.approx.ftz.f64(double) #2

; Unknown intrinsic
declare dso_local %struct.ulonglong2 @llvm.nvvm.mul.wide.u.i64(i64, i64) #2

; Unknown intrinsic
declare dso_local float @llvm.nvvm.cvt.f32.i64(i32, i64) #2

; Unknown intrinsic
declare dso_local i64 @llvm.nvvm.cvt.i64.f32(i32, float) #2

; Unknown intrinsic
declare dso_local i64 @llvm.nvvm.mad.hi.u.i64(i64, i64, i64) #2

; Function Attrs: noinline
define weak dso_local i64 @__cuda_sm20_div_u64(i64 %0, i64 %1) #0 {
  %3 = call float @llvm.nvvm.cvt.f32.i64(i32 44, i64 %1)
  %4 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %3)
  %5 = bitcast float %4 to i32
  %6 = add nsw i32 %5, 536870910
  %7 = bitcast i32 %6 to float
  %8 = call i64 @llvm.nvvm.cvt.i64.f32(i32 81, float %7)
  %9 = call %struct.ulonglong2 @llvm.nvvm.mul.wide.u.i64(i64 %8, i64 %1)
  %10 = extractvalue %struct.ulonglong2 %9, 0
  %11 = sub i64 0, %10
  %12 = call i64 @llvm.nvvm.mad.hi.u.i64(i64 %8, i64 %11, i64 %8)
  %13 = call %struct.ulonglong2 @llvm.nvvm.mul.wide.u.i64(i64 %12, i64 %1)
  %14 = extractvalue %struct.ulonglong2 %13, 0
  %15 = sub i64 0, %14
  %16 = call i64 @llvm.nvvm.mad.hi.u.i64(i64 %12, i64 %15, i64 %12)
  %17 = call %struct.ulonglong2 @llvm.nvvm.mul.wide.u.i64(i64 %16, i64 %0)
  %18 = extractvalue %struct.ulonglong2 %17, 1
  %19 = call %struct.ulonglong2 @llvm.nvvm.mul.wide.u.i64(i64 %18, i64 %1)
  %20 = extractvalue %struct.ulonglong2 %19, 0
  %21 = sub i64 %0, %20
  %22 = icmp ult i64 %21, %1
  %23 = sub i64 %21, %1
  %24 = add i64 %18, 1
  %.02 = select i1 %22, i64 %21, i64 %23
  %.01 = select i1 %22, i64 %18, i64 %24
  %25 = icmp ult i64 %.02, %1
  %26 = add i64 %.01, 1
  %spec.select = select i1 %25, i64 %.01, i64 %26
  %27 = icmp eq i64 %1, 0
  %.0 = select i1 %27, i64 -1, i64 %spec.select
  ret i64 %.0
}

; Unknown intrinsic
declare dso_local i32 @llvm.nvvm.unpack.hi.i32.i64(i64) #2

; Function Attrs: noinline
define weak dso_local i64 @__cuda_sm20_div_s64(i64 %0, i64 %1) #0 {
  %3 = call i32 @llvm.nvvm.unpack.hi.i32.i64(i64 %0)
  %4 = icmp slt i32 %3, 0
  %5 = sub nsw i64 0, %0
  %6 = select i1 %4, i64 %5, i64 %0
  %7 = call i32 @llvm.nvvm.unpack.hi.i32.i64(i64 %1)
  %8 = icmp slt i32 %7, 0
  %9 = sub nsw i64 0, %1
  %10 = select i1 %8, i64 %9, i64 %1
  %11 = call float @llvm.nvvm.cvt.f32.i64(i32 44, i64 %10)
  %12 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %11)
  %13 = bitcast float %12 to i32
  %14 = add nsw i32 %13, 536870910
  %15 = bitcast i32 %14 to float
  %16 = call i64 @llvm.nvvm.cvt.i64.f32(i32 81, float %15)
  %17 = call %struct.ulonglong2 @llvm.nvvm.mul.wide.u.i64(i64 %16, i64 %10)
  %18 = extractvalue %struct.ulonglong2 %17, 0
  %19 = sub i64 0, %18
  %20 = call i64 @llvm.nvvm.mad.hi.u.i64(i64 %16, i64 %19, i64 %16)
  %21 = call %struct.ulonglong2 @llvm.nvvm.mul.wide.u.i64(i64 %20, i64 %10)
  %22 = extractvalue %struct.ulonglong2 %21, 0
  %23 = sub i64 0, %22
  %24 = call i64 @llvm.nvvm.mad.hi.u.i64(i64 %20, i64 %23, i64 %20)
  %25 = call %struct.ulonglong2 @llvm.nvvm.mul.wide.u.i64(i64 %24, i64 %6)
  %26 = extractvalue %struct.ulonglong2 %25, 1
  %27 = call %struct.ulonglong2 @llvm.nvvm.mul.wide.u.i64(i64 %26, i64 %10)
  %28 = extractvalue %struct.ulonglong2 %27, 0
  %29 = sub i64 %6, %28
  %30 = icmp ult i64 %29, %10
  %31 = sub i64 %29, %10
  %32 = add i64 %26, 1
  %.03 = select i1 %30, i64 %29, i64 %31
  %.01 = select i1 %30, i64 %26, i64 %32
  %33 = icmp ult i64 %.03, %10
  %34 = add i64 %.01, 1
  %spec.select = select i1 %33, i64 %.01, i64 %34
  %35 = call i32 @llvm.nvvm.unpack.hi.i32.i64(i64 %0)
  %36 = call i32 @llvm.nvvm.unpack.hi.i32.i64(i64 %1)
  %37 = xor i32 %35, %36
  %38 = icmp slt i32 %37, 0
  %39 = sub nsw i64 0, %spec.select
  %.0 = select i1 %38, i64 %39, i64 %spec.select
  %40 = icmp eq i64 %1, 0
  %spec.select48 = select i1 %40, i64 -1, i64 %.0
  ret i64 %spec.select48
}

; Function Attrs: noinline
define weak dso_local i64 @__cuda_sm20_rem_u64(i64 %0, i64 %1) #0 {
  %3 = call float @llvm.nvvm.cvt.f32.i64(i32 44, i64 %1)
  %4 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %3)
  %5 = bitcast float %4 to i32
  %6 = add nsw i32 %5, 536870910
  %7 = bitcast i32 %6 to float
  %8 = call i64 @llvm.nvvm.cvt.i64.f32(i32 81, float %7)
  %9 = call %struct.ulonglong2 @llvm.nvvm.mul.wide.u.i64(i64 %8, i64 %1)
  %10 = extractvalue %struct.ulonglong2 %9, 0
  %11 = sub i64 0, %10
  %12 = call i64 @llvm.nvvm.mad.hi.u.i64(i64 %8, i64 %11, i64 %8)
  %13 = call %struct.ulonglong2 @llvm.nvvm.mul.wide.u.i64(i64 %12, i64 %1)
  %14 = extractvalue %struct.ulonglong2 %13, 0
  %15 = sub i64 0, %14
  %16 = call i64 @llvm.nvvm.mad.hi.u.i64(i64 %12, i64 %15, i64 %12)
  %17 = call %struct.ulonglong2 @llvm.nvvm.mul.wide.u.i64(i64 %16, i64 %0)
  %18 = extractvalue %struct.ulonglong2 %17, 1
  %19 = call %struct.ulonglong2 @llvm.nvvm.mul.wide.u.i64(i64 %18, i64 %1)
  %20 = extractvalue %struct.ulonglong2 %19, 0
  %21 = sub i64 %0, %20
  %22 = icmp ult i64 %21, %1
  %23 = sub i64 %21, %1
  %spec.select = select i1 %22, i64 %21, i64 %23
  %24 = icmp ult i64 %spec.select, %1
  %25 = sub i64 %spec.select, %1
  %.12 = select i1 %24, i64 %spec.select, i64 %25
  %26 = icmp eq i64 %1, 0
  %spec.select47 = select i1 %26, i64 -1, i64 %.12
  ret i64 %spec.select47
}

; Function Attrs: noinline
define weak dso_local i64 @__cuda_sm20_rem_s64(i64 %0, i64 %1) #0 {
  %3 = call i32 @llvm.nvvm.unpack.hi.i32.i64(i64 %0)
  %4 = icmp slt i32 %3, 0
  %5 = sub nsw i64 0, %0
  %6 = select i1 %4, i64 %5, i64 %0
  %7 = call i32 @llvm.nvvm.unpack.hi.i32.i64(i64 %1)
  %8 = icmp slt i32 %7, 0
  %9 = sub nsw i64 0, %1
  %10 = select i1 %8, i64 %9, i64 %1
  %11 = call float @llvm.nvvm.cvt.f32.i64(i32 44, i64 %10)
  %12 = call float @llvm.nvvm.rcp.approx.ftz.f32(float %11)
  %13 = bitcast float %12 to i32
  %14 = add nsw i32 %13, 536870910
  %15 = bitcast i32 %14 to float
  %16 = call i64 @llvm.nvvm.cvt.i64.f32(i32 81, float %15)
  %17 = call %struct.ulonglong2 @llvm.nvvm.mul.wide.u.i64(i64 %16, i64 %10)
  %18 = extractvalue %struct.ulonglong2 %17, 0
  %19 = sub i64 0, %18
  %20 = call i64 @llvm.nvvm.mad.hi.u.i64(i64 %16, i64 %19, i64 %16)
  %21 = call %struct.ulonglong2 @llvm.nvvm.mul.wide.u.i64(i64 %20, i64 %10)
  %22 = extractvalue %struct.ulonglong2 %21, 0
  %23 = sub i64 0, %22
  %24 = call i64 @llvm.nvvm.mad.hi.u.i64(i64 %20, i64 %23, i64 %20)
  %25 = call %struct.ulonglong2 @llvm.nvvm.mul.wide.u.i64(i64 %24, i64 %6)
  %26 = extractvalue %struct.ulonglong2 %25, 1
  %27 = call %struct.ulonglong2 @llvm.nvvm.mul.wide.u.i64(i64 %26, i64 %10)
  %28 = extractvalue %struct.ulonglong2 %27, 0
  %29 = sub i64 %6, %28
  %30 = icmp ult i64 %29, %10
  %31 = sub i64 %29, %10
  %spec.select = select i1 %30, i64 %29, i64 %31
  %32 = icmp ult i64 %spec.select, %10
  %33 = sub i64 %spec.select, %10
  %.12 = select i1 %32, i64 %spec.select, i64 %33
  %34 = call i32 @llvm.nvvm.unpack.hi.i32.i64(i64 %0)
  %35 = icmp slt i32 %34, 0
  %36 = sub nsw i64 0, %.12
  %spec.select48 = select i1 %35, i64 %36, i64 %.12
  %37 = icmp eq i64 %1, 0
  %.14 = select i1 %37, i64 -1, i64 %spec.select48
  ret i64 %.14
}

; Function Attrs: noinline
define weak dso_local double @__cuda_sm20_drsqrt_f64_slowpath_v2(double %0) #0 {
  %2 = bitcast double %0 to i64
  %3 = call i32 @llvm.nvvm.unpack.hi.i32.i64(i64 %2)
  %4 = and i32 %3, -2147483648
  %5 = fcmp oeq double %0, 0.000000e+00
  br i1 %5, label %6, label %10

6:                                                ; preds = %1
  %7 = or i32 %4, 2146435072
  %8 = call i64 @llvm.nvvm.pack.i64.i32(i32 %7, i32 0)
  %9 = bitcast i64 %8 to double
  br label %33

10:                                               ; preds = %1
  %11 = call double @llvm.nvvm.fabs.f64(double %0)
  %12 = fcmp ugt double %11, 0x7FF0000000000000
  br i1 %12, label %13, label %15

13:                                               ; preds = %10
  %14 = fadd double %0, %0
  br label %33

15:                                               ; preds = %10
  %16 = icmp eq i32 %4, 0
  br i1 %16, label %20, label %17

17:                                               ; preds = %15
  %18 = call i64 @llvm.nvvm.pack.i64.i32(i32 -524288, i32 0)
  %19 = bitcast i64 %18 to double
  br label %33

20:                                               ; preds = %15
  %21 = call double @llvm.nvvm.fabs.f64(double %0)
  %22 = fcmp oeq double %21, 0x7FF0000000000000
  br i1 %22, label %33, label %23

23:                                               ; preds = %20
  %24 = fmul double %0, 0x4350000000000000
  %25 = call double @llvm.nvvm.rsqrt.approx.ftz.f64(double %24)
  %26 = call double @llvm.nvvm.mul.f64(i32 1, double %25, double %25)
  %27 = fsub double -0.000000e+00, %26
  %28 = call double @llvm.nvvm.fma.f64(i32 1, double %24, double %27, double 1.000000e+00)
  %29 = call double @llvm.nvvm.fma.f64(i32 1, double 3.750000e-01, double %28, double 5.000000e-01)
  %30 = call double @llvm.nvvm.mul.f64(i32 1, double %28, double %25)
  %31 = call double @llvm.nvvm.fma.f64(i32 1, double %29, double %30, double %25)
  %32 = fmul double %31, 0x41A0000000000000
  br label %33

33:                                               ; preds = %13, %23, %20, %17, %6
  %.3 = phi double [ %9, %6 ], [ %14, %13 ], [ %19, %17 ], [ %32, %23 ], [ 0.000000e+00, %20 ]
  ret double %.3
}

; Unknown intrinsic
declare dso_local i64 @llvm.nvvm.pack.i64.i32(i32, i32) #2

attributes #0 = { noinline "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+ptx32,+sm_20" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+ptx32,+sm_20" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0, !0, !0, !0, !0, !0, !0, !0, !0}
!llvm.module.flags = !{!1}

!0 = !{!"clang version 7.1.0 "}
!1 = !{i32 1, !"wchar_size", i32 4}
