// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/SrednevaMaxDepthPass%shlibext --pass-pipeline="builtin.module(SrednevaMaxDepthPass)" %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>} {
  llvm.func @func1() -> (i32 {llvm.noundef}) attributes {maxDepth = 1 : i32, passthrough = ["mustprogress", "noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    // CHECK: llvm.func @func1() {{.*}}attributes {{.*}}maxDepth = 1 : i32
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(5 : i32) : i32
    %2 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %1, %2 : i32, !llvm.ptr
    %3 = llvm.load %2 : !llvm.ptr -> i32
    %4 = llvm.load %2 : !llvm.ptr -> i32
    %5 = llvm.mul %3, %4  : i32
    llvm.return %5 : i32
  }
  llvm.func @func2() -> (i32 {llvm.noundef}) attributes {maxDepth = 2 : i32, passthrough = ["mustprogress", "noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    // CHECK: llvm.func @func2() {{.*}}attributes {{.*}}maxDepth = 2 : i32
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(5 : i32) : i32
    %3 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %4 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %0, %3 : i32, !llvm.ptr
    llvm.store %1, %4 : i32, !llvm.ptr
    llvm.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb3
    %5 = llvm.load %4 : !llvm.ptr -> i32
    %6 = llvm.icmp "slt" %5, %2 : i32
    llvm.cond_br %6, ^bb2, ^bb4
  ^bb2:  // pred: ^bb1
    %7 = llvm.load %3 : !llvm.ptr -> i32
    %8 = llvm.load %4 : !llvm.ptr -> i32
    %9 = llvm.add %7, %8  : i32
    llvm.store %9, %3 : i32, !llvm.ptr
    llvm.br ^bb3
  ^bb3:  // pred: ^bb2
    %10 = llvm.load %4 : !llvm.ptr -> i32
    %11 = llvm.add %10, %0  : i32
    llvm.store %11, %4 : i32, !llvm.ptr
    llvm.br ^bb1
  ^bb4:  // pred: ^bb1
    %12 = llvm.load %3 : !llvm.ptr -> i32
    llvm.return %12 : i32
  }
  llvm.func @func3() attributes {maxDepth = 3 : i32, passthrough = ["mustprogress", "noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    // CHECK: llvm.func @func3() {{.*}}attributes {{.*}}maxDepth = 3 : i32
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(5 : i32) : i32
    %2 = llvm.mlir.constant(3 : i32) : i32
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(-1 : i32) : i32
    %5 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %6 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %1, %5 : i32, !llvm.ptr
    llvm.store %2, %6 : i32, !llvm.ptr
    %7 = llvm.load %5 : !llvm.ptr -> i32
    %8 = llvm.icmp "sgt" %7, %3 : i32
    llvm.cond_br %8, ^bb1, ^bb4
  ^bb1:  // pred: ^bb0
    %9 = llvm.load %6 : !llvm.ptr -> i32
    %10 = llvm.icmp "sgt" %9, %3 : i32
    llvm.cond_br %10, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %11 = llvm.load %5 : !llvm.ptr -> i32
    %12 = llvm.add %11, %4  : i32
    llvm.store %12, %5 : i32, !llvm.ptr
    llvm.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    %13 = llvm.load %6 : !llvm.ptr -> i32
    %14 = llvm.add %13, %4  : i32
    llvm.store %14, %6 : i32, !llvm.ptr
    llvm.br ^bb4
  ^bb4:  // 2 preds: ^bb0, ^bb3
    llvm.return
  }
}