// RUN: split-file %s %t
// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/SrednevaMaxDepthPass%shlibext --pass-pipeline="builtin.module(SrednevaMaxDepthPass)" %t/func1.mlir | FileCheck %t/func1.mlir
// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/SrednevaMaxDepthPass%shlibext --pass-pipeline="builtin.module(SrednevaMaxDepthPass)" %t/func2.mlir | FileCheck %t/func2.mlir
// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/SrednevaMaxDepthPass%shlibext --pass-pipeline="builtin.module(SrednevaMaxDepthPass)" %t/func3t.mlir | FileCheck %t/func3.mlir

//--- func1.mlir
func.func @func1(%arg0: i32) -> i32 {
// CHECK: func.func @func1(%arg0: i32) -> i32 attributes {maxDepth = 1 : i32}
  %0 = arith.muli %arg0, %arg0 : i32
  func.return %0 : i32
}

//--- func2.mlir
func.func @func2(%arg0: i32) -> i32 {
// CHECK: func.func @func2(%arg0: i32) -> i32 attributes {maxDepth = 2 : i32}
  %c5_i32 = arith.constant 5 : i32
  %0 = scf.for %arg1 = %arith.constant 0 : i32 to %c5_i32 step %arith.constant 1 : i32 iter_args(%arg2 = %arg0) -> (i32) {
    %1 = arith.addi %arg2, %arg1 : i32
    scf.yield %1 : i32
  }
  func.return %0 : i32
}

//--- func3.mlir
func.func @func3() {
// CHECK: func.func @func3() attributes {maxDepth = 3 : i32}
  %c3_i32 = arith.constant 3 : i32
  %c5_i32 = arith.constant 5 : i32
  %0 = arith.constant 0 : i32
  %1 = arith.constant 1 : i32
  %2 = arith.subi %c5_i32, %1 : i32
  %3 = arith.subi %c3_i32, %1 : i32
  %4 = arith.cmpi sgt, %c5_i32, %0 : i32
  scf.if %4 {
    %5 = arith.cmpi sgt, %c3_i32, %0 : i32
    scf.if %5 {
      %6 = arith.subi %c5_i32, %1 : i32
      %7 = arith.subi %3, %1 : i32
    }
    %8 = arith.subi %3, %1 : i32
  }
  func.return
}
