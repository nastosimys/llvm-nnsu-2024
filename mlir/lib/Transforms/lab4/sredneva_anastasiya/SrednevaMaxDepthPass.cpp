#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include <stack>

using namespace mlir;

namespace {
class SrednevaMaxDepthPass
    : public PassWrapper<SrednevaMaxDepthPass, OperationPass<func::FuncOp>> {
public:
  StringRef getArgument() const final { return "SrednevaMaxDepthPass"; }
  StringRef getDescription() const final {
    return "Counts the max depth of region nests in the function.";
  }
  void runOnOperation() override {
    SmallVector<std::pair<Operation *, int>, 8> stack;
    getOperation()->walk([&](Operation *op) { stack.emplace_back(op, 0); });

    while (!stack.empty()) {
      auto [op, currentDepth] = stack.back();
      stack.pop_back();

      int maxDepth = currentDepth;

      for (Region &region : op->getRegions()) {
        for (Block &block : region) {
          for (Operation &nestedOp : block) {
            stack.emplace_back(&nestedOp, currentDepth + 1);
          }
        }
      }

      op->setAttr(
          "maxDepth",
          IntegerAttr::get(IntegerType::get(op->getContext(), 32), maxDepth));
    }
  }
};
} // namespace

MLIR_DECLARE_EXPLICIT_TYPE_ID(SrednevaMaxDepthPass)
MLIR_DEFINE_EXPLICIT_TYPE_ID(SrednevaMaxDepthPass)

PassPluginLibraryInfo getFunctionCallCounterPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "SrednevaMaxDepthPass", LLVM_VERSION_STRING,
          []() { PassRegistration<SrednevaMaxDepthPass>(); }};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return getFunctionCallCounterPassPluginInfo();
}
