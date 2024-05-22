#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
///
#include "mlir/Dialect/Func/IR/FuncOps.h" //
#include "mlir/IR/BuiltinOps.h"           //
#include "mlir/IR/PatternMatch.h"         //

using namespace mlir;

namespace {
class SrednevaMaxDepthPass
    : public PassWrapper<SrednevaMaxDepthPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const final { return "SrednevaMaxDepthPass"; }
  StringRef getDescription() const final {
    return "Counts the max depth of region nests in the function.";
  }
  void runOnOperation() override {
    getOperation()->walk([&](Operation *op) {
      int maxDepth = 0;

      std::function<void(Operation *, int)> getMaxDepth =
          [&](Operation *op, int depth) {
            maxDepth = std::max(maxDepth, depth);
            for (Region &region : op->getRegions()) {
              for (Block &block : region) {
                for (Operation &op2 : block) {
                  getMaxDepth(&op2, depth + 1);
                }
              }
            }
          };

      getMaxDepth(op, 0);

      op->setAttr(
          "maxDepth",
          IntegerAttr::get(IntegerType::get(op->getContext(), 32), maxDepth));
    });
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
