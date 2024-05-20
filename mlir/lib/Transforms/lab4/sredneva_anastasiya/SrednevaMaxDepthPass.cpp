#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

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
    getOperation().walk([&](Operation *op) {
      if (auto funcOp = dyn_cast<LLVM::LLVMFuncOp>(op)) {
        int maxDepth = getMaxDepth(funcOp);
        funcOp->setAttr(
            "maxDepth",
            IntegerAttr::get(IntegerType::get(funcOp.getContext(), 32),
                             maxDepth));
      }
    });
  }

private:
  int getMaxDepth(Operation *op) {
    int maxDepth = 1;
    op->walk([&](Operation *nestedOp) {
      int currentDepth = 1;
      computeDepthRecursive(nestedOp, 1, currentDepth);
      maxDepth = std::max(maxDepth, currentDepth);
    });
    return maxDepth;
  }

  void computeDepthRecursive(Operation *op, int depth, int &maxDepth) {
    if (dyn_cast<scf::ForOp>(op) || dyn_cast<scf::IfOp>(op)) {
      maxDepth = std::max(maxDepth, depth);
      for (Region &region : op->getRegions()) {
        for (Operation &nestedOp : region.front()) {
          computeDepthRecursive(&nestedOp, depth + 1, maxDepth);
        }
      }
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
