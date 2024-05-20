#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
        int maxDepth = getMaxDepth(&funcOp);
        funcOp->setAttr(
            "maxDepth",
            IntegerAttr::get(IntegerType::get(funcOp.getContext(), 32),
                             maxDepth));
      }
    });
  }

private:
  int getMaxDepth(LLVM::LLVMFuncOp *funcOp) {
    int maxDepth = 1;
    bool hasLoops = false;
    bool hasConditions = false;
    std::function<void(Operation *, int, bool, bool)> calculateDepth =
        [&](Operation *op, int depth, bool inLoop, bool inCondition) {
          if (inLoop || inCondition) {
            maxDepth = std::max(maxDepth, depth);
          }

          if (auto loopOp = dyn_cast<LLVM::LLVMLoopOp>(op)) {
            hasLoops = true;
            inLoop = true;
          }

          if (auto condOp = dyn_cast<LLVM::LLVMIfOp>(op)) {
            hasConditions = true;
            inCondition = true;
          }

      if (op->getNumRegions() > 0) {
        Region &region = op->getRegion(0);
        for (Operation &nestedOp : region.front()) {
          calculateDepth(&nestedOp, depth + 1, inLoop, inCondition);
        }
      }
    };
    for (Operation &op : funcOp->getOps()) {
      calculateDepth(&op, 1, false, false);
    }
    if (hasLoops) {
      maxDepth += 1;
    }

    if (hasConditions) {
      maxDepth += 1;
    }
    return maxDepth;
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
