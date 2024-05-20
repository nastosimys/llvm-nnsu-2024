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
    std::function<void(Operation *, int)> calculateDepth = [&](Operation *op,
                                                               int depth) {
      if (op != nullptr) {
        maxDepth = std::max(maxDepth, depth);
      }
      if (op->getNumRegions() > 0) {
        Region &region = op->getRegion(0);
        if (!region.empty()) {
          for (Operation &nestedOp : region.front()) {
            maxDepth = std::max(maxDepth, depth + 1);
            calculateDepth(&nestedOp, depth + 1);
          }
        }
      }
    };
    for (Operation &op : funcOp->getOps()) {
      calculateDepth(&op, 1);
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
