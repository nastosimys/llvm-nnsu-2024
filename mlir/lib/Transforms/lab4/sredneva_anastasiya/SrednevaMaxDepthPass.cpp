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
        int maxDepth = getMaxDepth(funcOp.getBody());
        funcOp->setAttr(
            "maxDepth",
            IntegerAttr::get(IntegerType::get(op.getContext(), 32), maxDepth));
      }
    });
  }

private:
  int getMaxRegionDepth(Block &block) {
    int maxDepth = 0;
    for (Operation &op : block) {
      if (auto regionOp = dyn_cast<RegionOp>(op)) {
        int depth = 1 + getMaxRegionDepth(regionOp.body().front());
        maxDepth = std::max(maxDepth, depth);
      }
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
