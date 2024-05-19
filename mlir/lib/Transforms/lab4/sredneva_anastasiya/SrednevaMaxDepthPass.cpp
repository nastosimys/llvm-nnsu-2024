#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
    FuncOp func = getOperation();
    int maxDepth = getMaxDepth(func.getBody());
    func->setAttr(
        "maxDepth",
        IntegerAttr::get(IntegerType::get(getContext(), 32), maxDepth));
  }

private:
  int getMaxDepth(Block &block) {
    int maxDepth = 1;
    for (Operation &op : block) {
      if (auto regionOp = dyn_cast<RegionBranchOpInterface>(&op)) {
        for (Region &region : regionOp.getRegions()) {
          int depth = 1 + getMaxDepth(region.front());
          if (depth > maxDepth) {
            maxDepth = depth;
          }
        }
      }
    }
    return maxDepth;
    /* int maxDepth = 1;
    ModuleOp module = getOperation();

    module.walk([&](Operation *op) {
      if (auto func = dyn_cast<LLVM::LLVMFuncOp>(op)) {
        int currentDepth = 1;
        while (op) {
          if (auto *regionOp = dyn_cast<RegionBranchOpInterface>(op)) {
            currentDepth++;
            op = regionOp.getRegion().front().getTerminator();
          } else {
            break;
          }
        }
        maxDepth = std::max(maxDepth, currentDepth);
      }
    });
    module.setAttr(
        "MaxDepth",
        IntegerAttr::get(IntegerType::get(getContext(), 32), maxDepth));
    /* func->setAttr(
        "MaxDepth",
        IntegerAttr::get(IntegerType::get(func.getContext(), 32),
                         maxDepth));
  });*/
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
