#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include <stack>

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
        int maxDepth = getMaxDepth(funcOp.getBody(), 1);
        funcOp->setAttr(
            "maxDepth",
            IntegerAttr::get(IntegerType::get(funcOp.getContext(), 32),
                             maxDepth));
      }
    });
  }

private:
  int getMaxDepth(Block &block, int currentDepth) {
    int maxDepth = currentDepth;

    for (auto &op : block.getOperations()) {
      if (auto ifOp = dyn_cast<mlir::IfOp>(op)) {
        int nestedDepth = getMaxDepth(ifOp.thenBlock(), currentDepth + 1);
        maxDepth = std::max(maxDepth, nestedDepth);
        if (ifOp.elseRegion().empty())
          continue;
        nestedDepth = getMaxDepth(ifOp.elseBlock(), currentDepth + 1);
        maxDepth = std::max(maxDepth, nestedDepth);
      } else if (auto loopOp = dyn_cast<mlir::scf::ForOp>(op)) {
        int nestedDepth = getMaxDepth(loopOp.getLoopBody(), currentDepth + 1);
        maxDepth = std::max(maxDepth, nestedDepth);
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
