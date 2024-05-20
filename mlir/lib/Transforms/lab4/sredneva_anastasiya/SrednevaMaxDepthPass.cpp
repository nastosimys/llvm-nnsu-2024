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
        int maxDepth = getMaxDepth(funcOp.getBody());
        funcOp->setAttr(
            "maxDepth",
            IntegerAttr::get(IntegerType::get(funcOp.getContext(), 32),
                             maxDepth));
      }
    });
  }

private:
  int getMaxDepth(Region &region) {
    std::stack<std::pair<Region *, int>> stack;
    stack.push({&region, 1});
    int maxDepth = 0;

    while (!stack.empty()) {
      auto [currentRegion, currentDepth] = stack.top();
      stack.pop();
      maxDepth = std::max(maxDepth, currentDepth);

      for (Block &block : currentRegion->getBlocks()) {
        Region *parentRegion =
            block.getParent()->getParentOp()->getParentRegion();
        if (parentRegion != null) {
          maxDepth++;
          for (Operation &op : block) {
            int nestedDepth = currentDepth;
            if (op.hasTrait<OpTrait::IsTerminator>()) {
              nestedDepth++;
            }
            for (Region &nestedRegion : op.getRegions()) {
              if (!nestedRegion.empty()) {
                stack.push({&nestedRegion, nestedDepth});
              }
            }
          }
        }
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
