#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/StandardOps.h"
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
    ModuleOp module = getOperation();
    module.walk([&](LLVM::LLVMFuncOp funcOp) {
      int maxDepth = 1;
      std::function<void(Operation *, int)> calculateDepth = [&](Operation *op,
                                                                 int depth) {
        if (RegionOp *regionOp = dyn_cast<RegionOp>(op)) {
          maxDepth = std::max(maxDepth, depth);
          for (Block &block : regionOp.getBlocks()) {
            for (Operation &nestedOp : block.getOperations()) {
              calculateDepth(&nestedOp, depth + 1);
            }
          }
        }
      };

      for (Block &block : funcOp.getBody().getBlocks()) {
        for (Operation &op : block.getOperations()) {
          calculateDepth(&op, 1);
        }
      }

      funcOp->setAttr("MaxDepth",
                      IntegerAttr::get(
                          IntegerType::get(funcOp.getContext(), 32), maxDepth));
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
