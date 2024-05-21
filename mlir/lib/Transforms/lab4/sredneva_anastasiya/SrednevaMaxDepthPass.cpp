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
    ModuleOp module = getOperation();
    std::map<std::string, int> maxDepthMap;
    module.walk([&](Operation *op) {
      int currentDepth = 0;
      op->walk([&](Operation *childOp) {
        currentDepth++;
        if (currentDepth >
            maxDepthMap[op->getParentOfType<LLVM::LLVMFuncOp>().getName()]) {
          maxDepthMap[op->getParentOfType<LLVM::LLVMFuncOp>().getName()] =
              currentDepth;
        }
      });
    });

    module.walk([&](LLVM::LLVMFuncOp funcOp) {
      int maxDepth = maxDepthMap[funcOp.getName().str()];
      funcOp.setAttr("maxDepth",
                     IntegerAttr::get(IntegerType::get(funcOp.getContext(), 32),
                                      maxDepth));
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
