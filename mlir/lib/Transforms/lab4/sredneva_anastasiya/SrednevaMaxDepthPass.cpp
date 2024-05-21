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
    MLIRContext context;
    OpBuilder builder(&context);
    FuncOp func = builder.create<FuncOp>(builder.getUnknownLoc(), "test_func",
                                         builder.getFunctionType({}, {}));
    Block *block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);
    int maxDepth = getMaxDepth(func.getOperation(), 1);

    func.setAttr("maxDepth", builder.getI32IntegerAttr(maxDepth));
  }

private:
  int getMaxDepth(Operation *op, int currentDepth) {
    int maxDepth = currentDepth;
    op->walk([&](Operation *nestedOp) {
      if (nestedOp->getBlock()) {
        int nestedOpDepth = calculateMaxDepth(nestedOp, currentDepth + 1);
        if (nestedOpDepth > maxDepth) {
          maxDepth = nestedOpDepth;
        }
      }
    });
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
