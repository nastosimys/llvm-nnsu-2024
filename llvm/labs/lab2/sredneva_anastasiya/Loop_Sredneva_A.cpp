#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
struct LoopWrapperPass : public PassInfoMixin<LoopWrapperPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
    LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);
    FunctionCallee loopStartFunc = F.getParent()->getOrInsertFunction(
        "loop_start", FunctionType::getVoidTy(F.getContext()));
    FunctionCallee loopEndFunc = F.getParent()->getOrInsertFunction(
        "loop_end", FunctionType::getVoidTy(F.getContext()));
    for (auto &BB : F) {
      if (Loop *L = LI.getLoopFor(&BB)) {
        if (L->getHeader() == &BB) {
          BasicBlock *Preheader = L->getLoopPreheader();
          if (!Preheader) {
            Preheader =
                BasicBlock::Create(F.getContext(), "loop.preheader", &F, &BB);
            BranchInst::Create(&BB, Preheader);
          }
          bool hasLoopStart = false;
          for (auto &I : *Preheader) {
            if (auto *CI = dyn_cast<CallInst>(&I)) {
              if (CI->getCalledFunction() == loopStartFunc.getCallee()) {
                hasLoopStart = true;
                break;
              }
            }
          }
          if (!hasLoopStart) {
            IRBuilder<> builder(&*Preheader->getTerminator());
            builder.CreateCall(loopStartFunc);
          }
        }
        if (L->isLoopExiting(&BB)) {
          for (auto it = succ_begin(&BB), e = succ_end(&BB); it != e; ++it) {
            BasicBlock *Successor = *it;
            if (!L->contains(Successor)) {
              bool hasLoopEnd = false;
              for (auto &I : *Successor) {
                if (auto *CI = dyn_cast<CallInst>(&I)) {
                  if (CI->getCalledFunction() == loopEndFunc.getCallee()) {
                    hasLoopEnd = true;
                    break;
                  }
                }
              }
              if (!hasLoopEnd) {
                IRBuilder<> builder(&*Successor->getFirstInsertionPt());
                builder.CreateCall(loopEndFunc);
              }
              break;
            }
          }
        }
      }
    }
    return PreservedAnalyses::all();
  }
  static bool isRequired() { return true; }
};
} // namespace

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "LoopWrapperPass", "v0.1",
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "loop-wrapper") {
                    FPM.addPass(LoopWrapperPass());
                    return true;
                  }
                  return false;
                });
          }};
}
