#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsX86.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

#define PASS_NAME "x86-muladd-intrin-pass"
#define PASS_DESC "x86 Muladd Intrinsic Pass"

using namespace llvm;

namespace {
class MulAddPass : public MachineFunctionPass {
public:
  static char ID;
  MulAddPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    SmallVector<MachineInstr *> deletedInstrPtr;

    bool Changed = false;
    for (auto &MBB : MF) {
      for (auto MI = MBB.begin(); MI != MBB.end(); ++MI) {
        if (MI->getOpcode() == X86::MULPDrr) {

          Register Reg = MI->getOperand(0).getReg();
          for (auto NextMI = std::next(MI); NextMI != MBB.end(); ++NextMI) {
            if (NextMI->getOpcode() == X86::ADDPDrr) {

              if (NextMI->getOperand(1).getReg() == Reg ||
                  NextMI->getOperand(2).getReg() == Reg) {

                MachineInstrBuilder BuilderMI = BuildMI(
                    MBB, MI, MI->getDebugLoc(),
                    MF.getSubtarget().getInstrInfo()->get(X86::VFMADD213PDr));
                BuilderMI.addReg(NextMI->getOperand(0).getReg(),
                                 RegState::Define);
                BuilderMI.addReg(MI->getOperand(1).getReg());
                BuilderMI.addReg(MI->getOperand(2).getReg());

                BuilderMI.addReg(NextMI->getOperand(2).getReg());

                deletedInstrPtr.push_back(&*MI);
                deletedInstrPtr.push_back(&*NextMI);
                Changed = true;

                break;
              }
            } else if (NextMI->getOperand(1).getReg() == Reg ||
                       NextMI->getOperand(2).getReg() == Reg) {
              break;
            }
          }
        }
      }
    }
    for (auto it : deletedInstrPtr)
      it->eraseFromParent();

    return Changed;
  }
};
} // namespace

char MulAddPass::ID = 0;
static RegisterPass<MulAddPass> X(PASS_NAME, PASS_DESC);
