//===- RISCVRVVStoreFence.cpp - Drain RVV stores on Saturn ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Saturn may keep an RVV store's scalar base register live in the vector
// backend after the instruction has retired from Rocket. A younger scalar
// instruction can then overwrite the physical register before Saturn has
// consumed the address. Insert a full fence immediately after every RVV
// store. Rocket holds the fence while the vector backend is busy, so no
// younger scalar definition can reach Saturn before the store has drained.
//
// This pass intentionally runs after register allocation and immediately
// before the final RVV pseudo expansion. The pytorch-chipyard RVV toolchain
// targets Saturn, so no separate subtarget feature gate is required here.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-rvv-store-fence"
#define PASS_NAME "RISC-V RVV Store Fence Workaround"

namespace {

constexpr unsigned ReadWrite = RISCVFenceField::R | RISCVFenceField::W;

bool isFullReadWriteFence(const MachineInstr &MI) {
  return MI.getOpcode() == RISCV::FENCE && MI.getNumOperands() >= 2 &&
         MI.getOperand(0).isImm() && MI.getOperand(0).getImm() == ReadWrite &&
         MI.getOperand(1).isImm() && MI.getOperand(1).getImm() == ReadWrite;
}

class RISCVRVVStoreFence : public MachineFunctionPass {
public:
  static char ID;

  RISCVRVVStoreFence() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setNoVRegs();
  }

  StringRef getPassName() const override { return PASS_NAME; }
};

} // namespace

char RISCVRVVStoreFence::ID = 0;

INITIALIZE_PASS(RISCVRVVStoreFence, DEBUG_TYPE, PASS_NAME, false, false)

bool RISCVRVVStoreFence::runOnMachineFunction(MachineFunction &MF) {
  const RISCVInstrInfo *TII = MF.getSubtarget<RISCVSubtarget>().getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E;) {
      MachineInstr &MI = *I++;
      if (!MI.mayStore() ||
          !RISCVVPseudosTable::getPseudoInfo(MI.getOpcode()))
        continue;

      // Avoid adding a duplicate if this pass is explicitly run more than
      // once in a test or a downstream pipeline.
      auto Next = I;
      while (Next != E && Next->isDebugInstr())
        ++Next;
      if (Next != E && isFullReadWriteFence(*Next))
        continue;

      BuildMI(MBB, I, MI.getDebugLoc(), TII->get(RISCV::FENCE))
          .addImm(ReadWrite)
          .addImm(ReadWrite);
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createRISCVRVVStoreFencePass() {
  return new RISCVRVVStoreFence();
}
