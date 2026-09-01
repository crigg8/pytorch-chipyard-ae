//===- RISCVRVVBaseHazard.cpp - RVV base hazard workaround ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass works around a hardware hazard where an older RVV memory op may
// observe the value from a younger scalar instruction if both end up allocated
// to the same GPR. For RVV loads, keep the base live through the instructions
// that consume the loaded vector. A consumer cannot execute before the load has
// produced its result, so register allocation cannot reuse the base too early
// for an unrelated scalar value. Follow COPY/PHI forwarding across the CFG;
// the CNN descriptor-copy pattern commonly branches between the load and its
// vector store. Retain the X5 split below for stores and for the existing
// scalar/vector address-producer cases.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-rvv-base-hazard"
#define PASS_NAME "RISC-V RVV Base Hazard Workaround"

namespace {

class RISCVRVVBaseHazard : public MachineFunctionPass {
public:
  static char ID;

  RISCVRVVBaseHazard() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setIsSSA();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addPreserved<MachineDominatorTreeWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return PASS_NAME; }

private:
  static bool holdRVVLoadBaseThroughVectorUses(MachineInstr &MI,
                                               MachineRegisterInfo &MRI,
                                               MachineDominatorTree &MDT);
  static bool isVectorValueForwardingPseudo(const MachineInstr &MI);
  static bool isScalarAddrProducer(const MachineInstr &MI);
  static bool isRVVMemoryPseudo(const MachineInstr &MI);
  static bool usesRegAsMemoryBaseOperand(const MachineInstr &MI, Register Reg);
  static bool hasDirectScalarAddrProducerUserInMBB(const MachineInstr &MI,
                                                   MachineRegisterInfo &MRI);
  static bool hasDirectMemoryUserInMBB(const MachineInstr &MI,
                                       MachineRegisterInfo &MRI);
};

} // namespace

char RISCVRVVBaseHazard::ID = 0;

INITIALIZE_PASS_BEGIN(RISCVRVVBaseHazard, DEBUG_TYPE, PASS_NAME, false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(RISCVRVVBaseHazard, DEBUG_TYPE, PASS_NAME, false, false)

bool RISCVRVVBaseHazard::isVectorValueForwardingPseudo(
    const MachineInstr &MI) {
  if (MI.isCopy() || MI.isPHI())
    return true;

  switch (MI.getOpcode()) {
  default:
    return false;
  case TargetOpcode::REG_SEQUENCE:
  case TargetOpcode::INSERT_SUBREG:
  case TargetOpcode::SUBREG_TO_REG:
  case TargetOpcode::EXTRACT_SUBREG:
    return true;
  }
}

bool RISCVRVVBaseHazard::holdRVVLoadBaseThroughVectorUses(
    MachineInstr &MI, MachineRegisterInfo &MRI, MachineDominatorTree &MDT) {
  if (!isRVVMemoryPseudo(MI) || !MI.mayLoad())
    return false;

  unsigned BaseIdx = MI.getNumDefs() + 1;
  if (BaseIdx >= MI.getNumOperands() || !MI.getOperand(BaseIdx).isReg())
    return false;

  Register BaseReg = MI.getOperand(BaseIdx).getReg();
  if (!BaseReg.isVirtual())
    return false;
  MachineInstr *BaseDef = MRI.getVRegDef(BaseReg);
  if (!BaseDef)
    return false;

  SmallVector<Register, 4> VectorDefs;
  SmallSet<Register, 8> SeenVectorDefs;
  SmallVector<MachineInstr *, 8> Consumers;
  SmallPtrSet<MachineInstr *, 8> SeenConsumers;
  if (!MI.getOperand(0).isReg() || !MI.getOperand(0).isDef() ||
      !MI.getOperand(0).getReg().isVirtual())
    return false;
  VectorDefs.push_back(MI.getOperand(0).getReg());

  bool HasConsumer = false;
  for (size_t I = 0; I != VectorDefs.size(); ++I) {
    Register VectorDef = VectorDefs[I];
    if (!SeenVectorDefs.insert(VectorDef).second)
      continue;

    for (MachineInstr &UseMI : MRI.use_nodbg_instructions(VectorDef)) {
      // Copies, PHIs, and subregister assembly pseudos do not create a
      // hardware dependency. Follow their virtual definitions instead. This
      // also carries the hold across basic-block boundaries.
      if (isVectorValueForwardingPseudo(UseMI)) {
        for (const MachineOperand &MO : UseMI.operands()) {
          if (MO.isReg() && MO.isDef() && MO.getReg().isVirtual())
            VectorDefs.push_back(MO.getReg());
        }
        continue;
      }

      // A PHI may merge the loaded vector with a value from a path on which
      // the base is unavailable. Do not create a non-dominated base use in
      // that case; direct and COPY-forwarded uses remain fully covered.
      if (!MDT.dominates(BaseDef, &UseMI))
        continue;

      HasConsumer = true;
      if (SeenConsumers.insert(&UseMI).second)
        Consumers.push_back(&UseMI);
    }
  }

  // Adding an operand can reallocate a MachineInstr's operand storage. Do it
  // only after every MRI use-list traversal has completed; mutating a consumer
  // in the loop above invalidates the iterator used to find its next use.
  for (MachineInstr *UseMI : Consumers) {
    bool AlreadyUsed = false;
    for (const MachineOperand &MO : UseMI->operands()) {
      if (MO.isReg() && MO.isUse() && MO.getReg() == BaseReg) {
        AlreadyUsed = true;
        break;
      }
    }

    if (!AlreadyUsed)
      UseMI->addOperand(
          MachineOperand::CreateReg(BaseReg, /*isDef=*/false, /*isImp=*/true));
  }

  // A dead or not-yet-forwarded vector value has no dependency that can drain
  // the load. Preserve the previous local fallback by holding the base at the
  // final real instruction in this block. If the load itself ends the block,
  // there is no local younger definition to constrain.
  if (!HasConsumer) {
    MachineInstr *HoldAt = nullptr;
    auto I = std::next(MI.getIterator());
    auto E = MI.getParent()->end();
    for (; I != E; ++I) {
      if (!I->isDebugInstr())
        HoldAt = &*I;
    }
    if (!HoldAt)
      return false;

    bool AlreadyUsed = false;
    for (const MachineOperand &MO : HoldAt->operands()) {
      if (MO.isReg() && MO.isUse() && MO.getReg() == BaseReg) {
        AlreadyUsed = true;
        break;
      }
    }
    if (!AlreadyUsed)
      HoldAt->addOperand(
          MachineOperand::CreateReg(BaseReg, /*isDef=*/false, /*isImp=*/true));
  }

  // The base operand was commonly marked killed at the vector load before the
  // artificial uses were added. Remove stale kill flags so the verifier and
  // the register allocator see the extended live range across the CFG.
  MRI.clearKillFlags(BaseReg);
  return true;
}

bool RISCVRVVBaseHazard::isScalarAddrProducer(const MachineInstr &MI) {
  if (MI.getNumDefs() != 1 || MI.getNumOperands() < 2 ||
      !MI.getOperand(0).isReg())
    return false;

  Register DefReg = MI.getOperand(0).getReg();
  if (!DefReg.isVirtual())
    return false;

  switch (MI.getOpcode()) {
  default:
    return false;
  case RISCV::ADD:
  case RISCV::ADDI:
    return true;
  }
}

bool RISCVRVVBaseHazard::isRVVMemoryPseudo(const MachineInstr &MI) {
  if (!RISCVVPseudosTable::getPseudoInfo(MI.getOpcode()))
    return false;

  return MI.mayLoad() || MI.mayStore();
}

bool RISCVRVVBaseHazard::usesRegAsMemoryBaseOperand(const MachineInstr &MI,
                                                    Register Reg) {
  if (!Reg || (!MI.mayLoad() && !MI.mayStore()))
    return false;

  auto MatchesBaseOperand = [&](unsigned Idx) {
    return Idx < MI.getNumOperands() && MI.getOperand(Idx).isReg() &&
           MI.getOperand(Idx).getReg() == Reg;
  };

  // RVV memory pseudos keep their base after the vector data/passthru
  // operands. This is the same layout used when selecting the older RVV
  // operation's base below.
  if (isRVVMemoryPseudo(MI))
    return MatchesBaseOperand(MI.getNumDefs() + 1);

  if (MI.mayLoad() && MatchesBaseOperand(MI.getNumDefs()))
    return true;

  if (MI.mayStore() && MatchesBaseOperand(MI.getNumDefs() + 1))
    return true;

  return false;
}

bool RISCVRVVBaseHazard::hasDirectMemoryUserInMBB(
    const MachineInstr &MI, MachineRegisterInfo &MRI) {
  Register DefReg = MI.getOperand(0).getReg();
  const MachineBasicBlock *MBB = MI.getParent();

  for (MachineInstr &UseMI : MRI.use_instructions(DefReg)) {
    if (UseMI.getParent() != MBB)
      continue;
    if (usesRegAsMemoryBaseOperand(UseMI, DefReg))
      return true;
  }

  return false;
}

bool RISCVRVVBaseHazard::hasDirectScalarAddrProducerUserInMBB(
    const MachineInstr &MI, MachineRegisterInfo &MRI) {
  Register DefReg = MI.getOperand(0).getReg();
  const MachineBasicBlock *MBB = MI.getParent();

  for (MachineInstr &UseMI : MRI.use_instructions(DefReg)) {
    if (UseMI.getParent() != MBB)
      continue;
    if (isScalarAddrProducer(UseMI))
      return true;
  }

  return false;
}

bool RISCVRVVBaseHazard::runOnMachineFunction(MachineFunction &MF) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  MachineDominatorTree &MDT =
      getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB)
      Changed |= holdRVVLoadBaseThroughVectorUses(MI, MRI, MDT);

    MachineInstr *PendingAddr = nullptr;

    for (auto I = MBB.rbegin(), E = MBB.rend(); I != E; ++I) {
      MachineInstr &MI = *I;
      if (MI.isDebugInstr())
        continue;

      if (isScalarAddrProducer(MI) && hasDirectMemoryUserInMBB(MI, MRI) &&
          !hasDirectScalarAddrProducerUserInMBB(MI, MRI)) {
        PendingAddr = &MI;
        continue;
      }

      if (!PendingAddr || !isRVVMemoryPseudo(MI))
        continue;

      unsigned BaseIdx = MI.getNumDefs() + 1;
      if (BaseIdx >= MI.getNumOperands() || !MI.getOperand(BaseIdx).isReg())
        continue;

      Register BaseReg = MI.getOperand(BaseIdx).getReg();
      if (!BaseReg.isVirtual())
        continue;

      MachineInstr *BaseDef = MRI.getVRegDef(BaseReg);
      if (!BaseDef || BaseDef->getParent() != &MBB ||
          !isScalarAddrProducer(*BaseDef))
        continue;

      if (!MRI.constrainRegClass(BaseReg, &RISCV::GPRJALRRegClass))
        continue;

      Register OldReg = PendingAddr->getOperand(0).getReg();
      Register ScratchReg = MRI.createVirtualRegister(&RISCV::GPRX5RegClass);
      PendingAddr->getOperand(0).setReg(ScratchReg);
      MRI.replaceRegWith(OldReg, ScratchReg);

      PendingAddr = nullptr;
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createRISCVRVVBaseHazardPass() {
  return new RISCVRVVBaseHazard();
}
