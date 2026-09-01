	.attribute	4, 16
	.attribute	5, "rv64i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"base-hazard-workaround.ll"
	.text
	.globl	vle_then_lw                     # -- Begin function vle_then_lw
	.p2align	2
	.type	vle_then_lw,@function
vle_then_lw:                            # @vle_then_lw
# %bb.0:                                # %entry
	add	a0, a0, a2
	vsetvli	zero, a3, e32, m2, ta, ma
	vle32.v	v8, (a0)
	addi	t0, a1, 2047
	lw	a0, 1(t0)
	vmv.x.s	a1, v8
	addw	a0, a1, a0
	ret
.Lfunc_end0:
	.size	vle_then_lw, .Lfunc_end0-vle_then_lw
                                        # -- End function
	.globl	vlse_then_lw                    # -- Begin function vlse_then_lw
	.p2align	2
	.type	vlse_then_lw,@function
vlse_then_lw:                           # @vlse_then_lw
# %bb.0:                                # %entry
	add	a0, a0, a2
	vsetvli	zero, a4, e32, m2, ta, ma
	vlse32.v	v8, (a0), a3
	addi	t0, a1, 2047
	lw	a0, 1(t0)
	vmv.x.s	a1, v8
	addw	a0, a1, a0
	ret
.Lfunc_end1:
	.size	vlse_then_lw, .Lfunc_end1-vlse_then_lw
                                        # -- End function
	.globl	vse_then_sw                     # -- Begin function vse_then_sw
	.p2align	2
	.type	vse_then_sw,@function
vse_then_sw:                            # @vse_then_sw
# %bb.0:                                # %entry
	vsetvli	zero, a5, e32, m2, ta, ma
	vle32.v	v8, (a0)
	add	a1, a1, a3
	addi	t0, a2, 2047
	vse32.v	v8, (a1)
	fence	rw, rw
	sw	a4, 1(t0)
	ret
.Lfunc_end2:
	.size	vse_then_sw, .Lfunc_end2-vse_then_sw
                                        # -- End function
	.globl	vle_direct_base_then_lw         # -- Begin function vle_direct_base_then_lw
	.p2align	2
	.type	vle_direct_base_then_lw,@function
vle_direct_base_then_lw:                # @vle_direct_base_then_lw
# %bb.0:                                # %entry
	vsetvli	zero, a2, e32, m2, ta, ma
	vle32.v	v8, (a0)
	addi	a0, a1, 2047
	lw	a0, 1(a0)
	vmv.x.s	a1, v8
	addw	a0, a1, a0
	ret
.Lfunc_end3:
	.size	vle_direct_base_then_lw, .Lfunc_end3-vle_direct_base_then_lw
                                        # -- End function
	.globl	vle_then_nonmem_add             # -- Begin function vle_then_nonmem_add
	.p2align	2
	.type	vle_then_nonmem_add,@function
vle_then_nonmem_add:                    # @vle_then_nonmem_add
# %bb.0:                                # %entry
	add	a0, a0, a2
	vsetvli	zero, a3, e32, m2, ta, ma
	vle32.v	v8, (a0)
	vmv.x.s	a0, v8
	slli	a0, a0, 32
	srli	a0, a0, 32
	add	a0, a1, a0
	addi	a0, a0, 2047
	addi	a0, a0, 1
	ret
.Lfunc_end4:
	.size	vle_then_nonmem_add, .Lfunc_end4-vle_then_nonmem_add
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
