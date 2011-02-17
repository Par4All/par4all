	.file	"hyantes-openmp.c"
	.section	.debug_abbrev,"",@progbits
.Ldebug_abbrev0:
	.section	.debug_info,"",@progbits
.Ldebug_info0:
	.section	.debug_line,"",@progbits
.Ldebug_line0:
	.text
.Ltext0:
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC0:
	.string	"%lf %lf %lf\n"
.LC1:
	.string	"\n"
	.text
	.p2align 4,,15
.globl display
	.type	display, @function
display:
.LFB27:
	.file 1 "hyantes-openmp.c"
	.loc 1 108 0
	.cfi_startproc
.LVL0:
	pushq	%r12
.LCFI0:
	.cfi_def_cfa_offset 16
	xorl	%r12d, %r12d
	.cfi_offset 12, -16
.LVL1:
	pushq	%rbp
.LCFI1:
	.cfi_def_cfa_offset 24
	movq	%rdi, %rbp
	.cfi_offset 6, -24
	pushq	%rbx
.LCFI2:
	.cfi_def_cfa_offset 32
	.p2align 4,,10
	.p2align 3
.L2:
	.loc 1 110 0
	xorl	%ebx, %ebx
	.cfi_offset 3, -32
	.p2align 4,,10
	.p2align 3
.L3:
.LBB27:
.LBB28:
	.file 2 "/usr/include/bits/stdio2.h"
	.loc 2 105 0
	movsd	(%rbp,%rbx), %xmm0
	leaq	.LC0(%rip), %rsi
	movsd	16(%rbp,%rbx), %xmm2
	movl	$1, %edi
.LVL2:
	movsd	8(%rbp,%rbx), %xmm1
	movl	$3, %eax
	addq	$24, %rbx
	call	__printf_chk@PLT
.LBE28:
.LBE27:
	.loc 1 111 0
	cmpq	$7176, %rbx
	jne	.L3
.LBB29:
.LBB30:
	.loc 2 105 0
	leaq	.LC1(%rip), %rsi
	xorl	%eax, %eax
	movl	$1, %edi
.LBE30:
.LBE29:
	.loc 1 110 0
	addq	$1, %r12
	addq	$7176, %rbp
.LBB32:
.LBB31:
	.loc 2 105 0
	call	__printf_chk@PLT
.LBE31:
.LBE32:
	.loc 1 110 0
	cmpq	$290, %r12
	jne	.L2
	.loc 1 115 0
	popq	%rbx
	popq	%rbp
	popq	%r12
.LVL3:
	ret
	.cfi_endproc
.LFE27:
	.size	display, .-display
	.section	.rodata.str1.1
.LC2:
	.string	"r"
.LC3:
	.string	"begin parsing ...\n"
.LC4:
	.string	"%lf%*[ \t]%lf%*[ \t]%lf"
.LC6:
	.string	"parsed %zu towns\n"
	.text
	.p2align 4,,15
.globl read_towns
	.type	read_towns, @function
read_towns:
.LFB25:
	.loc 1 47 0
	.cfi_startproc
.LVL4:
	pushq	%r15
.LCFI3:
	.cfi_def_cfa_offset 16
	.loc 1 48 0
	leaq	.LC2(%rip), %rsi
	.loc 1 47 0
	pushq	%r14
.LCFI4:
	.cfi_def_cfa_offset 24
.LBB33:
.LBB35:
	.loc 2 98 0
	movl	$1, %r14d
	.cfi_offset 14, -24
	.cfi_offset 15, -16
.LBE35:
.LBE33:
	.loc 1 47 0
	pushq	%r13
.LCFI5:
	.cfi_def_cfa_offset 32
	pushq	%r12
.LCFI6:
	.cfi_def_cfa_offset 40
.LBB38:
.LBB36:
	.loc 2 98 0
	xorl	%r12d, %r12d
	.cfi_offset 12, -40
	.cfi_offset 13, -32
.LVL5:
.LBE36:
.LBE38:
	.loc 1 47 0
	pushq	%rbp
.LCFI7:
	.cfi_def_cfa_offset 48
	pushq	%rbx
.LCFI8:
	.cfi_def_cfa_offset 56
	subq	$40, %rsp
.LCFI9:
	.cfi_def_cfa_offset 96
	.loc 1 48 0
	.cfi_offset 3, -56
	.cfi_offset 6, -48
	call	fopen@PLT
.LVL6:
	.loc 1 51 0
	movl	$24, %edi
	.loc 1 48 0
	movq	%rax, %rbx
.LVL7:
	.loc 1 51 0
	call	malloc@PLT
.LBB39:
.LBB34:
	.loc 2 98 0
	movq	stderr@GOTPCREL(%rip), %r15
	leaq	.LC3(%rip), %rdx
.LBE34:
.LBE39:
	.loc 1 51 0
	movq	%rax, %r13
.LBB40:
.LBB37:
	.loc 2 98 0
	movl	$1, %esi
	xorl	%eax, %eax
	movq	(%r15), %rdi
	call	__fprintf_chk@PLT
.LVL8:
	.p2align 4,,10
	.p2align 3
.L17:
.LBE37:
.LBE40:
	.loc 1 54 0
	movq	%rbx, %rdi
	call	feof@PLT
.LVL9:
	testl	%eax, %eax
	jne	.L19
.L13:
	.loc 1 55 0
	cmpq	%r14, %r12
	.p2align 4,,2
	je	.L20
.L10:
	.loc 1 60 0
	leaq	(%r12,%r12,2), %rax
	leaq	.LC4(%rip), %rsi
	movq	%rbx, %rdi
	leaq	(%r13,%rax,8), %rbp
	xorl	%eax, %eax
	leaq	8(%rbp), %rcx
	leaq	16(%rbp), %r8
	movq	%rbp, %rdx
	call	__isoc99_fscanf@PLT
	cmpl	$3, %eax
	je	.L11
.LVL10:
.L18:
	.loc 1 62 0
	movq	%rbx, %rdi
	call	feof@PLT
.LVL11:
	testl	%eax, %eax
	.p2align 4,,2
	jne	.L17
	.loc 1 63 0
	movq	%rbx, %rdi
	call	fgetc@PLT
	.loc 1 64 0
	cmpb	$13, %al
.LVL12:
	.p2align 4,,2
	je	.L17
	cmpb	$10, %al
	.p2align 4,,2
	jne	.L18
	.loc 1 54 0
	movq	%rbx, %rdi
	.p2align 4,,5
	call	feof@PLT
.LVL13:
	testl	%eax, %eax
	.p2align 4,,2
	je	.L13
.L19:
	.loc 1 77 0
	movq	%rbx, %rdi
	call	fclose@PLT
	.loc 1 78 0
	leaq	(%r12,%r12,2), %rsi
	movq	%r13, %rdi
	salq	$3, %rsi
	call	realloc@PLT
.LBB41:
.LBB42:
	.loc 2 98 0
	movq	(%r15), %rdi
	leaq	.LC6(%rip), %rdx
.LBE42:
.LBE41:
	.loc 1 78 0
	movq	%rax, %rbx
.LVL14:
.LBB44:
.LBB43:
	.loc 2 98 0
	movq	%r12, %rcx
	movl	$1, %esi
	xorl	%eax, %eax
	call	__fprintf_chk@PLT
.LBE43:
.LBE44:
	.loc 1 86 0
	addq	$40, %rsp
	movq	%rbx, %rdx
	movq	%r12, %rax
	popq	%rbx
	popq	%rbp
	popq	%r12
.LVL15:
	popq	%r13
	popq	%r14
	popq	%r15
	ret
.LVL16:
	.p2align 4,,10
	.p2align 3
.L11:
	.loc 1 72 0
	movsd	.LC5(%rip), %xmm0
	.loc 1 74 0
	addq	$1, %r12
	.loc 1 72 0
	mulsd	(%rbp), %xmm0
	movsd	%xmm0, (%rbp)
	.loc 1 73 0
	movsd	.LC5(%rip), %xmm0
	mulsd	8(%rbp), %xmm0
	movsd	%xmm0, 8(%rbp)
	jmp	.L17
	.p2align 4,,10
	.p2align 3
.L20:
	.loc 1 56 0
	leaq	(%r12,%r12), %r14
	.loc 1 58 0
	movq	%r13, %rdi
	leaq	(%r14,%r12,4), %rsi
	salq	$3, %rsi
	call	realloc@PLT
	movq	%rax, %r13
	jmp	.L10
	.cfi_endproc
.LFE25:
	.size	read_towns, .-read_towns
	.p2align 4,,15
	.type	run.omp_fn.0, @function
run.omp_fn.0:
.LFB29:
	.loc 1 93 0
	.cfi_startproc
.LVL17:
	pushq	%r15
.LCFI10:
	.cfi_def_cfa_offset 16
	pushq	%r14
.LCFI11:
	.cfi_def_cfa_offset 24
	pushq	%r13
.LCFI12:
	.cfi_def_cfa_offset 32
	pushq	%r12
.LCFI13:
	.cfi_def_cfa_offset 40
	pushq	%rbp
.LCFI14:
	.cfi_def_cfa_offset 48
	pushq	%rbx
.LCFI15:
	.cfi_def_cfa_offset 56
	movq	%rdi, %rbx
	.cfi_offset 3, -56
	.cfi_offset 6, -48
	.cfi_offset 12, -40
	.cfi_offset 13, -32
	.cfi_offset 14, -24
	.cfi_offset 15, -16
	subq	$136, %rsp
.LCFI16:
	.cfi_def_cfa_offset 192
	.loc 1 94 0
	call	omp_get_num_threads@PLT
.LVL18:
	movslq	%eax,%rbp
	call	omp_get_thread_num@PLT
	movl	$290, %ecx
	movl	%eax, %esi
	xorl	%edx, %edx
	movq	%rcx, %rax
	movslq	%esi,%rsi
	divq	%rbp
	xorl	%edx, %edx
	imulq	%rax, %rbp
	cmpq	$290, %rbp
	setne	%dl
	leaq	(%rdx,%rax), %rax
	imulq	%rax, %rsi
	addq	%rsi, %rax
	movq	%rsi, 88(%rsp)
	cmpq	$290, %rax
	cmovbe	%rax, %rcx
	cmpq	%rcx, %rsi
	movq	%rcx, 96(%rsp)
	jae	.L34
	imulq	$7176, %rsi, %r15
	leaq	120(%rsp), %r14
	leaq	112(%rsp), %r13
.LVL19:
.L24:
.LBB45:
	.loc 1 96 0
	movq	88(%rsp), %rax
.LBE45:
	.loc 1 94 0
	movq	$0, 80(%rsp)
.LBB47:
	.loc 1 96 0
	shrq	%rax
	movq	%rax, 104(%rsp)
	movq	88(%rsp), %rax
	andl	$1, %eax
	orq	%rax, 104(%rsp)
	.p2align 4,,10
	.p2align 3
.L23:
	cmpq	$0, 88(%rsp)
	js	.L25
	cvtsi2sdq	88(%rsp), %xmm0
.LVL20:
	movsd	%xmm0, 64(%rsp)
.L26:
	mulsd	16(%rbx), %xmm0
	movq	80(%rsp), %rcx
	movq	32(%rbx), %rax
	leaq	(%rcx,%rcx,2), %rdx
	.loc 1 97 0
	testq	%rcx, %rcx
	.loc 1 96 0
	leaq	(%r15,%rdx,8), %rdx
	addsd	(%rbx), %xmm0
	mulsd	.LC7(%rip), %xmm0
	divsd	.LC8(%rip), %xmm0
	movsd	%xmm0, (%rdx,%rax)
	.loc 1 97 0
	js	.L27
	cvtsi2sdq	80(%rsp), %xmm0
	movsd	%xmm0, 56(%rsp)
.L28:
	mulsd	16(%rbx), %xmm0
	movq	80(%rsp), %rdx
	.loc 1 98 0
	xorl	%ebp, %ebp
	.loc 1 97 0
	leaq	(%rdx,%rdx,2), %rax
	salq	$3, %rax
	addsd	8(%rbx), %xmm0
	leaq	(%rax,%r15), %rdx
	movq	%rdx, %rcx
	addq	32(%rbx), %rcx
	mulsd	.LC7(%rip), %xmm0
	divsd	.LC8(%rip), %xmm0
	movsd	%xmm0, 8(%rcx)
	.loc 1 98 0
	addq	32(%rbx), %rdx
	movq	$0, 16(%rdx)
.LBB46:
	.loc 1 102 0
	movq	%rax, 72(%rsp)
.LVL21:
	.p2align 4,,10
	.p2align 3
.L29:
	.loc 1 100 0
	movsd	16(%rbx), %xmm1
	movq	%r13, %rsi
	movsd	64(%rsp), %xmm0
.LVL22:
	movq	%r14, %rdi
	movsd	%xmm1, 32(%rsp)
	movq	%rbp, %r12
	mulsd	%xmm1, %xmm0
	addsd	(%rbx), %xmm0
	call	sincos@PLT
	addq	40(%rbx), %r12
	movsd	112(%rsp), %xmm3
	movsd	120(%rsp), %xmm2
	movq	%r13, %rsi
	movq	%r14, %rdi
	movsd	(%r12), %xmm0
	movsd	%xmm2, (%rsp)
	movsd	%xmm3, 16(%rsp)
	call	sincos@PLT
	movsd	32(%rsp), %xmm1
	movsd	56(%rsp), %xmm0
	mulsd	%xmm1, %xmm0
	addsd	8(%rbx), %xmm0
	subsd	8(%r12), %xmm0
	call	cos@PLT
	movsd	16(%rsp), %xmm3
	movsd	(%rsp), %xmm2
	mulsd	112(%rsp), %xmm3
	mulsd	120(%rsp), %xmm2
	mulsd	%xmm0, %xmm3
	addsd	%xmm2, %xmm3
	movapd	%xmm3, %xmm0
	call	acos@PLT
	mulsd	.LC10(%rip), %xmm0
.LVL23:
	.loc 1 101 0
	movsd	24(%rbx), %xmm1
	ucomisd	%xmm0, %xmm1
	jbe	.L33
	.loc 1 102 0
	addsd	.LC11(%rip), %xmm0
.LVL24:
	movq	40(%rbx), %rdx
	movq	%r15, %rax
	addq	32(%rbx), %rax
	addq	72(%rsp), %rax
	movsd	16(%rdx,%rbp), %xmm1
	divsd	%xmm0, %xmm1
	addsd	16(%rax), %xmm1
	movsd	%xmm1, 16(%rax)
.LVL25:
.L33:
	.loc 1 101 0
	addq	$24, %rbp
.LBE46:
	.loc 1 99 0
	cmpq	$69072, %rbp
	jne	.L29
	.loc 1 95 0
	addq	$1, 80(%rsp)
	cmpq	$299, 80(%rsp)
	jne	.L23
.LBE47:
	.loc 1 94 0
	addq	$1, 88(%rsp)
	addq	$7176, %r15
	movq	88(%rsp), %rdx
	cmpq	%rdx, 96(%rsp)
	ja	.L24
.L34:
	.loc 1 93 0
	addq	$136, %rsp
	popq	%rbx
.LVL26:
	popq	%rbp
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	ret
.LVL27:
	.p2align 4,,10
	.p2align 3
.L27:
.LBB48:
	.loc 1 97 0
	movq	80(%rsp), %rax
	movq	80(%rsp), %rdx
	shrq	%rax
	andl	$1, %edx
	orq	%rdx, %rax
	cvtsi2sdq	%rax, %xmm0
	addsd	%xmm0, %xmm0
	movsd	%xmm0, 56(%rsp)
	jmp	.L28
.LVL28:
.L25:
	.loc 1 96 0
	cvtsi2sdq	104(%rsp), %xmm0
.LVL29:
	addsd	%xmm0, %xmm0
	movsd	%xmm0, 64(%rsp)
	jmp	.L26
.LBE48:
	.cfi_endproc
.LFE29:
	.size	run.omp_fn.0, .-run.omp_fn.0
	.section	.rodata.str1.1
.LC12:
	.string	"begin computation ...\n"
.LC13:
	.string	"end computation ...\n"
	.text
	.p2align 4,,15
.globl run
	.type	run, @function
run:
.LFB26:
	.loc 1 88 0
	.cfi_startproc
.LVL30:
	movq	%rbx, -24(%rsp)
	movq	%rbp, -16(%rsp)
.LBB49:
.LBB51:
	.loc 2 98 0
	leaq	.LC12(%rip), %rdx
.LBE51:
.LBE49:
	.loc 1 88 0
	movq	%r12, -8(%rsp)
	subq	$136, %rsp
.LCFI17:
	.cfi_def_cfa_offset 144
.LBB53:
.LBB50:
	.loc 2 98 0
	movq	stderr@GOTPCREL(%rip), %rbx
	.cfi_offset 12, -16
	.cfi_offset 6, -24
	.cfi_offset 3, -32
.LBE50:
.LBE53:
	.loc 1 88 0
	movq	%rdi, %r12
	movq	%rsi, %rbp
.LBB54:
.LBB52:
	.loc 2 98 0
	xorl	%eax, %eax
	movl	$1, %esi
.LVL31:
	movq	(%rbx), %rdi
.LVL32:
	movsd	%xmm0, 48(%rsp)
.LVL33:
	movsd	%xmm1, 32(%rsp)
.LVL34:
	movsd	%xmm4, 16(%rsp)
.LVL35:
	movsd	%xmm5, (%rsp)
.LVL36:
	call	__fprintf_chk@PLT
.LVL37:
.LBE52:
.LBE54:
	.loc 1 93 0
	movq	%rbp, 104(%rsp)
	leaq	64(%rsp), %rbp
.LVL38:
	leaq	run.omp_fn.0(%rip), %rdi
	movsd	48(%rsp), %xmm0
	xorl	%edx, %edx
	movsd	32(%rsp), %xmm1
	movq	%rbp, %rsi
	movsd	16(%rsp), %xmm4
	movq	%r12, 96(%rsp)
	movsd	(%rsp), %xmm5
	movsd	%xmm0, 64(%rsp)
.LVL39:
	movsd	%xmm1, 72(%rsp)
.LVL40:
	movsd	%xmm4, 80(%rsp)
.LVL41:
	movsd	%xmm5, 88(%rsp)
.LVL42:
	call	GOMP_parallel_start@PLT
.LVL43:
	movq	%rbp, %rdi
	call	run.omp_fn.0
	call	GOMP_parallel_end@PLT
.LBB55:
.LBB56:
	.loc 2 98 0
	movq	(%rbx), %rdi
	leaq	.LC13(%rip), %rdx
	movl	$1, %esi
	xorl	%eax, %eax
	call	__fprintf_chk@PLT
.LBE56:
.LBE55:
	.loc 1 106 0
	movq	112(%rsp), %rbx
	movq	120(%rsp), %rbp
	movq	128(%rsp), %r12
.LVL44:
	addq	$136, %rsp
	ret
	.cfi_endproc
.LFE26:
	.size	run, .-run
	.p2align 4,,15
.globl main
	.type	main, @function
main:
.LFB28:
	.loc 1 117 0
	.cfi_startproc
.LVL45:
	pushq	%rbx
.LCFI18:
	.cfi_def_cfa_offset 16
	.loc 1 118 0
	movl	$1, %eax
	.loc 1 117 0
	movq	%rsi, %rbx
	.cfi_offset 3, -16
	subq	$2081152, %rsp
.LCFI19:
	.cfi_def_cfa_offset 2081168
	.loc 1 118 0
	cmpl	$8, %edi
	je	.L46
.LVL46:
	.loc 1 133 0
	addq	$2081152, %rsp
	popq	%rbx
.LVL47:
	ret
.LVL48:
	.p2align 4,,10
	.p2align 3
.L46:
.LBB57:
	.loc 1 122 0
	movq	8(%rsi), %rdi
.LVL49:
	call	read_towns@PLT
	movq	%rax, 80(%rsp)
	movq	%rax, 2081136(%rsp)
.LBB58:
.LBB59:
	.file 3 "/usr/include/stdlib.h"
	.loc 3 281 0
	xorl	%esi, %esi
.LBE59:
.LBE58:
	.loc 1 122 0
	movq	%rdx, 88(%rsp)
	movq	%rdx, 2081144(%rsp)
.LBB61:
.LBB60:
	.loc 3 281 0
	movq	16(%rbx), %rdi
	call	strtod@PLT
	movsd	%xmm0, 72(%rsp)
.LBE60:
.LBE61:
.LBB62:
.LBB63:
	movq	24(%rbx), %rdi
	xorl	%esi, %esi
	call	strtod@PLT
.LBE63:
.LBE62:
.LBB64:
.LBB65:
	movq	32(%rbx), %rdi
	xorl	%esi, %esi
	movsd	%xmm0, (%rsp)
	call	strtod@PLT
.LBE65:
.LBE64:
.LBB66:
.LBB67:
	movq	40(%rbx), %rdi
	xorl	%esi, %esi
	movsd	%xmm0, 16(%rsp)
	call	strtod@PLT
.LBE67:
.LBE66:
.LBB68:
.LBB69:
	movq	48(%rbx), %rdi
	xorl	%esi, %esi
	movsd	%xmm0, 32(%rsp)
	call	strtod@PLT
.LBE69:
.LBE68:
.LBB70:
.LBB72:
	movq	56(%rbx), %rdi
	xorl	%esi, %esi
	movsd	%xmm0, 48(%rsp)
.LBE72:
.LBE70:
	.loc 1 129 0
	leaq	96(%rsp), %rbx
.LVL50:
.LBB74:
.LBB71:
	.loc 3 281 0
	call	strtod@PLT
.LBE71:
.LBE74:
	.loc 1 129 0
	movsd	48(%rsp), %xmm4
	movq	2081144(%rsp), %rsi
.LBB75:
.LBB73:
	.loc 3 281 0
	movapd	%xmm0, %xmm5
.LVL51:
.LBE73:
.LBE75:
	.loc 1 129 0
	movq	%rbx, %rdi
	movsd	.LC8(%rip), %xmm0
	movsd	32(%rsp), %xmm3
	movsd	16(%rsp), %xmm2
	movsd	(%rsp), %xmm1
	mulsd	%xmm0, %xmm4
	mulsd	%xmm0, %xmm3
	mulsd	%xmm0, %xmm2
	mulsd	%xmm0, %xmm1
	mulsd	72(%rsp), %xmm0
	movsd	.LC7(%rip), %xmm6
	divsd	%xmm6, %xmm4
	divsd	%xmm6, %xmm0
	divsd	%xmm6, %xmm3
	divsd	%xmm6, %xmm2
	divsd	%xmm6, %xmm1
	call	run@PLT
.LVL52:
	.loc 1 130 0
	movq	%rbx, %rdi
	call	display@PLT
	xorl	%eax, %eax
.LBE57:
	.loc 1 133 0
	addq	$2081152, %rsp
	popq	%rbx
	ret
	.cfi_endproc
.LFE28:
	.size	main, .-main
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC5:
	.long	2723323193
	.long	1066524486
	.align 8
.LC7:
	.long	0
	.long	1080459264
	.align 8
.LC8:
	.long	1413754136
	.long	1074340347
	.align 8
.LC10:
	.long	0
	.long	1085857792
	.align 8
.LC11:
	.long	0
	.long	1072693248
	.text
.Letext0:
	.section	.debug_loc,"",@progbits
.Ldebug_loc0:
.LLST0:
	.quad	.LFB27-.Ltext0
	.quad	.LCFI0-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 8
	.quad	.LCFI0-.Ltext0
	.quad	.LCFI1-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 16
	.quad	.LCFI1-.Ltext0
	.quad	.LCFI2-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 24
	.quad	.LCFI2-.Ltext0
	.quad	.LFE27-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 32
	.quad	0x0
	.quad	0x0
.LLST1:
	.quad	.LVL0-.Ltext0
	.quad	.LVL2-.Ltext0
	.value	0x1
	.byte	0x55
	.quad	0x0
	.quad	0x0
.LLST2:
	.quad	.LVL1-.Ltext0
	.quad	.LVL3-.Ltext0
	.value	0x1
	.byte	0x5c
	.quad	0x0
	.quad	0x0
.LLST3:
	.quad	.LFB25-.Ltext0
	.quad	.LCFI3-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 8
	.quad	.LCFI3-.Ltext0
	.quad	.LCFI4-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 16
	.quad	.LCFI4-.Ltext0
	.quad	.LCFI5-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 24
	.quad	.LCFI5-.Ltext0
	.quad	.LCFI6-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 32
	.quad	.LCFI6-.Ltext0
	.quad	.LCFI7-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 40
	.quad	.LCFI7-.Ltext0
	.quad	.LCFI8-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 48
	.quad	.LCFI8-.Ltext0
	.quad	.LCFI9-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 56
	.quad	.LCFI9-.Ltext0
	.quad	.LFE25-.Ltext0
	.value	0x3
	.byte	0x77
	.sleb128 96
	.quad	0x0
	.quad	0x0
.LLST4:
	.quad	.LVL4-.Ltext0
	.quad	.LVL6-.Ltext0
	.value	0x1
	.byte	0x55
	.quad	0x0
	.quad	0x0
.LLST5:
	.quad	.LVL7-.Ltext0
	.quad	.LVL14-.Ltext0
	.value	0x1
	.byte	0x53
	.quad	.LVL16-.Ltext0
	.quad	.LFE25-.Ltext0
	.value	0x1
	.byte	0x53
	.quad	0x0
	.quad	0x0
.LLST6:
	.quad	.LVL5-.Ltext0
	.quad	.LVL15-.Ltext0
	.value	0x1
	.byte	0x5c
	.quad	.LVL16-.Ltext0
	.quad	.LFE25-.Ltext0
	.value	0x1
	.byte	0x5c
	.quad	0x0
	.quad	0x0
.LLST7:
	.quad	.LVL8-.Ltext0
	.quad	.LVL9-.Ltext0
	.value	0x1
	.byte	0x50
	.quad	.LVL10-.Ltext0
	.quad	.LVL11-.Ltext0
	.value	0x1
	.byte	0x50
	.quad	.LVL12-.Ltext0
	.quad	.LVL13-.Ltext0
	.value	0x1
	.byte	0x50
	.quad	0x0
	.quad	0x0
.LLST8:
	.quad	.LFB29-.Ltext0
	.quad	.LCFI10-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 8
	.quad	.LCFI10-.Ltext0
	.quad	.LCFI11-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 16
	.quad	.LCFI11-.Ltext0
	.quad	.LCFI12-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 24
	.quad	.LCFI12-.Ltext0
	.quad	.LCFI13-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 32
	.quad	.LCFI13-.Ltext0
	.quad	.LCFI14-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 40
	.quad	.LCFI14-.Ltext0
	.quad	.LCFI15-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 48
	.quad	.LCFI15-.Ltext0
	.quad	.LCFI16-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 56
	.quad	.LCFI16-.Ltext0
	.quad	.LFE29-.Ltext0
	.value	0x3
	.byte	0x77
	.sleb128 192
	.quad	0x0
	.quad	0x0
.LLST9:
	.quad	.LVL17-.Ltext0
	.quad	.LVL18-.Ltext0
	.value	0x1
	.byte	0x55
	.quad	.LVL18-.Ltext0
	.quad	.LVL26-.Ltext0
	.value	0x1
	.byte	0x53
	.quad	.LVL27-.Ltext0
	.quad	.LFE29-.Ltext0
	.value	0x1
	.byte	0x53
	.quad	0x0
	.quad	0x0
.LLST10:
	.quad	.LVL19-.Ltext0
	.quad	.LVL20-.Ltext0
	.value	0x1
	.byte	0x61
	.quad	.LVL21-.Ltext0
	.quad	.LVL22-.Ltext0
	.value	0x1
	.byte	0x61
	.quad	.LVL23-.Ltext0
	.quad	.LVL24-.Ltext0
	.value	0x1
	.byte	0x61
	.quad	.LVL25-.Ltext0
	.quad	.LVL27-.Ltext0
	.value	0x1
	.byte	0x61
	.quad	.LVL28-.Ltext0
	.quad	.LVL29-.Ltext0
	.value	0x1
	.byte	0x61
	.quad	0x0
	.quad	0x0
.LLST11:
	.quad	.LFB26-.Ltext0
	.quad	.LCFI17-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 8
	.quad	.LCFI17-.Ltext0
	.quad	.LFE26-.Ltext0
	.value	0x3
	.byte	0x77
	.sleb128 144
	.quad	0x0
	.quad	0x0
.LLST12:
	.quad	.LVL30-.Ltext0
	.quad	.LVL33-.Ltext0
	.value	0x1
	.byte	0x61
	.quad	.LVL39-.Ltext0
	.quad	.LVL43-.Ltext0
	.value	0x1
	.byte	0x61
	.quad	0x0
	.quad	0x0
.LLST13:
	.quad	.LVL30-.Ltext0
	.quad	.LVL34-.Ltext0
	.value	0x1
	.byte	0x62
	.quad	.LVL40-.Ltext0
	.quad	.LVL43-.Ltext0
	.value	0x1
	.byte	0x62
	.quad	0x0
	.quad	0x0
.LLST14:
	.quad	.LVL30-.Ltext0
	.quad	.LVL37-.Ltext0
	.value	0x1
	.byte	0x63
	.quad	0x0
	.quad	0x0
.LLST15:
	.quad	.LVL30-.Ltext0
	.quad	.LVL37-.Ltext0
	.value	0x1
	.byte	0x64
	.quad	0x0
	.quad	0x0
.LLST16:
	.quad	.LVL30-.Ltext0
	.quad	.LVL35-.Ltext0
	.value	0x1
	.byte	0x65
	.quad	.LVL41-.Ltext0
	.quad	.LVL43-.Ltext0
	.value	0x1
	.byte	0x65
	.quad	0x0
	.quad	0x0
.LLST17:
	.quad	.LVL30-.Ltext0
	.quad	.LVL36-.Ltext0
	.value	0x1
	.byte	0x66
	.quad	.LVL42-.Ltext0
	.quad	.LVL43-.Ltext0
	.value	0x1
	.byte	0x66
	.quad	0x0
	.quad	0x0
.LLST18:
	.quad	.LVL30-.Ltext0
	.quad	.LVL32-.Ltext0
	.value	0x1
	.byte	0x55
	.quad	.LVL32-.Ltext0
	.quad	.LVL44-.Ltext0
	.value	0x1
	.byte	0x5c
	.quad	0x0
	.quad	0x0
.LLST19:
	.quad	.LVL30-.Ltext0
	.quad	.LVL31-.Ltext0
	.value	0x1
	.byte	0x54
	.quad	.LVL31-.Ltext0
	.quad	.LVL38-.Ltext0
	.value	0x1
	.byte	0x56
	.quad	0x0
	.quad	0x0
.LLST20:
	.quad	.LFB28-.Ltext0
	.quad	.LCFI18-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 8
	.quad	.LCFI18-.Ltext0
	.quad	.LCFI19-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 16
	.quad	.LCFI19-.Ltext0
	.quad	.LFE28-.Ltext0
	.value	0x5
	.byte	0x77
	.sleb128 2081168
	.quad	0x0
	.quad	0x0
.LLST21:
	.quad	.LVL45-.Ltext0
	.quad	.LVL49-.Ltext0
	.value	0x1
	.byte	0x55
	.quad	0x0
	.quad	0x0
.LLST22:
	.quad	.LVL45-.Ltext0
	.quad	.LVL46-.Ltext0
	.value	0x1
	.byte	0x54
	.quad	.LVL46-.Ltext0
	.quad	.LVL47-.Ltext0
	.value	0x1
	.byte	0x53
	.quad	.LVL47-.Ltext0
	.quad	.LVL48-.Ltext0
	.value	0x1
	.byte	0x54
	.quad	.LVL48-.Ltext0
	.quad	.LVL50-.Ltext0
	.value	0x1
	.byte	0x53
	.quad	0x0
	.quad	0x0
.LLST23:
	.quad	.LVL51-.Ltext0
	.quad	.LVL52-.Ltext0
	.value	0x1
	.byte	0x66
	.quad	0x0
	.quad	0x0
	.file 4 "/usr/lib/gcc/x86_64-linux-gnu/4.4.5/include/stddef.h"
	.file 5 "/usr/include/bits/types.h"
	.file 6 "/usr/include/stdio.h"
	.file 7 "/usr/include/libio.h"
	.section	.debug_info
	.long	0x91b
	.value	0x2
	.long	.Ldebug_abbrev0
	.byte	0x8
	.uleb128 0x1
	.long	.LASF81
	.byte	0x1
	.long	.LASF82
	.long	.LASF83
	.quad	.Ltext0
	.quad	.Letext0
	.long	.Ldebug_line0
	.uleb128 0x2
	.long	.LASF7
	.byte	0x4
	.byte	0xd3
	.long	0x38
	.uleb128 0x3
	.byte	0x8
	.byte	0x7
	.long	.LASF0
	.uleb128 0x3
	.byte	0x1
	.byte	0x8
	.long	.LASF1
	.uleb128 0x3
	.byte	0x2
	.byte	0x7
	.long	.LASF2
	.uleb128 0x3
	.byte	0x4
	.byte	0x7
	.long	.LASF3
	.uleb128 0x3
	.byte	0x1
	.byte	0x6
	.long	.LASF4
	.uleb128 0x3
	.byte	0x2
	.byte	0x5
	.long	.LASF5
	.uleb128 0x4
	.byte	0x4
	.byte	0x5
	.string	"int"
	.uleb128 0x3
	.byte	0x8
	.byte	0x5
	.long	.LASF6
	.uleb128 0x2
	.long	.LASF8
	.byte	0x5
	.byte	0x8d
	.long	0x69
	.uleb128 0x2
	.long	.LASF9
	.byte	0x5
	.byte	0x8e
	.long	0x69
	.uleb128 0x5
	.byte	0x8
	.byte	0x7
	.uleb128 0x6
	.byte	0x8
	.uleb128 0x7
	.byte	0x8
	.long	0x91
	.uleb128 0x3
	.byte	0x1
	.byte	0x6
	.long	.LASF10
	.uleb128 0x2
	.long	.LASF11
	.byte	0x6
	.byte	0x31
	.long	0xa3
	.uleb128 0x8
	.long	.LASF41
	.byte	0xd8
	.byte	0x6
	.byte	0x2d
	.long	0x26f
	.uleb128 0x9
	.long	.LASF12
	.byte	0x7
	.value	0x110
	.long	0x62
	.byte	0x2
	.byte	0x23
	.uleb128 0x0
	.uleb128 0x9
	.long	.LASF13
	.byte	0x7
	.value	0x115
	.long	0x8b
	.byte	0x2
	.byte	0x23
	.uleb128 0x8
	.uleb128 0x9
	.long	.LASF14
	.byte	0x7
	.value	0x116
	.long	0x8b
	.byte	0x2
	.byte	0x23
	.uleb128 0x10
	.uleb128 0x9
	.long	.LASF15
	.byte	0x7
	.value	0x117
	.long	0x8b
	.byte	0x2
	.byte	0x23
	.uleb128 0x18
	.uleb128 0x9
	.long	.LASF16
	.byte	0x7
	.value	0x118
	.long	0x8b
	.byte	0x2
	.byte	0x23
	.uleb128 0x20
	.uleb128 0x9
	.long	.LASF17
	.byte	0x7
	.value	0x119
	.long	0x8b
	.byte	0x2
	.byte	0x23
	.uleb128 0x28
	.uleb128 0x9
	.long	.LASF18
	.byte	0x7
	.value	0x11a
	.long	0x8b
	.byte	0x2
	.byte	0x23
	.uleb128 0x30
	.uleb128 0x9
	.long	.LASF19
	.byte	0x7
	.value	0x11b
	.long	0x8b
	.byte	0x2
	.byte	0x23
	.uleb128 0x38
	.uleb128 0x9
	.long	.LASF20
	.byte	0x7
	.value	0x11c
	.long	0x8b
	.byte	0x2
	.byte	0x23
	.uleb128 0x40
	.uleb128 0x9
	.long	.LASF21
	.byte	0x7
	.value	0x11e
	.long	0x8b
	.byte	0x2
	.byte	0x23
	.uleb128 0x48
	.uleb128 0x9
	.long	.LASF22
	.byte	0x7
	.value	0x11f
	.long	0x8b
	.byte	0x2
	.byte	0x23
	.uleb128 0x50
	.uleb128 0x9
	.long	.LASF23
	.byte	0x7
	.value	0x120
	.long	0x8b
	.byte	0x2
	.byte	0x23
	.uleb128 0x58
	.uleb128 0x9
	.long	.LASF24
	.byte	0x7
	.value	0x122
	.long	0x2ad
	.byte	0x2
	.byte	0x23
	.uleb128 0x60
	.uleb128 0x9
	.long	.LASF25
	.byte	0x7
	.value	0x124
	.long	0x2b3
	.byte	0x2
	.byte	0x23
	.uleb128 0x68
	.uleb128 0x9
	.long	.LASF26
	.byte	0x7
	.value	0x126
	.long	0x62
	.byte	0x2
	.byte	0x23
	.uleb128 0x70
	.uleb128 0x9
	.long	.LASF27
	.byte	0x7
	.value	0x12a
	.long	0x62
	.byte	0x2
	.byte	0x23
	.uleb128 0x74
	.uleb128 0x9
	.long	.LASF28
	.byte	0x7
	.value	0x12c
	.long	0x70
	.byte	0x2
	.byte	0x23
	.uleb128 0x78
	.uleb128 0x9
	.long	.LASF29
	.byte	0x7
	.value	0x130
	.long	0x46
	.byte	0x3
	.byte	0x23
	.uleb128 0x80
	.uleb128 0x9
	.long	.LASF30
	.byte	0x7
	.value	0x131
	.long	0x54
	.byte	0x3
	.byte	0x23
	.uleb128 0x82
	.uleb128 0x9
	.long	.LASF31
	.byte	0x7
	.value	0x132
	.long	0x2b9
	.byte	0x3
	.byte	0x23
	.uleb128 0x83
	.uleb128 0x9
	.long	.LASF32
	.byte	0x7
	.value	0x136
	.long	0x2c9
	.byte	0x3
	.byte	0x23
	.uleb128 0x88
	.uleb128 0x9
	.long	.LASF33
	.byte	0x7
	.value	0x13f
	.long	0x7b
	.byte	0x3
	.byte	0x23
	.uleb128 0x90
	.uleb128 0x9
	.long	.LASF34
	.byte	0x7
	.value	0x148
	.long	0x89
	.byte	0x3
	.byte	0x23
	.uleb128 0x98
	.uleb128 0x9
	.long	.LASF35
	.byte	0x7
	.value	0x149
	.long	0x89
	.byte	0x3
	.byte	0x23
	.uleb128 0xa0
	.uleb128 0x9
	.long	.LASF36
	.byte	0x7
	.value	0x14a
	.long	0x89
	.byte	0x3
	.byte	0x23
	.uleb128 0xa8
	.uleb128 0x9
	.long	.LASF37
	.byte	0x7
	.value	0x14b
	.long	0x89
	.byte	0x3
	.byte	0x23
	.uleb128 0xb0
	.uleb128 0x9
	.long	.LASF38
	.byte	0x7
	.value	0x14c
	.long	0x2d
	.byte	0x3
	.byte	0x23
	.uleb128 0xb8
	.uleb128 0x9
	.long	.LASF39
	.byte	0x7
	.value	0x14e
	.long	0x62
	.byte	0x3
	.byte	0x23
	.uleb128 0xc0
	.uleb128 0x9
	.long	.LASF40
	.byte	0x7
	.value	0x150
	.long	0x2cf
	.byte	0x3
	.byte	0x23
	.uleb128 0xc4
	.byte	0x0
	.uleb128 0xa
	.long	.LASF84
	.byte	0x7
	.byte	0xb4
	.uleb128 0x8
	.long	.LASF42
	.byte	0x18
	.byte	0x7
	.byte	0xba
	.long	0x2ad
	.uleb128 0xb
	.long	.LASF43
	.byte	0x7
	.byte	0xbb
	.long	0x2ad
	.byte	0x2
	.byte	0x23
	.uleb128 0x0
	.uleb128 0xb
	.long	.LASF44
	.byte	0x7
	.byte	0xbc
	.long	0x2b3
	.byte	0x2
	.byte	0x23
	.uleb128 0x8
	.uleb128 0xb
	.long	.LASF45
	.byte	0x7
	.byte	0xc0
	.long	0x62
	.byte	0x2
	.byte	0x23
	.uleb128 0x10
	.byte	0x0
	.uleb128 0x7
	.byte	0x8
	.long	0x276
	.uleb128 0x7
	.byte	0x8
	.long	0xa3
	.uleb128 0xc
	.long	0x91
	.long	0x2c9
	.uleb128 0xd
	.long	0x86
	.byte	0x0
	.byte	0x0
	.uleb128 0x7
	.byte	0x8
	.long	0x26f
	.uleb128 0xc
	.long	0x91
	.long	0x2df
	.uleb128 0xd
	.long	0x86
	.byte	0x13
	.byte	0x0
	.uleb128 0x7
	.byte	0x8
	.long	0x2e5
	.uleb128 0xe
	.long	0x91
	.uleb128 0x3
	.byte	0x8
	.byte	0x5
	.long	.LASF46
	.uleb128 0x3
	.byte	0x4
	.byte	0x4
	.long	.LASF47
	.uleb128 0x3
	.byte	0x8
	.byte	0x4
	.long	.LASF48
	.uleb128 0x2
	.long	.LASF49
	.byte	0x1
	.byte	0x18
	.long	0x2f8
	.uleb128 0xf
	.byte	0x18
	.byte	0x1
	.byte	0x19
	.long	0x33d
	.uleb128 0xb
	.long	.LASF50
	.byte	0x1
	.byte	0x1a
	.long	0x2ff
	.byte	0x2
	.byte	0x23
	.uleb128 0x0
	.uleb128 0xb
	.long	.LASF51
	.byte	0x1
	.byte	0x1b
	.long	0x2ff
	.byte	0x2
	.byte	0x23
	.uleb128 0x8
	.uleb128 0xb
	.long	.LASF52
	.byte	0x1
	.byte	0x1c
	.long	0x2ff
	.byte	0x2
	.byte	0x23
	.uleb128 0x10
	.byte	0x0
	.uleb128 0x2
	.long	.LASF53
	.byte	0x1
	.byte	0x1d
	.long	0x30a
	.uleb128 0xf
	.byte	0x10
	.byte	0x1
	.byte	0x1e
	.long	0x36b
	.uleb128 0x10
	.string	"n"
	.byte	0x1
	.byte	0x1f
	.long	0x2d
	.byte	0x2
	.byte	0x23
	.uleb128 0x0
	.uleb128 0xb
	.long	.LASF54
	.byte	0x1
	.byte	0x20
	.long	0x36b
	.byte	0x2
	.byte	0x23
	.uleb128 0x8
	.byte	0x0
	.uleb128 0x7
	.byte	0x8
	.long	0x33d
	.uleb128 0x2
	.long	.LASF55
	.byte	0x1
	.byte	0x21
	.long	0x348
	.uleb128 0x11
	.byte	0x1
	.long	.LASF56
	.byte	0x2
	.byte	0x67
	.byte	0x1
	.long	0x62
	.byte	0x3
	.byte	0x1
	.long	0x39c
	.uleb128 0x12
	.long	.LASF58
	.byte	0x2
	.byte	0x67
	.long	0x2df
	.uleb128 0x13
	.byte	0x0
	.uleb128 0x11
	.byte	0x1
	.long	.LASF57
	.byte	0x2
	.byte	0x60
	.byte	0x1
	.long	0x62
	.byte	0x3
	.byte	0x1
	.long	0x3c7
	.uleb128 0x12
	.long	.LASF59
	.byte	0x2
	.byte	0x60
	.long	0x3c7
	.uleb128 0x12
	.long	.LASF58
	.byte	0x2
	.byte	0x60
	.long	0x2df
	.uleb128 0x13
	.byte	0x0
	.uleb128 0x7
	.byte	0x8
	.long	0x98
	.uleb128 0x14
	.byte	0x1
	.long	.LASF85
	.byte	0x3
	.value	0x117
	.byte	0x1
	.long	0x2f8
	.byte	0x3
	.long	0x3ed
	.uleb128 0x15
	.long	.LASF60
	.byte	0x3
	.value	0x117
	.long	0x2df
	.byte	0x0
	.uleb128 0x16
	.byte	0x1
	.long	.LASF61
	.byte	0x1
	.byte	0x6b
	.byte	0x1
	.quad	.LFB27
	.quad	.LFE27
	.long	.LLST0
	.long	0x46d
	.uleb128 0x17
	.string	"pt"
	.byte	0x1
	.byte	0x6b
	.long	0x47e
	.long	.LLST1
	.uleb128 0x18
	.string	"i"
	.byte	0x1
	.byte	0x6d
	.long	0x2d
	.long	.LLST2
	.uleb128 0x19
	.string	"j"
	.byte	0x1
	.byte	0x6d
	.long	0x2d
	.uleb128 0x1a
	.long	0x37c
	.quad	.LBB27
	.quad	.LBE27
	.byte	0x1
	.byte	0x70
	.long	0x453
	.uleb128 0x1b
	.long	0x38f
	.byte	0x0
	.uleb128 0x1c
	.long	0x37c
	.quad	.LBB29
	.long	.Ldebug_ranges0+0x0
	.byte	0x1
	.byte	0x71
	.uleb128 0x1b
	.long	0x38f
	.byte	0x0
	.byte	0x0
	.uleb128 0xc
	.long	0x33d
	.long	0x47e
	.uleb128 0x1d
	.long	0x86
	.value	0x12a
	.byte	0x0
	.uleb128 0x7
	.byte	0x8
	.long	0x46d
	.uleb128 0x1e
	.byte	0x1
	.long	.LASF74
	.byte	0x1
	.byte	0x2e
	.byte	0x1
	.long	0x371
	.quad	.LFB25
	.quad	.LFE25
	.long	.LLST3
	.long	0x53c
	.uleb128 0x1f
	.long	.LASF62
	.byte	0x1
	.byte	0x2e
	.long	0x2df
	.long	.LLST4
	.uleb128 0x18
	.string	"fd"
	.byte	0x1
	.byte	0x30
	.long	0x3c7
	.long	.LLST5
	.uleb128 0x20
	.long	.LASF63
	.byte	0x1
	.byte	0x31
	.long	0x2d
	.long	.LLST6
	.uleb128 0x18
	.string	"c"
	.byte	0x1
	.byte	0x32
	.long	0x91
	.long	.LLST7
	.uleb128 0x21
	.long	.LASF64
	.byte	0x1
	.byte	0x33
	.long	0x371
	.uleb128 0x22
	.long	.LASF65
	.byte	0x1
	.byte	0x43
	.uleb128 0x22
	.long	.LASF66
	.byte	0x1
	.byte	0x45
	.uleb128 0x23
	.long	0x39c
	.quad	.LBB33
	.long	.Ldebug_ranges0+0x30
	.byte	0x1
	.byte	0x34
	.long	0x51d
	.uleb128 0x1b
	.long	0x3ba
	.uleb128 0x1b
	.long	0x3af
	.byte	0x0
	.uleb128 0x1c
	.long	0x39c
	.quad	.LBB41
	.long	.Ldebug_ranges0+0x80
	.byte	0x1
	.byte	0x50
	.uleb128 0x1b
	.long	0x3ba
	.uleb128 0x1b
	.long	0x3af
	.byte	0x0
	.byte	0x0
	.uleb128 0x24
	.long	.LASF86
	.byte	0x1
	.byte	0x1
	.quad	.LFB29
	.quad	.LFE29
	.long	.LLST8
	.long	0x60f
	.uleb128 0x25
	.long	.LASF67
	.long	0x66b
	.byte	0x1
	.long	.LLST9
	.uleb128 0x19
	.string	"k"
	.byte	0x1
	.byte	0x59
	.long	0x2d
	.uleb128 0x26
	.string	"j"
	.byte	0x1
	.byte	0x59
	.long	0x2d
	.byte	0x3
	.byte	0x77
	.sleb128 80
	.uleb128 0x26
	.string	"t"
	.byte	0x1
	.byte	0x57
	.long	0x36b
	.byte	0x4
	.byte	0x73
	.sleb128 0
	.byte	0x23
	.uleb128 0x28
	.uleb128 0x26
	.string	"pt"
	.byte	0x1
	.byte	0x57
	.long	0x47e
	.byte	0x4
	.byte	0x73
	.sleb128 0
	.byte	0x23
	.uleb128 0x20
	.uleb128 0x27
	.long	.LASF68
	.byte	0x1
	.byte	0x57
	.long	0x2ff
	.byte	0x4
	.byte	0x73
	.sleb128 0
	.byte	0x23
	.uleb128 0x18
	.uleb128 0x27
	.long	.LASF69
	.byte	0x1
	.byte	0x57
	.long	0x2ff
	.byte	0x4
	.byte	0x73
	.sleb128 0
	.byte	0x23
	.uleb128 0x10
	.uleb128 0x27
	.long	.LASF70
	.byte	0x1
	.byte	0x57
	.long	0x2ff
	.byte	0x4
	.byte	0x73
	.sleb128 0
	.byte	0x23
	.uleb128 0x8
	.uleb128 0x27
	.long	.LASF71
	.byte	0x1
	.byte	0x57
	.long	0x2ff
	.byte	0x2
	.byte	0x73
	.sleb128 0
	.uleb128 0x28
	.long	.Ldebug_ranges0+0xb0
	.uleb128 0x26
	.string	"i"
	.byte	0x1
	.byte	0x59
	.long	0x2d
	.byte	0x3
	.byte	0x77
	.sleb128 88
	.uleb128 0x29
	.quad	.LBB46
	.quad	.LBE46
	.uleb128 0x18
	.string	"tmp"
	.byte	0x1
	.byte	0x64
	.long	0x2ff
	.long	.LLST10
	.byte	0x0
	.byte	0x0
	.byte	0x0
	.uleb128 0x2a
	.long	.LASF87
	.byte	0x30
	.long	0x66b
	.uleb128 0xb
	.long	.LASF71
	.byte	0x1
	.byte	0x5d
	.long	0x2ff
	.byte	0x2
	.byte	0x23
	.uleb128 0x0
	.uleb128 0xb
	.long	.LASF70
	.byte	0x1
	.byte	0x5d
	.long	0x2ff
	.byte	0x2
	.byte	0x23
	.uleb128 0x8
	.uleb128 0xb
	.long	.LASF69
	.byte	0x1
	.byte	0x5d
	.long	0x2ff
	.byte	0x2
	.byte	0x23
	.uleb128 0x10
	.uleb128 0xb
	.long	.LASF68
	.byte	0x1
	.byte	0x5d
	.long	0x2ff
	.byte	0x2
	.byte	0x23
	.uleb128 0x18
	.uleb128 0x10
	.string	"pt"
	.byte	0x1
	.byte	0x5d
	.long	0x47e
	.byte	0x2
	.byte	0x23
	.uleb128 0x20
	.uleb128 0x10
	.string	"t"
	.byte	0x1
	.byte	0x5d
	.long	0x36b
	.byte	0x2
	.byte	0x23
	.uleb128 0x28
	.byte	0x0
	.uleb128 0x7
	.byte	0x8
	.long	0x60f
	.uleb128 0x2b
	.byte	0x1
	.string	"run"
	.byte	0x1
	.byte	0x57
	.byte	0x1
	.quad	.LFB26
	.quad	.LFE26
	.long	.LLST11
	.long	0x767
	.uleb128 0x1f
	.long	.LASF71
	.byte	0x1
	.byte	0x57
	.long	0x2ff
	.long	.LLST12
	.uleb128 0x1f
	.long	.LASF70
	.byte	0x1
	.byte	0x57
	.long	0x2ff
	.long	.LLST13
	.uleb128 0x1f
	.long	.LASF72
	.byte	0x1
	.byte	0x57
	.long	0x2ff
	.long	.LLST14
	.uleb128 0x1f
	.long	.LASF73
	.byte	0x1
	.byte	0x57
	.long	0x2ff
	.long	.LLST15
	.uleb128 0x1f
	.long	.LASF69
	.byte	0x1
	.byte	0x57
	.long	0x2ff
	.long	.LLST16
	.uleb128 0x1f
	.long	.LASF68
	.byte	0x1
	.byte	0x57
	.long	0x2ff
	.long	.LLST17
	.uleb128 0x17
	.string	"pt"
	.byte	0x1
	.byte	0x57
	.long	0x47e
	.long	.LLST18
	.uleb128 0x17
	.string	"t"
	.byte	0x1
	.byte	0x57
	.long	0x36b
	.long	.LLST19
	.uleb128 0x19
	.string	"i"
	.byte	0x1
	.byte	0x59
	.long	0x2d
	.uleb128 0x19
	.string	"j"
	.byte	0x1
	.byte	0x59
	.long	0x2d
	.uleb128 0x19
	.string	"k"
	.byte	0x1
	.byte	0x59
	.long	0x2d
	.uleb128 0x23
	.long	0x39c
	.quad	.LBB49
	.long	.Ldebug_ranges0+0xf0
	.byte	0x1
	.byte	0x5b
	.long	0x744
	.uleb128 0x1b
	.long	0x3ba
	.uleb128 0x1b
	.long	0x3af
	.byte	0x0
	.uleb128 0x2c
	.long	0x39c
	.quad	.LBB55
	.quad	.LBE55
	.byte	0x1
	.byte	0x69
	.uleb128 0x1b
	.long	0x3ba
	.uleb128 0x1b
	.long	0x3af
	.byte	0x0
	.byte	0x0
	.uleb128 0x1e
	.byte	0x1
	.long	.LASF75
	.byte	0x1
	.byte	0x74
	.byte	0x1
	.long	0x62
	.quad	.LFB28
	.quad	.LFE28
	.long	.LLST20
	.long	0x8d9
	.uleb128 0x1f
	.long	.LASF76
	.byte	0x1
	.byte	0x74
	.long	0x62
	.long	.LLST21
	.uleb128 0x1f
	.long	.LASF77
	.byte	0x1
	.byte	0x74
	.long	0x8d9
	.long	.LLST22
	.uleb128 0x29
	.quad	.LBB57
	.quad	.LBE57
	.uleb128 0x26
	.string	"pt"
	.byte	0x1
	.byte	0x79
	.long	0x8df
	.byte	0x5
	.byte	0x91
	.sleb128 -2081072
	.uleb128 0x26
	.string	"t"
	.byte	0x1
	.byte	0x7a
	.long	0x371
	.byte	0x2
	.byte	0x91
	.sleb128 -32
	.uleb128 0x21
	.long	.LASF71
	.byte	0x1
	.byte	0x80
	.long	0x2ff
	.uleb128 0x21
	.long	.LASF70
	.byte	0x1
	.byte	0x80
	.long	0x2ff
	.uleb128 0x21
	.long	.LASF72
	.byte	0x1
	.byte	0x80
	.long	0x2ff
	.uleb128 0x21
	.long	.LASF73
	.byte	0x1
	.byte	0x80
	.long	0x2ff
	.uleb128 0x21
	.long	.LASF69
	.byte	0x1
	.byte	0x80
	.long	0x2ff
	.uleb128 0x20
	.long	.LASF68
	.byte	0x1
	.byte	0x80
	.long	0x2ff
	.long	.LLST23
	.uleb128 0x23
	.long	0x3cd
	.quad	.LBB58
	.long	.Ldebug_ranges0+0x130
	.byte	0x1
	.byte	0x80
	.long	0x83a
	.uleb128 0x1b
	.long	0x3e0
	.byte	0x0
	.uleb128 0x1a
	.long	0x3cd
	.quad	.LBB62
	.quad	.LBE62
	.byte	0x1
	.byte	0x80
	.long	0x85b
	.uleb128 0x1b
	.long	0x3e0
	.byte	0x0
	.uleb128 0x1a
	.long	0x3cd
	.quad	.LBB64
	.quad	.LBE64
	.byte	0x1
	.byte	0x80
	.long	0x87c
	.uleb128 0x1b
	.long	0x3e0
	.byte	0x0
	.uleb128 0x1a
	.long	0x3cd
	.quad	.LBB66
	.quad	.LBE66
	.byte	0x1
	.byte	0x80
	.long	0x89d
	.uleb128 0x1b
	.long	0x3e0
	.byte	0x0
	.uleb128 0x1a
	.long	0x3cd
	.quad	.LBB68
	.quad	.LBE68
	.byte	0x1
	.byte	0x80
	.long	0x8be
	.uleb128 0x1b
	.long	0x3e0
	.byte	0x0
	.uleb128 0x1c
	.long	0x3cd
	.quad	.LBB70
	.long	.Ldebug_ranges0+0x160
	.byte	0x1
	.byte	0x80
	.uleb128 0x1b
	.long	0x3e0
	.byte	0x0
	.byte	0x0
	.byte	0x0
	.uleb128 0x7
	.byte	0x8
	.long	0x8b
	.uleb128 0xc
	.long	0x33d
	.long	0x8f7
	.uleb128 0x1d
	.long	0x86
	.value	0x121
	.uleb128 0x1d
	.long	0x86
	.value	0x12a
	.byte	0x0
	.uleb128 0x2d
	.long	.LASF78
	.byte	0x6
	.byte	0xa5
	.long	0x2b3
	.byte	0x1
	.byte	0x1
	.uleb128 0x2d
	.long	.LASF79
	.byte	0x6
	.byte	0xa6
	.long	0x2b3
	.byte	0x1
	.byte	0x1
	.uleb128 0x2d
	.long	.LASF80
	.byte	0x6
	.byte	0xa7
	.long	0x2b3
	.byte	0x1
	.byte	0x1
	.byte	0x0
	.section	.debug_abbrev
	.uleb128 0x1
	.uleb128 0x11
	.byte	0x1
	.uleb128 0x25
	.uleb128 0xe
	.uleb128 0x13
	.uleb128 0xb
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x1b
	.uleb128 0xe
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x1
	.uleb128 0x10
	.uleb128 0x6
	.byte	0x0
	.byte	0x0
	.uleb128 0x2
	.uleb128 0x16
	.byte	0x0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.byte	0x0
	.byte	0x0
	.uleb128 0x3
	.uleb128 0x24
	.byte	0x0
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3e
	.uleb128 0xb
	.uleb128 0x3
	.uleb128 0xe
	.byte	0x0
	.byte	0x0
	.uleb128 0x4
	.uleb128 0x24
	.byte	0x0
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3e
	.uleb128 0xb
	.uleb128 0x3
	.uleb128 0x8
	.byte	0x0
	.byte	0x0
	.uleb128 0x5
	.uleb128 0x24
	.byte	0x0
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3e
	.uleb128 0xb
	.byte	0x0
	.byte	0x0
	.uleb128 0x6
	.uleb128 0xf
	.byte	0x0
	.uleb128 0xb
	.uleb128 0xb
	.byte	0x0
	.byte	0x0
	.uleb128 0x7
	.uleb128 0xf
	.byte	0x0
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.byte	0x0
	.byte	0x0
	.uleb128 0x8
	.uleb128 0x13
	.byte	0x1
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x1
	.uleb128 0x13
	.byte	0x0
	.byte	0x0
	.uleb128 0x9
	.uleb128 0xd
	.byte	0x0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0x5
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x38
	.uleb128 0xa
	.byte	0x0
	.byte	0x0
	.uleb128 0xa
	.uleb128 0x16
	.byte	0x0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.byte	0x0
	.byte	0x0
	.uleb128 0xb
	.uleb128 0xd
	.byte	0x0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x38
	.uleb128 0xa
	.byte	0x0
	.byte	0x0
	.uleb128 0xc
	.uleb128 0x1
	.byte	0x1
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x1
	.uleb128 0x13
	.byte	0x0
	.byte	0x0
	.uleb128 0xd
	.uleb128 0x21
	.byte	0x0
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2f
	.uleb128 0xb
	.byte	0x0
	.byte	0x0
	.uleb128 0xe
	.uleb128 0x26
	.byte	0x0
	.uleb128 0x49
	.uleb128 0x13
	.byte	0x0
	.byte	0x0
	.uleb128 0xf
	.uleb128 0x13
	.byte	0x1
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x1
	.uleb128 0x13
	.byte	0x0
	.byte	0x0
	.uleb128 0x10
	.uleb128 0xd
	.byte	0x0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x38
	.uleb128 0xa
	.byte	0x0
	.byte	0x0
	.uleb128 0x11
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0xc
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x27
	.uleb128 0xc
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x20
	.uleb128 0xb
	.uleb128 0x34
	.uleb128 0xc
	.uleb128 0x1
	.uleb128 0x13
	.byte	0x0
	.byte	0x0
	.uleb128 0x12
	.uleb128 0x5
	.byte	0x0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.byte	0x0
	.byte	0x0
	.uleb128 0x13
	.uleb128 0x18
	.byte	0x0
	.byte	0x0
	.byte	0x0
	.uleb128 0x14
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0xc
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0x5
	.uleb128 0x27
	.uleb128 0xc
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x20
	.uleb128 0xb
	.uleb128 0x1
	.uleb128 0x13
	.byte	0x0
	.byte	0x0
	.uleb128 0x15
	.uleb128 0x5
	.byte	0x0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0x5
	.uleb128 0x49
	.uleb128 0x13
	.byte	0x0
	.byte	0x0
	.uleb128 0x16
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0xc
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x27
	.uleb128 0xc
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x1
	.uleb128 0x40
	.uleb128 0x6
	.uleb128 0x1
	.uleb128 0x13
	.byte	0x0
	.byte	0x0
	.uleb128 0x17
	.uleb128 0x5
	.byte	0x0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x6
	.byte	0x0
	.byte	0x0
	.uleb128 0x18
	.uleb128 0x34
	.byte	0x0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x6
	.byte	0x0
	.byte	0x0
	.uleb128 0x19
	.uleb128 0x34
	.byte	0x0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.byte	0x0
	.byte	0x0
	.uleb128 0x1a
	.uleb128 0x1d
	.byte	0x1
	.uleb128 0x31
	.uleb128 0x13
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x1
	.uleb128 0x58
	.uleb128 0xb
	.uleb128 0x59
	.uleb128 0xb
	.uleb128 0x1
	.uleb128 0x13
	.byte	0x0
	.byte	0x0
	.uleb128 0x1b
	.uleb128 0x5
	.byte	0x0
	.uleb128 0x31
	.uleb128 0x13
	.byte	0x0
	.byte	0x0
	.uleb128 0x1c
	.uleb128 0x1d
	.byte	0x1
	.uleb128 0x31
	.uleb128 0x13
	.uleb128 0x52
	.uleb128 0x1
	.uleb128 0x55
	.uleb128 0x6
	.uleb128 0x58
	.uleb128 0xb
	.uleb128 0x59
	.uleb128 0xb
	.byte	0x0
	.byte	0x0
	.uleb128 0x1d
	.uleb128 0x21
	.byte	0x0
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2f
	.uleb128 0x5
	.byte	0x0
	.byte	0x0
	.uleb128 0x1e
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0xc
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x27
	.uleb128 0xc
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x1
	.uleb128 0x40
	.uleb128 0x6
	.uleb128 0x1
	.uleb128 0x13
	.byte	0x0
	.byte	0x0
	.uleb128 0x1f
	.uleb128 0x5
	.byte	0x0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x6
	.byte	0x0
	.byte	0x0
	.uleb128 0x20
	.uleb128 0x34
	.byte	0x0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x6
	.byte	0x0
	.byte	0x0
	.uleb128 0x21
	.uleb128 0x34
	.byte	0x0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.byte	0x0
	.byte	0x0
	.uleb128 0x22
	.uleb128 0xa
	.byte	0x0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.byte	0x0
	.byte	0x0
	.uleb128 0x23
	.uleb128 0x1d
	.byte	0x1
	.uleb128 0x31
	.uleb128 0x13
	.uleb128 0x52
	.uleb128 0x1
	.uleb128 0x55
	.uleb128 0x6
	.uleb128 0x58
	.uleb128 0xb
	.uleb128 0x59
	.uleb128 0xb
	.uleb128 0x1
	.uleb128 0x13
	.byte	0x0
	.byte	0x0
	.uleb128 0x24
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x27
	.uleb128 0xc
	.uleb128 0x34
	.uleb128 0xc
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x1
	.uleb128 0x40
	.uleb128 0x6
	.uleb128 0x1
	.uleb128 0x13
	.byte	0x0
	.byte	0x0
	.uleb128 0x25
	.uleb128 0x5
	.byte	0x0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x34
	.uleb128 0xc
	.uleb128 0x2
	.uleb128 0x6
	.byte	0x0
	.byte	0x0
	.uleb128 0x26
	.uleb128 0x34
	.byte	0x0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0xa
	.byte	0x0
	.byte	0x0
	.uleb128 0x27
	.uleb128 0x34
	.byte	0x0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0xa
	.byte	0x0
	.byte	0x0
	.uleb128 0x28
	.uleb128 0xb
	.byte	0x1
	.uleb128 0x55
	.uleb128 0x6
	.byte	0x0
	.byte	0x0
	.uleb128 0x29
	.uleb128 0xb
	.byte	0x1
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x1
	.byte	0x0
	.byte	0x0
	.uleb128 0x2a
	.uleb128 0x13
	.byte	0x1
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x1
	.uleb128 0x13
	.byte	0x0
	.byte	0x0
	.uleb128 0x2b
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0xc
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x27
	.uleb128 0xc
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x1
	.uleb128 0x40
	.uleb128 0x6
	.uleb128 0x1
	.uleb128 0x13
	.byte	0x0
	.byte	0x0
	.uleb128 0x2c
	.uleb128 0x1d
	.byte	0x1
	.uleb128 0x31
	.uleb128 0x13
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x1
	.uleb128 0x58
	.uleb128 0xb
	.uleb128 0x59
	.uleb128 0xb
	.byte	0x0
	.byte	0x0
	.uleb128 0x2d
	.uleb128 0x34
	.byte	0x0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x3f
	.uleb128 0xc
	.uleb128 0x3c
	.uleb128 0xc
	.byte	0x0
	.byte	0x0
	.byte	0x0
	.section	.debug_pubnames,"",@progbits
	.long	0x3a
	.value	0x2
	.long	.Ldebug_info0
	.long	0x91f
	.long	0x3ed
	.string	"display"
	.long	0x484
	.string	"read_towns"
	.long	0x671
	.string	"run"
	.long	0x767
	.string	"main"
	.long	0x0
	.section	.debug_aranges,"",@progbits
	.long	0x2c
	.value	0x2
	.long	.Ldebug_info0
	.byte	0x8
	.byte	0x0
	.value	0x0
	.value	0x0
	.quad	.Ltext0
	.quad	.Letext0-.Ltext0
	.quad	0x0
	.quad	0x0
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.LBB29-.Ltext0
	.quad	.LBE29-.Ltext0
	.quad	.LBB32-.Ltext0
	.quad	.LBE32-.Ltext0
	.quad	0x0
	.quad	0x0
	.quad	.LBB33-.Ltext0
	.quad	.LBE33-.Ltext0
	.quad	.LBB40-.Ltext0
	.quad	.LBE40-.Ltext0
	.quad	.LBB39-.Ltext0
	.quad	.LBE39-.Ltext0
	.quad	.LBB38-.Ltext0
	.quad	.LBE38-.Ltext0
	.quad	0x0
	.quad	0x0
	.quad	.LBB41-.Ltext0
	.quad	.LBE41-.Ltext0
	.quad	.LBB44-.Ltext0
	.quad	.LBE44-.Ltext0
	.quad	0x0
	.quad	0x0
	.quad	.LBB45-.Ltext0
	.quad	.LBE45-.Ltext0
	.quad	.LBB48-.Ltext0
	.quad	.LBE48-.Ltext0
	.quad	.LBB47-.Ltext0
	.quad	.LBE47-.Ltext0
	.quad	0x0
	.quad	0x0
	.quad	.LBB49-.Ltext0
	.quad	.LBE49-.Ltext0
	.quad	.LBB54-.Ltext0
	.quad	.LBE54-.Ltext0
	.quad	.LBB53-.Ltext0
	.quad	.LBE53-.Ltext0
	.quad	0x0
	.quad	0x0
	.quad	.LBB58-.Ltext0
	.quad	.LBE58-.Ltext0
	.quad	.LBB61-.Ltext0
	.quad	.LBE61-.Ltext0
	.quad	0x0
	.quad	0x0
	.quad	.LBB70-.Ltext0
	.quad	.LBE70-.Ltext0
	.quad	.LBB75-.Ltext0
	.quad	.LBE75-.Ltext0
	.quad	.LBB74-.Ltext0
	.quad	.LBE74-.Ltext0
	.quad	0x0
	.quad	0x0
	.section	.debug_str,"MS",@progbits,1
.LASF56:
	.string	"printf"
.LASF8:
	.string	"__off_t"
.LASF13:
	.string	"_IO_read_ptr"
.LASF64:
	.string	"the_towns"
.LASF25:
	.string	"_chain"
.LASF7:
	.string	"size_t"
.LASF31:
	.string	"_shortbuf"
.LASF86:
	.string	"run.omp_fn.0"
.LASF51:
	.string	"longitude"
.LASF74:
	.string	"read_towns"
.LASF19:
	.string	"_IO_buf_base"
.LASF82:
	.string	"hyantes-openmp.c"
.LASF81:
	.string	"GNU C 4.4.5"
.LASF46:
	.string	"long long int"
.LASF4:
	.string	"signed char"
.LASF26:
	.string	"_fileno"
.LASF14:
	.string	"_IO_read_end"
.LASF6:
	.string	"long int"
.LASF12:
	.string	"_flags"
.LASF20:
	.string	"_IO_buf_end"
.LASF29:
	.string	"_cur_column"
.LASF48:
	.string	"double"
.LASF28:
	.string	"_old_offset"
.LASF33:
	.string	"_offset"
.LASF85:
	.string	"atof"
.LASF61:
	.string	"display"
.LASF42:
	.string	"_IO_marker"
.LASF78:
	.string	"stdin"
.LASF3:
	.string	"unsigned int"
.LASF57:
	.string	"fprintf"
.LASF59:
	.string	"__stream"
.LASF0:
	.string	"long unsigned int"
.LASF17:
	.string	"_IO_write_ptr"
.LASF44:
	.string	"_sbuf"
.LASF65:
	.string	"l99999"
.LASF54:
	.string	"data"
.LASF2:
	.string	"short unsigned int"
.LASF21:
	.string	"_IO_save_base"
.LASF52:
	.string	"stock"
.LASF32:
	.string	"_lock"
.LASF27:
	.string	"_flags2"
.LASF39:
	.string	"_mode"
.LASF62:
	.string	"fname"
.LASF50:
	.string	"latitude"
.LASF18:
	.string	"_IO_write_end"
.LASF53:
	.string	"town"
.LASF84:
	.string	"_IO_lock_t"
.LASF41:
	.string	"_IO_FILE"
.LASF60:
	.string	"__nptr"
.LASF68:
	.string	"range"
.LASF47:
	.string	"float"
.LASF45:
	.string	"_pos"
.LASF24:
	.string	"_markers"
.LASF55:
	.string	"towns"
.LASF1:
	.string	"unsigned char"
.LASF63:
	.string	"curr"
.LASF69:
	.string	"step"
.LASF5:
	.string	"short int"
.LASF30:
	.string	"_vtable_offset"
.LASF11:
	.string	"FILE"
.LASF83:
	.string	"/home/janice/par4all/examples/P4A/Hyantes"
.LASF87:
	.string	".omp_data_s.11"
.LASF73:
	.string	"ymax"
.LASF10:
	.string	"char"
.LASF79:
	.string	"stdout"
.LASF70:
	.string	"ymin"
.LASF67:
	.string	".omp_data_i"
.LASF43:
	.string	"_next"
.LASF9:
	.string	"__off64_t"
.LASF49:
	.string	"data_t"
.LASF15:
	.string	"_IO_read_base"
.LASF23:
	.string	"_IO_save_end"
.LASF58:
	.string	"__fmt"
.LASF34:
	.string	"__pad1"
.LASF35:
	.string	"__pad2"
.LASF36:
	.string	"__pad3"
.LASF37:
	.string	"__pad4"
.LASF38:
	.string	"__pad5"
.LASF40:
	.string	"_unused2"
.LASF80:
	.string	"stderr"
.LASF77:
	.string	"argv"
.LASF72:
	.string	"xmax"
.LASF71:
	.string	"xmin"
.LASF22:
	.string	"_IO_backup_base"
.LASF66:
	.string	"break_2"
.LASF76:
	.string	"argc"
.LASF75:
	.string	"main"
.LASF16:
	.string	"_IO_write_base"
	.ident	"GCC: (Ubuntu/Linaro 4.4.4-14ubuntu5) 4.4.5"
	.section	.note.GNU-stack,"",@progbits
