	.file	"hyantes-accel.c"
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
.LFB50:
	.file 1 "hyantes-accel.c"
	.loc 1 148 0
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
	.loc 1 150 0
	xorl	%ebx, %ebx
	.cfi_offset 3, -32
	.p2align 4,,10
	.p2align 3
.L3:
.LBB28:
.LBB29:
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
.LBE29:
.LBE28:
	.loc 1 151 0
	cmpq	$7176, %rbx
	jne	.L3
.LBB30:
.LBB31:
	.loc 2 105 0
	leaq	.LC1(%rip), %rsi
	xorl	%eax, %eax
	movl	$1, %edi
.LBE31:
.LBE30:
	.loc 1 150 0
	addq	$1, %r12
	addq	$7176, %rbp
.LBB33:
.LBB32:
	.loc 2 105 0
	call	__printf_chk@PLT
.LBE32:
.LBE33:
	.loc 1 150 0
	cmpq	$290, %r12
	jne	.L2
	.loc 1 155 0
	popq	%rbx
	popq	%rbp
	popq	%r12
.LVL3:
	ret
	.cfi_endproc
.LFE50:
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
.LFB48:
	.loc 1 86 0
	.cfi_startproc
.LVL4:
	pushq	%r15
.LCFI3:
	.cfi_def_cfa_offset 16
	.loc 1 87 0
	leaq	.LC2(%rip), %rsi
	.loc 1 86 0
	pushq	%r14
.LCFI4:
	.cfi_def_cfa_offset 24
.LBB34:
.LBB36:
	.loc 2 98 0
	movl	$1, %r14d
	.cfi_offset 14, -24
	.cfi_offset 15, -16
.LBE36:
.LBE34:
	.loc 1 86 0
	pushq	%r13
.LCFI5:
	.cfi_def_cfa_offset 32
	pushq	%r12
.LCFI6:
	.cfi_def_cfa_offset 40
.LBB39:
.LBB37:
	.loc 2 98 0
	xorl	%r12d, %r12d
	.cfi_offset 12, -40
	.cfi_offset 13, -32
.LVL5:
.LBE37:
.LBE39:
	.loc 1 86 0
	pushq	%rbp
.LCFI7:
	.cfi_def_cfa_offset 48
	pushq	%rbx
.LCFI8:
	.cfi_def_cfa_offset 56
	subq	$40, %rsp
.LCFI9:
	.cfi_def_cfa_offset 96
	.loc 1 87 0
	.cfi_offset 3, -56
	.cfi_offset 6, -48
	call	fopen@PLT
.LVL6:
	.loc 1 90 0
	movl	$24, %edi
	.loc 1 87 0
	movq	%rax, %rbx
.LVL7:
	.loc 1 90 0
	call	malloc@PLT
.LBB40:
.LBB35:
	.loc 2 98 0
	movq	stderr@GOTPCREL(%rip), %r15
	leaq	.LC3(%rip), %rdx
.LBE35:
.LBE40:
	.loc 1 90 0
	movq	%rax, %r13
.LBB41:
.LBB38:
	.loc 2 98 0
	movl	$1, %esi
	xorl	%eax, %eax
	movq	(%r15), %rdi
	call	__fprintf_chk@PLT
.LVL8:
	.p2align 4,,10
	.p2align 3
.L17:
.LBE38:
.LBE41:
	.loc 1 93 0
	movq	%rbx, %rdi
	call	feof@PLT
.LVL9:
	testl	%eax, %eax
	jne	.L19
.L13:
	.loc 1 94 0
	cmpq	%r14, %r12
	.p2align 4,,2
	je	.L20
.L10:
	.loc 1 99 0
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
	.loc 1 101 0
	movq	%rbx, %rdi
	call	feof@PLT
.LVL11:
	testl	%eax, %eax
	.p2align 4,,2
	jne	.L17
	.loc 1 102 0
	movq	%rbx, %rdi
	call	fgetc@PLT
	.loc 1 103 0
	cmpb	$13, %al
.LVL12:
	.p2align 4,,2
	je	.L17
	cmpb	$10, %al
	.p2align 4,,2
	jne	.L18
	.loc 1 93 0
	movq	%rbx, %rdi
	.p2align 4,,5
	call	feof@PLT
.LVL13:
	testl	%eax, %eax
	.p2align 4,,2
	je	.L13
.L19:
	.loc 1 116 0
	movq	%rbx, %rdi
	call	fclose@PLT
	.loc 1 117 0
	leaq	(%r12,%r12,2), %rsi
	movq	%r13, %rdi
	salq	$3, %rsi
	call	realloc@PLT
.LBB42:
.LBB43:
	.loc 2 98 0
	movq	(%r15), %rdi
	leaq	.LC6(%rip), %rdx
.LBE43:
.LBE42:
	.loc 1 117 0
	movq	%rax, %rbx
.LVL14:
.LBB45:
.LBB44:
	.loc 2 98 0
	movq	%r12, %rcx
	movl	$1, %esi
	xorl	%eax, %eax
	call	__fprintf_chk@PLT
.LBE44:
.LBE45:
	.loc 1 125 0
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
	.loc 1 111 0
	movsd	.LC5(%rip), %xmm0
	.loc 1 113 0
	addq	$1, %r12
	.loc 1 111 0
	mulsd	(%rbp), %xmm0
	movsd	%xmm0, (%rbp)
	.loc 1 112 0
	movsd	.LC5(%rip), %xmm0
	mulsd	8(%rbp), %xmm0
	movsd	%xmm0, 8(%rbp)
	jmp	.L17
	.p2align 4,,10
	.p2align 3
.L20:
	.loc 1 95 0
	leaq	(%r12,%r12), %r14
	.loc 1 97 0
	movq	%r13, %rdi
	leaq	(%r14,%r12,4), %rsi
	salq	$3, %rsi
	call	realloc@PLT
	movq	%rax, %r13
	jmp	.L10
	.cfi_endproc
.LFE48:
	.size	read_towns, .-read_towns
	.p2align 4,,15
.globl p4a_kernel_run
	.type	p4a_kernel_run, @function
p4a_kernel_run:
.LFB46:
	.loc 1 64 0
	.cfi_startproc
.LVL17:
	pushq	%r14
.LCFI10:
	.cfi_def_cfa_offset 16
	pushq	%r13
.LCFI11:
	.cfi_def_cfa_offset 24
	pushq	%r12
.LCFI12:
	.cfi_def_cfa_offset 32
	pushq	%rbp
.LCFI13:
	.cfi_def_cfa_offset 40
	movq	%rcx, %rbp
	.cfi_offset 6, -40
	.cfi_offset 12, -32
	.cfi_offset 13, -24
	.cfi_offset 14, -16
	pushq	%rbx
.LCFI14:
	.cfi_def_cfa_offset 48
	subq	$48, %rsp
.LCFI15:
	.cfi_def_cfa_offset 96
	.loc 1 68 0
	cmpq	$298, %rsi
	.loc 1 64 0
	movsd	%xmm0, 8(%rsp)
	.loc 1 68 0
	ja	.L30
	.cfi_offset 3, -48
.LVL18:
	cmpq	$289, %rdi
	ja	.L30
	.loc 1 69 0
	imulq	$7176, %rdi, %rax
	leaq	(%rsi,%rsi,2), %rcx
	testq	%rdi, %rdi
	leaq	(%rax,%rcx,8), %r12
	leaq	(%rdx,%r12), %r12
	js	.L23
	cvtsi2sdq	%rdi, %xmm0
.LVL19:
.L24:
	mulsd	%xmm1, %xmm0
	.loc 1 70 0
	testq	%rsi, %rsi
	.loc 1 69 0
	movsd	.LC7(%rip), %xmm5
	movsd	.LC8(%rip), %xmm4
	addsd	%xmm2, %xmm0
	movapd	%xmm0, %xmm2
.LVL20:
	mulsd	%xmm5, %xmm2
	divsd	%xmm4, %xmm2
	movsd	%xmm2, (%r12)
	.loc 1 70 0
	js	.L25
	cvtsi2sdq	%rsi, %xmm2
.L26:
	mulsd	%xmm1, %xmm2
	movapd	%xmm5, %xmm1
.LVL21:
	.loc 1 63 0
	leaq	40(%rsp), %r13
	leaq	32(%rsp), %r14
	xorl	%ebx, %ebx
	movq	%r14, %rsi
.LVL22:
	movq	%r13, %rdi
.LVL23:
	.loc 1 70 0
	addsd	%xmm2, %xmm3
.LVL24:
	mulsd	%xmm3, %xmm1
.LVL25:
	movsd	%xmm3, (%rsp)
.LVL26:
	.loc 1 71 0
	movq	$0, 16(%r12)
	.loc 1 70 0
	divsd	%xmm4, %xmm1
	movsd	%xmm1, 8(%r12)
	.loc 1 63 0
	call	sincos@PLT
.LVL27:
	movsd	32(%rsp), %xmm0
	movsd	40(%rsp), %xmm1
	movsd	%xmm0, 16(%rsp)
	movsd	%xmm1, 24(%rsp)
.LVL28:
	.p2align 4,,10
	.p2align 3
.L29:
	movsd	(%rbp,%rbx), %xmm0
.LVL29:
	movq	%r14, %rsi
	movq	%r13, %rdi
	call	sincos@PLT
.LBB46:
	.loc 1 73 0
	movsd	(%rsp), %xmm0
	subsd	8(%rbp,%rbx), %xmm0
	call	cos@PLT
	movapd	%xmm0, %xmm1
	movsd	16(%rsp), %xmm0
	mulsd	32(%rsp), %xmm0
	mulsd	%xmm1, %xmm0
	movsd	24(%rsp), %xmm1
	mulsd	40(%rsp), %xmm1
	addsd	%xmm1, %xmm0
	call	acos@PLT
	mulsd	.LC10(%rip), %xmm0
.LVL30:
	.loc 1 74 0
	movsd	8(%rsp), %xmm1
	ucomisd	%xmm0, %xmm1
	jbe	.L27
	.loc 1 75 0
	addsd	.LC11(%rip), %xmm0
.LVL31:
	movsd	16(%rbp,%rbx), %xmm1
	divsd	%xmm0, %xmm1
	addsd	16(%r12), %xmm1
	movsd	%xmm1, 16(%r12)
.LVL32:
.L27:
	addq	$24, %rbx
.LBE46:
	.loc 1 72 0
	cmpq	$69072, %rbx
	jne	.L29
.LVL33:
.L30:
	.loc 1 78 0
	addq	$48, %rsp
	popq	%rbx
	popq	%rbp
.LVL34:
	popq	%r12
	popq	%r13
	popq	%r14
	ret
.LVL35:
.L23:
	.loc 1 69 0
	movq	%rdi, %rax
	andl	$1, %edi
.LVL36:
	shrq	%rax
	orq	%rdi, %rax
	cvtsi2sdq	%rax, %xmm0
.LVL37:
	addsd	%xmm0, %xmm0
	jmp	.L24
.LVL38:
.L25:
	.loc 1 70 0
	movq	%rsi, %rax
	andl	$1, %esi
.LVL39:
	shrq	%rax
	orq	%rsi, %rax
	cvtsi2sdq	%rax, %xmm2
	addsd	%xmm2, %xmm2
	jmp	.L26
	.cfi_endproc
.LFE46:
	.size	p4a_kernel_run, .-p4a_kernel_run
	.p2align 4,,15
.globl p4a_wrapper_run
	.type	p4a_wrapper_run, @function
p4a_wrapper_run:
.LFB45:
	.loc 1 55 0
	.cfi_startproc
.LVL40:
	movq	%rbx, -16(%rsp)
	movq	%rbp, -8(%rsp)
	subq	$88, %rsp
.LCFI16:
	.cfi_def_cfa_offset 96
	.loc 1 55 0
	movq	%rdx, %rbx
	.cfi_offset 6, -16
	.cfi_offset 3, -24
	movq	%rcx, %rbp
	.loc 1 61 0
	movsd	%xmm0, 48(%rsp)
.LVL41:
	movsd	%xmm1, 32(%rsp)
.LVL42:
	movsd	%xmm2, 16(%rsp)
.LVL43:
	movsd	%xmm3, (%rsp)
.LVL44:
	.byte	0x66
	leaq	P4A_vp_coordinate@TLSGD(%rip), %rdi
	.value	0x6666
	rex64
	call	__tls_get_addr@PLT
.LVL45:
	movsd	(%rsp), %xmm3
	movq	%rbp, %rcx
	movslq	(%rax),%rsi
	movslq	4(%rax),%rdi
	movq	%rbx, %rdx
	movsd	16(%rsp), %xmm2
	.loc 1 62 0
	movq	72(%rsp), %rbx
.LVL46:
	.loc 1 61 0
	movsd	32(%rsp), %xmm1
	.loc 1 62 0
	movq	80(%rsp), %rbp
.LVL47:
	.loc 1 61 0
	movsd	48(%rsp), %xmm0
	.loc 1 62 0
	addq	$88, %rsp
	.loc 1 61 0
	jmp	p4a_kernel_run@PLT
	.cfi_endproc
.LFE45:
	.size	p4a_wrapper_run, .-p4a_wrapper_run
	.p2align 4,,15
	.type	p4a_launcher_run.omp_fn.0, @function
p4a_launcher_run.omp_fn.0:
.LFB52:
	.loc 1 83 0
	.cfi_startproc
.LVL48:
	pushq	%r13
.LCFI17:
	.cfi_def_cfa_offset 16
	pushq	%r12
.LCFI18:
	.cfi_def_cfa_offset 24
	pushq	%rbp
.LCFI19:
	.cfi_def_cfa_offset 32
	pushq	%rbx
.LCFI20:
	.cfi_def_cfa_offset 40
	movq	%rdi, %rbx
	.cfi_offset 3, -40
	.cfi_offset 6, -32
	.cfi_offset 12, -24
	.cfi_offset 13, -16
	subq	$8, %rsp
.LCFI21:
	.cfi_def_cfa_offset 48
	.loc 1 83 0
	call	omp_get_num_threads@PLT
.LVL49:
	movl	%eax, %ebp
	call	omp_get_thread_num@PLT
	movl	$290, %ecx
	movl	%eax, %r12d
	movl	%ecx, %edx
	movl	%ecx, %eax
	sarl	$31, %edx
	idivl	%ebp
	xorl	%edx, %edx
	imull	%eax, %ebp
	cmpl	$290, %ebp
	setne	%dl
	leal	(%rdx,%rax), %eax
	imull	%eax, %r12d
.LVL50:
	leal	(%r12,%rax), %r13d
	cmpl	$290, %r13d
	cmovg	%ecx, %r13d
	cmpl	%r13d, %r12d
	jge	.L40
.LVL51:
	.p2align 4,,10
	.p2align 3
.L41:
	xorl	%ebp, %ebp
.LVL52:
	.p2align 4,,10
	.p2align 3
.L37:
.LBB47:
.LBB48:
	.byte	0x66
	leaq	P4A_vp_coordinate@TLSGD(%rip), %rdi
	.value	0x6666
	rex64
	call	__tls_get_addr@PLT
	movl	%ebp, (%rax)
	movl	%r12d, 4(%rax)
	addl	$1, %ebp
.LVL53:
	movl	$0, 8(%rax)
	movq	24(%rbx), %rcx
	movq	56(%rbx), %rsi
	movq	48(%rbx), %rdi
	movq	(%rbx), %rdx
	movsd	8(%rbx), %xmm0
	movsd	40(%rbx), %xmm3
	movsd	32(%rbx), %xmm2
	movsd	16(%rbx), %xmm1
	call	p4a_wrapper_run@PLT
	cmpl	$299, %ebp
	jne	.L37
.LBE48:
.LBE47:
	addl	$1, %r12d
	cmpl	%r12d, %r13d
	jg	.L41
.L40:
	addq	$8, %rsp
	popq	%rbx
.LVL54:
	popq	%rbp
.LVL55:
	popq	%r12
.LVL56:
	popq	%r13
	ret
	.cfi_endproc
.LFE52:
	.size	p4a_launcher_run.omp_fn.0, .-p4a_launcher_run.omp_fn.0
	.p2align 4,,15
.globl p4a_launcher_run
	.type	p4a_launcher_run, @function
p4a_launcher_run:
.LFB47:
	.loc 1 80 0
	.cfi_startproc
.LVL57:
	pushq	%rbx
.LCFI22:
	.cfi_def_cfa_offset 16
	.loc 1 83 0
	xorl	%edx, %edx
	.loc 1 80 0
	subq	$64, %rsp
.LCFI23:
	.cfi_def_cfa_offset 80
	.loc 1 83 0
	movq	%rdi, (%rsp)
	leaq	p4a_launcher_run.omp_fn.0(%rip), %rdi
.LVL58:
	movq	%rsi, 24(%rsp)
	movq	%rsp, %rsi
.LVL59:
	movsd	%xmm0, 8(%rsp)
	movq	$0, 48(%rsp)
	movsd	%xmm1, 16(%rsp)
	movq	$0, 56(%rsp)
	movsd	%xmm2, 32(%rsp)
	movsd	%xmm3, 40(%rsp)
	.cfi_offset 3, -16
	call	GOMP_parallel_start@PLT
.LVL60:
	movq	%rsp, %rdi
	call	p4a_launcher_run.omp_fn.0
	call	GOMP_parallel_end@PLT
	.loc 1 84 0
	addq	$64, %rsp
	popq	%rbx
	ret
	.cfi_endproc
.LFE47:
	.size	p4a_launcher_run, .-p4a_launcher_run
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
.LFB49:
	.loc 1 127 0
	.cfi_startproc
.LVL61:
	pushq	%r12
.LCFI24:
	.cfi_def_cfa_offset 16
.LBB49:
.LBB51:
	.loc 2 98 0
	leaq	.LC12(%rip), %rdx
.LBE51:
.LBE49:
	.loc 1 127 0
	movq	%rsi, %r12
	.cfi_offset 12, -16
.LBB53:
.LBB50:
	.loc 2 98 0
	xorl	%eax, %eax
	movl	$1, %esi
.LVL62:
.LBE50:
.LBE53:
	.loc 1 127 0
	pushq	%rbp
.LCFI25:
	.cfi_def_cfa_offset 24
	pushq	%rbx
.LCFI26:
	.cfi_def_cfa_offset 32
	movq	%rdi, %rbx
	.cfi_offset 3, -32
	.cfi_offset 6, -24
	subq	$112, %rsp
.LCFI27:
	.cfi_def_cfa_offset 144
.LBB54:
.LBB52:
	.loc 2 98 0
	movq	stderr@GOTPCREL(%rip), %rbp
	movq	(%rbp), %rdi
.LVL63:
	movsd	%xmm1, 64(%rsp)
.LVL64:
	movsd	%xmm0, 80(%rsp)
.LVL65:
	movsd	%xmm4, 48(%rsp)
.LVL66:
	movsd	%xmm5, 32(%rsp)
.LVL67:
	call	__fprintf_chk@PLT
.LVL68:
.LBE52:
.LBE54:
.LBB55:
	.loc 1 134 0
	leaq	96(%rsp), %rdi
	movl	$2081040, %esi
	.loc 1 133 0
	movq	$0, 104(%rsp)
.LVL69:
	movq	$0, 96(%rsp)
.LVL70:
	.loc 1 134 0
	call	P4A_accel_malloc@PLT
	.loc 1 135 0
	leaq	104(%rsp), %rdi
	movl	$69072, %esi
	call	P4A_accel_malloc@PLT
	.loc 1 136 0
	movq	96(%rsp), %rax
	xorl	%r9d, %r9d
	movl	$299, %r8d
	movl	$290, %ecx
	movl	$299, %edx
	movl	$290, %esi
	movl	$24, %edi
	movq	%rbx, 8(%rsp)
	movq	$0, (%rsp)
	movq	%rax, 16(%rsp)
	call	P4A_copy_to_accel_2d@PLT
	.loc 1 137 0
	movq	104(%rsp), %r9
	movq	%r12, %r8
	xorl	%ecx, %ecx
	movl	$2878, %edx
	movl	$2878, %esi
	movl	$24, %edi
	call	P4A_copy_to_accel_1d@PLT
	.loc 1 140 0
	movsd	48(%rsp), %xmm4
	movq	104(%rsp), %rsi
	movsd	32(%rsp), %xmm5
	movq	96(%rsp), %rdi
	movsd	64(%rsp), %xmm1
	movsd	80(%rsp), %xmm2
	movapd	%xmm5, %xmm0
.LVL71:
	movapd	%xmm1, %xmm3
.LVL72:
	movapd	%xmm4, %xmm1
.LVL73:
	call	p4a_launcher_run@PLT
.LVL74:
	.loc 1 141 0
	movq	96(%rsp), %rax
	xorl	%r9d, %r9d
	movl	$299, %r8d
	movl	$290, %ecx
	movl	$299, %edx
	movl	$290, %esi
	movl	$24, %edi
	movq	%rbx, 8(%rsp)
	movq	$0, (%rsp)
	movq	%rax, 16(%rsp)
	call	P4A_copy_from_accel_2d@PLT
	.loc 1 142 0
	movq	96(%rsp), %rdi
	call	P4A_accel_free@PLT
	.loc 1 143 0
	movq	104(%rsp), %rdi
	call	P4A_accel_free@PLT
.LBE55:
.LBB56:
.LBB57:
	.loc 2 98 0
	movq	(%rbp), %rdi
	leaq	.LC13(%rip), %rdx
	movl	$1, %esi
	xorl	%eax, %eax
	call	__fprintf_chk@PLT
.LBE57:
.LBE56:
	.loc 1 146 0
	addq	$112, %rsp
	popq	%rbx
.LVL75:
	popq	%rbp
	popq	%r12
.LVL76:
	ret
	.cfi_endproc
.LFE49:
	.size	run, .-run
	.p2align 4,,15
.globl main
	.type	main, @function
main:
.LFB51:
	.loc 1 157 0
	.cfi_startproc
.LVL77:
	pushq	%rbx
.LCFI28:
	.cfi_def_cfa_offset 16
	.loc 1 159 0
	movl	$1, %eax
	.loc 1 157 0
	movq	%rsi, %rbx
	.cfi_offset 3, -16
	subq	$2081152, %rsp
.LCFI29:
	.cfi_def_cfa_offset 2081168
	.loc 1 159 0
	cmpl	$8, %edi
	je	.L53
.LVL78:
	.loc 1 174 0
	addq	$2081152, %rsp
	popq	%rbx
.LVL79:
	ret
.LVL80:
	.p2align 4,,10
	.p2align 3
.L53:
.LBB58:
	.loc 1 163 0
	movq	8(%rsi), %rdi
.LVL81:
	call	read_towns@PLT
	movq	%rax, 80(%rsp)
	movq	%rax, 2081136(%rsp)
.LBB59:
.LBB60:
	.file 3 "/usr/include/stdlib.h"
	.loc 3 281 0
	xorl	%esi, %esi
.LBE60:
.LBE59:
	.loc 1 163 0
	movq	%rdx, 88(%rsp)
	movq	%rdx, 2081144(%rsp)
.LBB62:
.LBB61:
	.loc 3 281 0
	movq	16(%rbx), %rdi
	call	strtod@PLT
	movsd	%xmm0, 72(%rsp)
.LBE61:
.LBE62:
.LBB63:
.LBB64:
	movq	24(%rbx), %rdi
	xorl	%esi, %esi
	call	strtod@PLT
.LBE64:
.LBE63:
.LBB65:
.LBB66:
	movq	32(%rbx), %rdi
	xorl	%esi, %esi
	movsd	%xmm0, (%rsp)
	call	strtod@PLT
.LBE66:
.LBE65:
.LBB67:
.LBB68:
	movq	40(%rbx), %rdi
	xorl	%esi, %esi
	movsd	%xmm0, 16(%rsp)
	call	strtod@PLT
.LBE68:
.LBE67:
.LBB69:
.LBB70:
	movq	48(%rbx), %rdi
	xorl	%esi, %esi
	movsd	%xmm0, 32(%rsp)
	call	strtod@PLT
.LBE70:
.LBE69:
.LBB71:
.LBB73:
	movq	56(%rbx), %rdi
	xorl	%esi, %esi
	movsd	%xmm0, 48(%rsp)
.LBE73:
.LBE71:
	.loc 1 170 0
	leaq	96(%rsp), %rbx
.LVL82:
.LBB75:
.LBB72:
	.loc 3 281 0
	call	strtod@PLT
.LBE72:
.LBE75:
	.loc 1 170 0
	movsd	48(%rsp), %xmm4
	movq	2081144(%rsp), %rsi
.LBB76:
.LBB74:
	.loc 3 281 0
	movapd	%xmm0, %xmm5
.LVL83:
.LBE74:
.LBE76:
	.loc 1 170 0
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
.LVL84:
	.loc 1 171 0
	movq	%rbx, %rdi
	call	display@PLT
	xorl	%eax, %eax
.LBE58:
	.loc 1 174 0
	addq	$2081152, %rsp
	popq	%rbx
	ret
	.cfi_endproc
.LFE51:
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
	.quad	.LFB50-.Ltext0
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
	.quad	.LFE50-.Ltext0
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
	.quad	.LFB48-.Ltext0
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
	.quad	.LFE48-.Ltext0
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
	.quad	.LFE48-.Ltext0
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
	.quad	.LFE48-.Ltext0
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
	.quad	.LFB46-.Ltext0
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
	.quad	.LFE46-.Ltext0
	.value	0x3
	.byte	0x77
	.sleb128 96
	.quad	0x0
	.quad	0x0
.LLST9:
	.quad	.LVL17-.Ltext0
	.quad	.LVL23-.Ltext0
	.value	0x1
	.byte	0x55
	.quad	.LVL33-.Ltext0
	.quad	.LVL36-.Ltext0
	.value	0x1
	.byte	0x55
	.quad	.LVL38-.Ltext0
	.quad	.LFE46-.Ltext0
	.value	0x1
	.byte	0x55
	.quad	0x0
	.quad	0x0
.LLST10:
	.quad	.LVL17-.Ltext0
	.quad	.LVL22-.Ltext0
	.value	0x1
	.byte	0x54
	.quad	.LVL33-.Ltext0
	.quad	.LVL39-.Ltext0
	.value	0x1
	.byte	0x54
	.quad	0x0
	.quad	0x0
.LLST11:
	.quad	.LVL17-.Ltext0
	.quad	.LVL27-.Ltext0
	.value	0x1
	.byte	0x51
	.quad	.LVL33-.Ltext0
	.quad	.LFE46-.Ltext0
	.value	0x1
	.byte	0x51
	.quad	0x0
	.quad	0x0
.LLST12:
	.quad	.LVL17-.Ltext0
	.quad	.LVL19-.Ltext0
	.value	0x1
	.byte	0x61
	.quad	.LVL33-.Ltext0
	.quad	.LVL37-.Ltext0
	.value	0x1
	.byte	0x61
	.quad	0x0
	.quad	0x0
.LLST13:
	.quad	.LVL17-.Ltext0
	.quad	.LVL21-.Ltext0
	.value	0x1
	.byte	0x62
	.quad	.LVL33-.Ltext0
	.quad	.LFE46-.Ltext0
	.value	0x1
	.byte	0x62
	.quad	0x0
	.quad	0x0
.LLST14:
	.quad	.LVL17-.Ltext0
	.quad	.LVL18-.Ltext0
	.value	0x1
	.byte	0x52
	.quad	.LVL18-.Ltext0
	.quad	.LVL34-.Ltext0
	.value	0x1
	.byte	0x56
	.quad	.LVL34-.Ltext0
	.quad	.LVL35-.Ltext0
	.value	0x1
	.byte	0x52
	.quad	.LVL35-.Ltext0
	.quad	.LFE46-.Ltext0
	.value	0x1
	.byte	0x56
	.quad	0x0
	.quad	0x0
.LLST15:
	.quad	.LVL17-.Ltext0
	.quad	.LVL20-.Ltext0
	.value	0x1
	.byte	0x63
	.quad	.LVL33-.Ltext0
	.quad	.LVL38-.Ltext0
	.value	0x1
	.byte	0x63
	.quad	0x0
	.quad	0x0
.LLST16:
	.quad	.LVL17-.Ltext0
	.quad	.LVL25-.Ltext0
	.value	0x1
	.byte	0x64
	.quad	.LVL26-.Ltext0
	.quad	.LVL27-.Ltext0
	.value	0x1
	.byte	0x64
	.quad	.LVL33-.Ltext0
	.quad	.LFE46-.Ltext0
	.value	0x1
	.byte	0x64
	.quad	0x0
	.quad	0x0
.LLST17:
	.quad	.LVL28-.Ltext0
	.quad	.LVL29-.Ltext0
	.value	0x1
	.byte	0x61
	.quad	.LVL30-.Ltext0
	.quad	.LVL31-.Ltext0
	.value	0x1
	.byte	0x61
	.quad	.LVL32-.Ltext0
	.quad	.LVL35-.Ltext0
	.value	0x1
	.byte	0x61
	.quad	0x0
	.quad	0x0
.LLST18:
	.quad	.LFB45-.Ltext0
	.quad	.LCFI16-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 8
	.quad	.LCFI16-.Ltext0
	.quad	.LFE45-.Ltext0
	.value	0x3
	.byte	0x77
	.sleb128 96
	.quad	0x0
	.quad	0x0
.LLST19:
	.quad	.LVL40-.Ltext0
	.quad	.LVL45-.Ltext0
	.value	0x1
	.byte	0x55
	.quad	0x0
	.quad	0x0
.LLST20:
	.quad	.LVL40-.Ltext0
	.quad	.LVL45-.Ltext0
	.value	0x1
	.byte	0x54
	.quad	0x0
	.quad	0x0
.LLST21:
	.quad	.LVL40-.Ltext0
	.quad	.LVL45-.Ltext0
	.value	0x1
	.byte	0x51
	.quad	.LVL45-.Ltext0
	.quad	.LVL46-.Ltext0
	.value	0x1
	.byte	0x53
	.quad	0x0
	.quad	0x0
.LLST22:
	.quad	.LVL40-.Ltext0
	.quad	.LVL41-.Ltext0
	.value	0x1
	.byte	0x61
	.quad	0x0
	.quad	0x0
.LLST23:
	.quad	.LVL40-.Ltext0
	.quad	.LVL42-.Ltext0
	.value	0x1
	.byte	0x62
	.quad	0x0
	.quad	0x0
.LLST24:
	.quad	.LVL40-.Ltext0
	.quad	.LVL45-.Ltext0
	.value	0x1
	.byte	0x52
	.quad	.LVL45-.Ltext0
	.quad	.LVL47-.Ltext0
	.value	0x1
	.byte	0x56
	.quad	0x0
	.quad	0x0
.LLST25:
	.quad	.LVL40-.Ltext0
	.quad	.LVL43-.Ltext0
	.value	0x1
	.byte	0x63
	.quad	0x0
	.quad	0x0
.LLST26:
	.quad	.LVL40-.Ltext0
	.quad	.LVL44-.Ltext0
	.value	0x1
	.byte	0x64
	.quad	0x0
	.quad	0x0
.LLST27:
	.quad	.LFB52-.Ltext0
	.quad	.LCFI17-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 8
	.quad	.LCFI17-.Ltext0
	.quad	.LCFI18-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 16
	.quad	.LCFI18-.Ltext0
	.quad	.LCFI19-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 24
	.quad	.LCFI19-.Ltext0
	.quad	.LCFI20-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 32
	.quad	.LCFI20-.Ltext0
	.quad	.LCFI21-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 40
	.quad	.LCFI21-.Ltext0
	.quad	.LFE52-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 48
	.quad	0x0
	.quad	0x0
.LLST28:
	.quad	.LVL48-.Ltext0
	.quad	.LVL49-.Ltext0
	.value	0x1
	.byte	0x55
	.quad	.LVL49-.Ltext0
	.quad	.LVL54-.Ltext0
	.value	0x1
	.byte	0x53
	.quad	0x0
	.quad	0x0
.LLST29:
	.quad	.LVL50-.Ltext0
	.quad	.LVL56-.Ltext0
	.value	0x1
	.byte	0x5c
	.quad	0x0
	.quad	0x0
.LLST30:
	.quad	.LVL51-.Ltext0
	.quad	.LVL55-.Ltext0
	.value	0x1
	.byte	0x56
	.quad	0x0
	.quad	0x0
.LLST31:
	.quad	.LVL51-.Ltext0
	.quad	.LVL55-.Ltext0
	.value	0x1
	.byte	0x56
	.quad	0x0
	.quad	0x0
.LLST32:
	.quad	.LFB47-.Ltext0
	.quad	.LCFI22-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 8
	.quad	.LCFI22-.Ltext0
	.quad	.LCFI23-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 16
	.quad	.LCFI23-.Ltext0
	.quad	.LFE47-.Ltext0
	.value	0x3
	.byte	0x77
	.sleb128 80
	.quad	0x0
	.quad	0x0
.LLST33:
	.quad	.LVL57-.Ltext0
	.quad	.LVL58-.Ltext0
	.value	0x1
	.byte	0x55
	.quad	0x0
	.quad	0x0
.LLST34:
	.quad	.LVL57-.Ltext0
	.quad	.LVL60-.Ltext0
	.value	0x1
	.byte	0x61
	.quad	0x0
	.quad	0x0
.LLST35:
	.quad	.LVL57-.Ltext0
	.quad	.LVL60-.Ltext0
	.value	0x1
	.byte	0x62
	.quad	0x0
	.quad	0x0
.LLST36:
	.quad	.LVL57-.Ltext0
	.quad	.LVL59-.Ltext0
	.value	0x1
	.byte	0x54
	.quad	0x0
	.quad	0x0
.LLST37:
	.quad	.LVL57-.Ltext0
	.quad	.LVL60-.Ltext0
	.value	0x1
	.byte	0x63
	.quad	0x0
	.quad	0x0
.LLST38:
	.quad	.LVL57-.Ltext0
	.quad	.LVL60-.Ltext0
	.value	0x1
	.byte	0x64
	.quad	0x0
	.quad	0x0
.LLST39:
	.quad	.LFB49-.Ltext0
	.quad	.LCFI24-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 8
	.quad	.LCFI24-.Ltext0
	.quad	.LCFI25-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 16
	.quad	.LCFI25-.Ltext0
	.quad	.LCFI26-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 24
	.quad	.LCFI26-.Ltext0
	.quad	.LCFI27-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 32
	.quad	.LCFI27-.Ltext0
	.quad	.LFE49-.Ltext0
	.value	0x3
	.byte	0x77
	.sleb128 144
	.quad	0x0
	.quad	0x0
.LLST40:
	.quad	.LVL61-.Ltext0
	.quad	.LVL65-.Ltext0
	.value	0x1
	.byte	0x61
	.quad	0x0
	.quad	0x0
.LLST41:
	.quad	.LVL61-.Ltext0
	.quad	.LVL64-.Ltext0
	.value	0x1
	.byte	0x62
	.quad	.LVL72-.Ltext0
	.quad	.LVL73-.Ltext0
	.value	0x1
	.byte	0x62
	.quad	0x0
	.quad	0x0
.LLST42:
	.quad	.LVL61-.Ltext0
	.quad	.LVL68-.Ltext0
	.value	0x1
	.byte	0x63
	.quad	0x0
	.quad	0x0
.LLST43:
	.quad	.LVL61-.Ltext0
	.quad	.LVL68-.Ltext0
	.value	0x1
	.byte	0x64
	.quad	0x0
	.quad	0x0
.LLST44:
	.quad	.LVL61-.Ltext0
	.quad	.LVL66-.Ltext0
	.value	0x1
	.byte	0x65
	.quad	.LVL73-.Ltext0
	.quad	.LVL74-.Ltext0
	.value	0x1
	.byte	0x65
	.quad	0x0
	.quad	0x0
.LLST45:
	.quad	.LVL61-.Ltext0
	.quad	.LVL67-.Ltext0
	.value	0x1
	.byte	0x66
	.quad	.LVL71-.Ltext0
	.quad	.LVL74-.Ltext0
	.value	0x1
	.byte	0x66
	.quad	0x0
	.quad	0x0
.LLST46:
	.quad	.LVL61-.Ltext0
	.quad	.LVL63-.Ltext0
	.value	0x1
	.byte	0x55
	.quad	.LVL63-.Ltext0
	.quad	.LVL75-.Ltext0
	.value	0x1
	.byte	0x53
	.quad	0x0
	.quad	0x0
.LLST47:
	.quad	.LVL61-.Ltext0
	.quad	.LVL62-.Ltext0
	.value	0x1
	.byte	0x54
	.quad	.LVL62-.Ltext0
	.quad	.LVL76-.Ltext0
	.value	0x1
	.byte	0x5c
	.quad	0x0
	.quad	0x0
.LLST48:
	.quad	.LFB51-.Ltext0
	.quad	.LCFI28-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 8
	.quad	.LCFI28-.Ltext0
	.quad	.LCFI29-.Ltext0
	.value	0x2
	.byte	0x77
	.sleb128 16
	.quad	.LCFI29-.Ltext0
	.quad	.LFE51-.Ltext0
	.value	0x5
	.byte	0x77
	.sleb128 2081168
	.quad	0x0
	.quad	0x0
.LLST49:
	.quad	.LVL77-.Ltext0
	.quad	.LVL81-.Ltext0
	.value	0x1
	.byte	0x55
	.quad	0x0
	.quad	0x0
.LLST50:
	.quad	.LVL77-.Ltext0
	.quad	.LVL78-.Ltext0
	.value	0x1
	.byte	0x54
	.quad	.LVL78-.Ltext0
	.quad	.LVL79-.Ltext0
	.value	0x1
	.byte	0x53
	.quad	.LVL79-.Ltext0
	.quad	.LVL80-.Ltext0
	.value	0x1
	.byte	0x54
	.quad	.LVL80-.Ltext0
	.quad	.LVL82-.Ltext0
	.value	0x1
	.byte	0x53
	.quad	0x0
	.quad	0x0
.LLST51:
	.quad	.LVL83-.Ltext0
	.quad	.LVL84-.Ltext0
	.value	0x1
	.byte	0x66
	.quad	0x0
	.quad	0x0
	.file 4 "/usr/lib/gcc/x86_64-linux-gnu/4.4.5/include/stddef.h"
	.file 5 "/usr/include/bits/types.h"
	.file 6 "/usr/include/stdio.h"
	.file 7 "/usr/include/libio.h"
	.file 8 "/home/janice/p4a/share/p4a_accel/p4a_accel-OpenMP.h"
	.section	.debug_info
	.long	0xba6
	.value	0x2
	.long	.Ldebug_abbrev0
	.byte	0x8
	.uleb128 0x1
	.long	.LASF89
	.byte	0x1
	.long	.LASF90
	.long	.LASF91
	.quad	.Ltext0
	.quad	.Letext0
	.long	.Ldebug_line0
	.uleb128 0x2
	.byte	0x8
	.byte	0x5
	.long	.LASF0
	.uleb128 0x3
	.long	.LASF7
	.byte	0x4
	.byte	0xd3
	.long	0x3f
	.uleb128 0x2
	.byte	0x8
	.byte	0x7
	.long	.LASF1
	.uleb128 0x4
	.byte	0x4
	.byte	0x5
	.string	"int"
	.uleb128 0x2
	.byte	0x1
	.byte	0x8
	.long	.LASF2
	.uleb128 0x2
	.byte	0x2
	.byte	0x7
	.long	.LASF3
	.uleb128 0x2
	.byte	0x4
	.byte	0x7
	.long	.LASF4
	.uleb128 0x2
	.byte	0x1
	.byte	0x6
	.long	.LASF5
	.uleb128 0x2
	.byte	0x2
	.byte	0x5
	.long	.LASF6
	.uleb128 0x3
	.long	.LASF8
	.byte	0x5
	.byte	0x8d
	.long	0x2d
	.uleb128 0x3
	.long	.LASF9
	.byte	0x5
	.byte	0x8e
	.long	0x2d
	.uleb128 0x5
	.byte	0x8
	.byte	0x7
	.uleb128 0x6
	.byte	0x8
	.uleb128 0x7
	.byte	0x8
	.long	0x91
	.uleb128 0x2
	.byte	0x1
	.byte	0x6
	.long	.LASF10
	.uleb128 0x3
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
	.long	0x46
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
	.long	0x46
	.byte	0x2
	.byte	0x23
	.uleb128 0x70
	.uleb128 0x9
	.long	.LASF27
	.byte	0x7
	.value	0x12a
	.long	0x46
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
	.long	0x54
	.byte	0x3
	.byte	0x23
	.uleb128 0x80
	.uleb128 0x9
	.long	.LASF30
	.byte	0x7
	.value	0x131
	.long	0x62
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
	.long	0x34
	.byte	0x3
	.byte	0x23
	.uleb128 0xb8
	.uleb128 0x9
	.long	.LASF39
	.byte	0x7
	.value	0x14e
	.long	0x46
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
	.long	.LASF92
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
	.long	0x46
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
	.uleb128 0x2
	.byte	0x8
	.byte	0x5
	.long	.LASF46
	.uleb128 0x2
	.byte	0x4
	.byte	0x4
	.long	.LASF47
	.uleb128 0x2
	.byte	0x8
	.byte	0x4
	.long	.LASF48
	.uleb128 0x3
	.long	.LASF49
	.byte	0x1
	.byte	0x1a
	.long	0x2f8
	.uleb128 0xf
	.byte	0x18
	.byte	0x1
	.byte	0x1b
	.long	0x33d
	.uleb128 0xb
	.long	.LASF50
	.byte	0x1
	.byte	0x1c
	.long	0x2ff
	.byte	0x2
	.byte	0x23
	.uleb128 0x0
	.uleb128 0xb
	.long	.LASF51
	.byte	0x1
	.byte	0x1d
	.long	0x2ff
	.byte	0x2
	.byte	0x23
	.uleb128 0x8
	.uleb128 0xb
	.long	.LASF52
	.byte	0x1
	.byte	0x1e
	.long	0x2ff
	.byte	0x2
	.byte	0x23
	.uleb128 0x10
	.byte	0x0
	.uleb128 0x3
	.long	.LASF53
	.byte	0x1
	.byte	0x1f
	.long	0x30a
	.uleb128 0xf
	.byte	0x10
	.byte	0x1
	.byte	0x20
	.long	0x36b
	.uleb128 0x10
	.string	"n"
	.byte	0x1
	.byte	0x21
	.long	0x34
	.byte	0x2
	.byte	0x23
	.uleb128 0x0
	.uleb128 0xb
	.long	.LASF54
	.byte	0x1
	.byte	0x22
	.long	0x36b
	.byte	0x2
	.byte	0x23
	.uleb128 0x8
	.byte	0x0
	.uleb128 0x7
	.byte	0x8
	.long	0x33d
	.uleb128 0x3
	.long	.LASF55
	.byte	0x1
	.byte	0x23
	.long	0x348
	.uleb128 0x11
	.byte	0x1
	.long	.LASF56
	.byte	0x2
	.byte	0x67
	.byte	0x1
	.long	0x46
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
	.long	0x46
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
	.long	.LASF93
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
	.byte	0x93
	.byte	0x1
	.quad	.LFB50
	.quad	.LFE50
	.long	.LLST0
	.long	0x46d
	.uleb128 0x17
	.string	"pt"
	.byte	0x1
	.byte	0x93
	.long	0x47e
	.long	.LLST1
	.uleb128 0x18
	.string	"i"
	.byte	0x1
	.byte	0x95
	.long	0x34
	.long	.LLST2
	.uleb128 0x19
	.string	"j"
	.byte	0x1
	.byte	0x95
	.long	0x34
	.uleb128 0x1a
	.long	0x37c
	.quad	.LBB28
	.quad	.LBE28
	.byte	0x1
	.byte	0x98
	.long	0x453
	.uleb128 0x1b
	.long	0x38f
	.byte	0x0
	.uleb128 0x1c
	.long	0x37c
	.quad	.LBB30
	.long	.Ldebug_ranges0+0x0
	.byte	0x1
	.byte	0x99
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
	.long	.LASF81
	.byte	0x1
	.byte	0x55
	.byte	0x1
	.long	0x371
	.quad	.LFB48
	.quad	.LFE48
	.long	.LLST3
	.long	0x53c
	.uleb128 0x1f
	.long	.LASF62
	.byte	0x1
	.byte	0x55
	.long	0x2df
	.long	.LLST4
	.uleb128 0x18
	.string	"fd"
	.byte	0x1
	.byte	0x57
	.long	0x3c7
	.long	.LLST5
	.uleb128 0x20
	.long	.LASF63
	.byte	0x1
	.byte	0x58
	.long	0x34
	.long	.LLST6
	.uleb128 0x18
	.string	"c"
	.byte	0x1
	.byte	0x59
	.long	0x91
	.long	.LLST7
	.uleb128 0x21
	.long	.LASF64
	.byte	0x1
	.byte	0x5a
	.long	0x371
	.uleb128 0x22
	.long	.LASF65
	.byte	0x1
	.byte	0x6a
	.uleb128 0x22
	.long	.LASF66
	.byte	0x1
	.byte	0x6c
	.uleb128 0x23
	.long	0x39c
	.quad	.LBB34
	.long	.Ldebug_ranges0+0x30
	.byte	0x1
	.byte	0x5b
	.long	0x51d
	.uleb128 0x1b
	.long	0x3ba
	.uleb128 0x1b
	.long	0x3af
	.byte	0x0
	.uleb128 0x1c
	.long	0x39c
	.quad	.LBB42
	.long	.Ldebug_ranges0+0x80
	.byte	0x1
	.byte	0x77
	.uleb128 0x1b
	.long	0x3ba
	.uleb128 0x1b
	.long	0x3af
	.byte	0x0
	.byte	0x0
	.uleb128 0x16
	.byte	0x1
	.long	.LASF67
	.byte	0x1
	.byte	0x3f
	.byte	0x1
	.quad	.LFB46
	.quad	.LFE46
	.long	.LLST8
	.long	0x5f9
	.uleb128 0x17
	.string	"i"
	.byte	0x1
	.byte	0x3f
	.long	0x34
	.long	.LLST9
	.uleb128 0x17
	.string	"j"
	.byte	0x1
	.byte	0x3f
	.long	0x34
	.long	.LLST10
	.uleb128 0x17
	.string	"pt"
	.byte	0x1
	.byte	0x3f
	.long	0x36b
	.long	.LLST11
	.uleb128 0x1f
	.long	.LASF68
	.byte	0x1
	.byte	0x3f
	.long	0x2ff
	.long	.LLST12
	.uleb128 0x1f
	.long	.LASF69
	.byte	0x1
	.byte	0x3f
	.long	0x2ff
	.long	.LLST13
	.uleb128 0x17
	.string	"t"
	.byte	0x1
	.byte	0x3f
	.long	0x36b
	.long	.LLST14
	.uleb128 0x1f
	.long	.LASF70
	.byte	0x1
	.byte	0x3f
	.long	0x2ff
	.long	.LLST15
	.uleb128 0x1f
	.long	.LASF71
	.byte	0x1
	.byte	0x3f
	.long	0x2ff
	.long	.LLST16
	.uleb128 0x19
	.string	"k"
	.byte	0x1
	.byte	0x42
	.long	0x34
	.uleb128 0x24
	.quad	.LBB46
	.quad	.LBE46
	.uleb128 0x18
	.string	"tmp"
	.byte	0x1
	.byte	0x49
	.long	0x2ff
	.long	.LLST17
	.byte	0x0
	.byte	0x0
	.uleb128 0x16
	.byte	0x1
	.long	.LASF72
	.byte	0x1
	.byte	0x36
	.byte	0x1
	.quad	.LFB45
	.quad	.LFE45
	.long	.LLST18
	.long	0x68c
	.uleb128 0x17
	.string	"i"
	.byte	0x1
	.byte	0x36
	.long	0x34
	.long	.LLST19
	.uleb128 0x17
	.string	"j"
	.byte	0x1
	.byte	0x36
	.long	0x34
	.long	.LLST20
	.uleb128 0x17
	.string	"pt"
	.byte	0x1
	.byte	0x36
	.long	0x36b
	.long	.LLST21
	.uleb128 0x1f
	.long	.LASF68
	.byte	0x1
	.byte	0x36
	.long	0x2ff
	.long	.LLST22
	.uleb128 0x1f
	.long	.LASF69
	.byte	0x1
	.byte	0x36
	.long	0x2ff
	.long	.LLST23
	.uleb128 0x17
	.string	"t"
	.byte	0x1
	.byte	0x36
	.long	0x36b
	.long	.LLST24
	.uleb128 0x1f
	.long	.LASF70
	.byte	0x1
	.byte	0x36
	.long	0x2ff
	.long	.LLST25
	.uleb128 0x1f
	.long	.LASF71
	.byte	0x1
	.byte	0x36
	.long	0x2ff
	.long	.LLST26
	.byte	0x0
	.uleb128 0x25
	.long	.LASF94
	.byte	0x1
	.byte	0x1
	.quad	.LFB52
	.quad	.LFE52
	.long	.LLST27
	.long	0x782
	.uleb128 0x26
	.long	.LASF73
	.long	0x7f6
	.byte	0x1
	.long	.LLST28
	.uleb128 0x20
	.long	.LASF74
	.byte	0x1
	.byte	0x53
	.long	0x46
	.long	.LLST29
	.uleb128 0x27
	.string	"j"
	.byte	0x1
	.byte	0x52
	.long	0x34
	.byte	0x4
	.byte	0x73
	.sleb128 0
	.byte	0x23
	.uleb128 0x38
	.uleb128 0x27
	.string	"i"
	.byte	0x1
	.byte	0x52
	.long	0x34
	.byte	0x4
	.byte	0x73
	.sleb128 0
	.byte	0x23
	.uleb128 0x30
	.uleb128 0x28
	.long	.LASF71
	.byte	0x1
	.byte	0x4f
	.long	0x2ff
	.byte	0x4
	.byte	0x73
	.sleb128 0
	.byte	0x23
	.uleb128 0x28
	.uleb128 0x28
	.long	.LASF70
	.byte	0x1
	.byte	0x4f
	.long	0x2ff
	.byte	0x4
	.byte	0x73
	.sleb128 0
	.byte	0x23
	.uleb128 0x20
	.uleb128 0x27
	.string	"t"
	.byte	0x1
	.byte	0x4f
	.long	0x36b
	.byte	0x4
	.byte	0x73
	.sleb128 0
	.byte	0x23
	.uleb128 0x18
	.uleb128 0x28
	.long	.LASF69
	.byte	0x1
	.byte	0x4f
	.long	0x2ff
	.byte	0x4
	.byte	0x73
	.sleb128 0
	.byte	0x23
	.uleb128 0x10
	.uleb128 0x28
	.long	.LASF68
	.byte	0x1
	.byte	0x4f
	.long	0x2ff
	.byte	0x4
	.byte	0x73
	.sleb128 0
	.byte	0x23
	.uleb128 0x8
	.uleb128 0x27
	.string	"pt"
	.byte	0x1
	.byte	0x4f
	.long	0x47e
	.byte	0x2
	.byte	0x73
	.sleb128 0
	.uleb128 0x24
	.quad	.LBB47
	.quad	.LBE47
	.uleb128 0x20
	.long	.LASF75
	.byte	0x1
	.byte	0x53
	.long	0x46
	.long	.LLST30
	.uleb128 0x24
	.quad	.LBB48
	.quad	.LBE48
	.uleb128 0x20
	.long	.LASF75
	.byte	0x1
	.byte	0x53
	.long	0x46
	.long	.LLST31
	.byte	0x0
	.byte	0x0
	.byte	0x0
	.uleb128 0x29
	.long	.LASF95
	.byte	0x40
	.long	0x7f6
	.uleb128 0x10
	.string	"pt"
	.byte	0x1
	.byte	0x53
	.long	0x47e
	.byte	0x2
	.byte	0x23
	.uleb128 0x0
	.uleb128 0xb
	.long	.LASF68
	.byte	0x1
	.byte	0x53
	.long	0x2ff
	.byte	0x2
	.byte	0x23
	.uleb128 0x8
	.uleb128 0xb
	.long	.LASF69
	.byte	0x1
	.byte	0x53
	.long	0x2ff
	.byte	0x2
	.byte	0x23
	.uleb128 0x10
	.uleb128 0x10
	.string	"t"
	.byte	0x1
	.byte	0x53
	.long	0x36b
	.byte	0x2
	.byte	0x23
	.uleb128 0x18
	.uleb128 0xb
	.long	.LASF70
	.byte	0x1
	.byte	0x53
	.long	0x2ff
	.byte	0x2
	.byte	0x23
	.uleb128 0x20
	.uleb128 0xb
	.long	.LASF71
	.byte	0x1
	.byte	0x53
	.long	0x2ff
	.byte	0x2
	.byte	0x23
	.uleb128 0x28
	.uleb128 0x10
	.string	"i"
	.byte	0x1
	.byte	0x53
	.long	0x34
	.byte	0x2
	.byte	0x23
	.uleb128 0x30
	.uleb128 0x10
	.string	"j"
	.byte	0x1
	.byte	0x53
	.long	0x34
	.byte	0x2
	.byte	0x23
	.uleb128 0x38
	.byte	0x0
	.uleb128 0x7
	.byte	0x8
	.long	0x782
	.uleb128 0x16
	.byte	0x1
	.long	.LASF76
	.byte	0x1
	.byte	0x4f
	.byte	0x1
	.quad	.LFB47
	.quad	.LFE47
	.long	.LLST32
	.long	0x890
	.uleb128 0x17
	.string	"pt"
	.byte	0x1
	.byte	0x4f
	.long	0x47e
	.long	.LLST33
	.uleb128 0x1f
	.long	.LASF68
	.byte	0x1
	.byte	0x4f
	.long	0x2ff
	.long	.LLST34
	.uleb128 0x1f
	.long	.LASF69
	.byte	0x1
	.byte	0x4f
	.long	0x2ff
	.long	.LLST35
	.uleb128 0x17
	.string	"t"
	.byte	0x1
	.byte	0x4f
	.long	0x36b
	.long	.LLST36
	.uleb128 0x1f
	.long	.LASF70
	.byte	0x1
	.byte	0x4f
	.long	0x2ff
	.long	.LLST37
	.uleb128 0x1f
	.long	.LASF71
	.byte	0x1
	.byte	0x4f
	.long	0x2ff
	.long	.LLST38
	.uleb128 0x19
	.string	"i"
	.byte	0x1
	.byte	0x52
	.long	0x34
	.uleb128 0x19
	.string	"j"
	.byte	0x1
	.byte	0x52
	.long	0x34
	.uleb128 0x19
	.string	"k"
	.byte	0x1
	.byte	0x52
	.long	0x34
	.byte	0x0
	.uleb128 0x2a
	.byte	0x1
	.string	"run"
	.byte	0x1
	.byte	0x7e
	.byte	0x1
	.quad	.LFB49
	.quad	.LFE49
	.long	.LLST39
	.long	0x9b8
	.uleb128 0x1f
	.long	.LASF70
	.byte	0x1
	.byte	0x7e
	.long	0x2ff
	.long	.LLST40
	.uleb128 0x1f
	.long	.LASF71
	.byte	0x1
	.byte	0x7e
	.long	0x2ff
	.long	.LLST41
	.uleb128 0x1f
	.long	.LASF77
	.byte	0x1
	.byte	0x7e
	.long	0x2ff
	.long	.LLST42
	.uleb128 0x1f
	.long	.LASF78
	.byte	0x1
	.byte	0x7e
	.long	0x2ff
	.long	.LLST43
	.uleb128 0x1f
	.long	.LASF69
	.byte	0x1
	.byte	0x7e
	.long	0x2ff
	.long	.LLST44
	.uleb128 0x1f
	.long	.LASF68
	.byte	0x1
	.byte	0x7e
	.long	0x2ff
	.long	.LLST45
	.uleb128 0x17
	.string	"pt"
	.byte	0x1
	.byte	0x7e
	.long	0x47e
	.long	.LLST46
	.uleb128 0x17
	.string	"t"
	.byte	0x1
	.byte	0x7e
	.long	0x36b
	.long	.LLST47
	.uleb128 0x19
	.string	"i"
	.byte	0x1
	.byte	0x80
	.long	0x34
	.uleb128 0x19
	.string	"j"
	.byte	0x1
	.byte	0x80
	.long	0x34
	.uleb128 0x19
	.string	"k"
	.byte	0x1
	.byte	0x80
	.long	0x34
	.uleb128 0x23
	.long	0x39c
	.quad	.LBB49
	.long	.Ldebug_ranges0+0xb0
	.byte	0x1
	.byte	0x82
	.long	0x963
	.uleb128 0x1b
	.long	0x3ba
	.uleb128 0x1b
	.long	0x3af
	.byte	0x0
	.uleb128 0x2b
	.quad	.LBB55
	.quad	.LBE55
	.long	0x995
	.uleb128 0x28
	.long	.LASF79
	.byte	0x1
	.byte	0x85
	.long	0x9c9
	.byte	0x2
	.byte	0x91
	.sleb128 -40
	.uleb128 0x28
	.long	.LASF80
	.byte	0x1
	.byte	0x85
	.long	0x9e7
	.byte	0x2
	.byte	0x91
	.sleb128 -48
	.byte	0x0
	.uleb128 0x2c
	.long	0x39c
	.quad	.LBB56
	.quad	.LBE56
	.byte	0x1
	.byte	0x91
	.uleb128 0x1b
	.long	0x3ba
	.uleb128 0x1b
	.long	0x3af
	.byte	0x0
	.byte	0x0
	.uleb128 0xc
	.long	0x33d
	.long	0x9c9
	.uleb128 0x1d
	.long	0x86
	.value	0xb3d
	.byte	0x0
	.uleb128 0x7
	.byte	0x8
	.long	0x9b8
	.uleb128 0xc
	.long	0x33d
	.long	0x9e7
	.uleb128 0x1d
	.long	0x86
	.value	0x121
	.uleb128 0x1d
	.long	0x86
	.value	0x12a
	.byte	0x0
	.uleb128 0x7
	.byte	0x8
	.long	0x9cf
	.uleb128 0x1e
	.byte	0x1
	.long	.LASF82
	.byte	0x1
	.byte	0x9c
	.byte	0x1
	.long	0x46
	.quad	.LFB51
	.quad	.LFE51
	.long	.LLST48
	.long	0xb5f
	.uleb128 0x1f
	.long	.LASF83
	.byte	0x1
	.byte	0x9c
	.long	0x46
	.long	.LLST49
	.uleb128 0x1f
	.long	.LASF84
	.byte	0x1
	.byte	0x9c
	.long	0xb5f
	.long	.LLST50
	.uleb128 0x24
	.quad	.LBB58
	.quad	.LBE58
	.uleb128 0x27
	.string	"pt"
	.byte	0x1
	.byte	0xa2
	.long	0x9cf
	.byte	0x5
	.byte	0x91
	.sleb128 -2081072
	.uleb128 0x27
	.string	"t"
	.byte	0x1
	.byte	0xa3
	.long	0x371
	.byte	0x2
	.byte	0x91
	.sleb128 -32
	.uleb128 0x21
	.long	.LASF70
	.byte	0x1
	.byte	0xa9
	.long	0x2ff
	.uleb128 0x21
	.long	.LASF71
	.byte	0x1
	.byte	0xa9
	.long	0x2ff
	.uleb128 0x21
	.long	.LASF77
	.byte	0x1
	.byte	0xa9
	.long	0x2ff
	.uleb128 0x21
	.long	.LASF78
	.byte	0x1
	.byte	0xa9
	.long	0x2ff
	.uleb128 0x21
	.long	.LASF69
	.byte	0x1
	.byte	0xa9
	.long	0x2ff
	.uleb128 0x20
	.long	.LASF68
	.byte	0x1
	.byte	0xa9
	.long	0x2ff
	.long	.LLST51
	.uleb128 0x23
	.long	0x3cd
	.quad	.LBB59
	.long	.Ldebug_ranges0+0xf0
	.byte	0x1
	.byte	0xa9
	.long	0xac0
	.uleb128 0x1b
	.long	0x3e0
	.byte	0x0
	.uleb128 0x1a
	.long	0x3cd
	.quad	.LBB63
	.quad	.LBE63
	.byte	0x1
	.byte	0xa9
	.long	0xae1
	.uleb128 0x1b
	.long	0x3e0
	.byte	0x0
	.uleb128 0x1a
	.long	0x3cd
	.quad	.LBB65
	.quad	.LBE65
	.byte	0x1
	.byte	0xa9
	.long	0xb02
	.uleb128 0x1b
	.long	0x3e0
	.byte	0x0
	.uleb128 0x1a
	.long	0x3cd
	.quad	.LBB67
	.quad	.LBE67
	.byte	0x1
	.byte	0xa9
	.long	0xb23
	.uleb128 0x1b
	.long	0x3e0
	.byte	0x0
	.uleb128 0x1a
	.long	0x3cd
	.quad	.LBB69
	.quad	.LBE69
	.byte	0x1
	.byte	0xa9
	.long	0xb44
	.uleb128 0x1b
	.long	0x3e0
	.byte	0x0
	.uleb128 0x1c
	.long	0x3cd
	.quad	.LBB71
	.long	.Ldebug_ranges0+0x120
	.byte	0x1
	.byte	0xa9
	.uleb128 0x1b
	.long	0x3e0
	.byte	0x0
	.byte	0x0
	.byte	0x0
	.uleb128 0x7
	.byte	0x8
	.long	0x8b
	.uleb128 0x2d
	.long	.LASF85
	.byte	0x6
	.byte	0xa5
	.long	0x2b3
	.byte	0x1
	.byte	0x1
	.uleb128 0x2d
	.long	.LASF86
	.byte	0x6
	.byte	0xa6
	.long	0x2b3
	.byte	0x1
	.byte	0x1
	.uleb128 0x2d
	.long	.LASF87
	.byte	0x6
	.byte	0xa7
	.long	0x2b3
	.byte	0x1
	.byte	0x1
	.uleb128 0xc
	.long	0x46
	.long	0xb9c
	.uleb128 0xd
	.long	0x86
	.byte	0x2
	.byte	0x0
	.uleb128 0x2d
	.long	.LASF88
	.byte	0x8
	.byte	0x21
	.long	0xb8c
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
	.uleb128 0x3
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
	.uleb128 0xb
	.byte	0x1
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x1
	.byte	0x0
	.byte	0x0
	.uleb128 0x25
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
	.uleb128 0x26
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
	.uleb128 0x27
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
	.uleb128 0x28
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
	.uleb128 0x29
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
	.uleb128 0x2a
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
	.uleb128 0x2b
	.uleb128 0xb
	.byte	0x1
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x1
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
	.long	0x76
	.value	0x2
	.long	.Ldebug_info0
	.long	0xbaa
	.long	0x3ed
	.string	"display"
	.long	0x484
	.string	"read_towns"
	.long	0x53c
	.string	"p4a_kernel_run"
	.long	0x5f9
	.string	"p4a_wrapper_run"
	.long	0x7fc
	.string	"p4a_launcher_run"
	.long	0x890
	.string	"run"
	.long	0x9ed
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
	.quad	.LBB30-.Ltext0
	.quad	.LBE30-.Ltext0
	.quad	.LBB33-.Ltext0
	.quad	.LBE33-.Ltext0
	.quad	0x0
	.quad	0x0
	.quad	.LBB34-.Ltext0
	.quad	.LBE34-.Ltext0
	.quad	.LBB41-.Ltext0
	.quad	.LBE41-.Ltext0
	.quad	.LBB40-.Ltext0
	.quad	.LBE40-.Ltext0
	.quad	.LBB39-.Ltext0
	.quad	.LBE39-.Ltext0
	.quad	0x0
	.quad	0x0
	.quad	.LBB42-.Ltext0
	.quad	.LBE42-.Ltext0
	.quad	.LBB45-.Ltext0
	.quad	.LBE45-.Ltext0
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
	.quad	.LBB59-.Ltext0
	.quad	.LBE59-.Ltext0
	.quad	.LBB62-.Ltext0
	.quad	.LBE62-.Ltext0
	.quad	0x0
	.quad	0x0
	.quad	.LBB71-.Ltext0
	.quad	.LBE71-.Ltext0
	.quad	.LBB76-.Ltext0
	.quad	.LBE76-.Ltext0
	.quad	.LBB75-.Ltext0
	.quad	.LBE75-.Ltext0
	.quad	0x0
	.quad	0x0
	.section	.debug_str,"MS",@progbits,1
.LASF56:
	.string	"printf"
.LASF8:
	.string	"__off_t"
.LASF72:
	.string	"p4a_wrapper_run"
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
.LASF51:
	.string	"longitude"
.LASF79:
	.string	"P4A_var_t0"
.LASF81:
	.string	"read_towns"
.LASF19:
	.string	"_IO_buf_base"
.LASF90:
	.string	"hyantes-accel.c"
.LASF89:
	.string	"GNU C 4.4.5"
.LASF46:
	.string	"long long int"
.LASF5:
	.string	"signed char"
.LASF76:
	.string	"p4a_launcher_run"
.LASF26:
	.string	"_fileno"
.LASF14:
	.string	"_IO_read_end"
.LASF0:
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
.LASF93:
	.string	"atof"
.LASF61:
	.string	"display"
.LASF80:
	.string	"P4A_var_pt0"
.LASF42:
	.string	"_IO_marker"
.LASF85:
	.string	"stdin"
.LASF4:
	.string	"unsigned int"
.LASF57:
	.string	"fprintf"
.LASF59:
	.string	"__stream"
.LASF95:
	.string	".omp_data_s.20"
.LASF1:
	.string	"long unsigned int"
.LASF17:
	.string	"_IO_write_ptr"
.LASF74:
	.string	"P4A_index_1"
.LASF44:
	.string	"_sbuf"
.LASF65:
	.string	"l99999"
.LASF54:
	.string	"data"
.LASF3:
	.string	"short unsigned int"
.LASF21:
	.string	"_IO_save_base"
.LASF52:
	.string	"stock"
.LASF75:
	.string	"P4A_index_0"
.LASF32:
	.string	"_lock"
.LASF27:
	.string	"_flags2"
.LASF39:
	.string	"_mode"
.LASF62:
	.string	"fname"
.LASF88:
	.string	"P4A_vp_coordinate"
.LASF50:
	.string	"latitude"
.LASF18:
	.string	"_IO_write_end"
.LASF53:
	.string	"town"
.LASF92:
	.string	"_IO_lock_t"
.LASF41:
	.string	"_IO_FILE"
.LASF60:
	.string	"__nptr"
.LASF68:
	.string	"range"
.LASF47:
	.string	"float"
.LASF94:
	.string	"p4a_launcher_run.omp_fn.0"
.LASF45:
	.string	"_pos"
.LASF24:
	.string	"_markers"
.LASF55:
	.string	"towns"
.LASF2:
	.string	"unsigned char"
.LASF63:
	.string	"curr"
.LASF69:
	.string	"step"
.LASF6:
	.string	"short int"
.LASF30:
	.string	"_vtable_offset"
.LASF11:
	.string	"FILE"
.LASF91:
	.string	"/home/janice/par4all/examples/P4A/Hyantes"
.LASF67:
	.string	"p4a_kernel_run"
.LASF78:
	.string	"ymax"
.LASF10:
	.string	"char"
.LASF86:
	.string	"stdout"
.LASF71:
	.string	"ymin"
.LASF73:
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
.LASF87:
	.string	"stderr"
.LASF84:
	.string	"argv"
.LASF77:
	.string	"xmax"
.LASF70:
	.string	"xmin"
.LASF22:
	.string	"_IO_backup_base"
.LASF66:
	.string	"break_2"
.LASF83:
	.string	"argc"
.LASF82:
	.string	"main"
.LASF16:
	.string	"_IO_write_base"
	.ident	"GCC: (Ubuntu/Linaro 4.4.4-14ubuntu5) 4.4.5"
	.section	.note.GNU-stack,"",@progbits
