	.file	"biquad_one_section.c"
	.text
	.p2align 4,,15
.globl pin_down
	.type	pin_down, @function
pin_down:
	pushl	%ebp
	flds	.LC0
	movl	%esp, %ebp
	popl	%ebp
	ret
	.size	pin_down, .-pin_down
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC2:
	.string	"%llu\n"
	.text
	.p2align 4,,15
.globl main
	.type	main, @function
main:
	pushl	%ebp
	movl	%esp, %ebp
	andl	$-16, %esp
	subl	$80, %esp
#APP
# 7 "bench.h" 1
	rdtsc
# 0 "" 2
#NO_APP
	flds	w1.1855
	flds	.LC0
	fld	%st(1)
	fmul	%st(1), %st
	movl	%eax, 72(%esp)
	movl	%edx, 76(%esp)
	flds	x.1854
	fsub	%st(1), %st
	flds	w2.1856
	fmul	%st(3), %st
	fsubr	%st, %st(1)
	fld	%st(1)
	fmul	%st(4), %st
	faddp	%st, %st(3)
	fxch	%st(4)
	fstps	w2.1856
	fstps	w1.1855
#APP
# 7 "bench.h" 1
	rdtsc
# 0 "" 2
#NO_APP
	subl	72(%esp), %eax
	sbbl	76(%esp), %edx
	fstps	48(%esp)
	fstps	16(%esp)
	fstps	32(%esp)
	movl	%eax, 4(%esp)
	movl	%edx, 8(%esp)
	movl	$.LC2, (%esp)
	call	printf
	flds	16(%esp)
	fsts	x.1854
	fsts	w1.1855
	fstps	w2.1856
	flds	48(%esp)
	flds	32(%esp)
	faddp	%st, %st(1)
	leave
	ret
	.size	main, .-main
	.data
	.align 4
	.type	w2.1856, @object
	.size	w2.1856, 4
w2.1856:
	.long	1088421888
	.align 4
	.type	w1.1855, @object
	.size	w1.1855, 4
w1.1855:
	.long	1088421888
	.align 4
	.type	x.1854, @object
	.size	x.1854, 4
x.1854:
	.long	1088421888
	.section	.rodata.cst4,"aM",@progbits,4
	.align 4
.LC0:
	.long	1088421888
	.ident	"GCC: (Debian 4.4.5-5) 4.4.5"
	.section	.note.GNU-stack,"",@progbits
