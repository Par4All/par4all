# -- Machine type EFI2
# mark_description "Intel(R) C Intel(R) 64 Compiler XE for applications running on Intel(R) 64, Version 12.0.1.107 Build 2010111";
# mark_description "6";
# mark_description "-S -O3 -vec-report=2";
	.file "tools.c"
	.text
..TXTST0:
# -- Begin  close_data_file
# mark_begin;
       .align    16,0x90
	.globl close_data_file
close_data_file:
..B1.1:                         # Preds ..B1.0
..___tag_value_close_data_file.1:                               #48.1
        pushq     %rsi                                          #48.1
..___tag_value_close_data_file.3:                               #
        movq      _f_data_file(%rip), %rdi                      #49.6
        testq     %rdi, %rdi                                    #49.22
        je        ..B1.3        # Prob 33%                      #49.22
                                # LOE rbx rbp rdi r12 r13 r14 r15
..B1.2:                         # Preds ..B1.1
        call      fclose                                        #50.3
                                # LOE rbx rbp r12 r13 r14 r15
..B1.3:                         # Preds ..B1.2 ..B1.1
        popq      %rcx                                          #51.1
..___tag_value_close_data_file.4:                               #
        ret                                                     #51.1
        .align    16,0x90
..___tag_value_close_data_file.5:                               #
                                # LOE
# mark_end;
	.type	close_data_file,@function
	.size	close_data_file,.-close_data_file
	.data
# -- End  close_data_file
	.text
# -- Begin  print_array_float
# mark_begin;
       .align    16,0x90
	.globl print_array_float
print_array_float:
# parameter 1: %rdi
# parameter 2: %rsi
# parameter 3: %edx
..B2.1:                         # Preds ..B2.0
..___tag_value_print_array_float.6:                             #54.1
        pushq     %r12                                          #54.1
..___tag_value_print_array_float.8:                             #
        pushq     %r13                                          #54.1
..___tag_value_print_array_float.10:                            #
        subq      $24, %rsp                                     #54.1
..___tag_value_print_array_float.12:                            #
        movq      %rsi, %r13                                    #54.1
        movq      %rdi, %rsi                                    #54.1
        movl      $.L_2__STRING.2, %edi                         #55.2
        xorl      %eax, %eax                                    #55.2
        movl      %edx, %r12d                                   #54.1
        call      printf                                        #55.2
                                # LOE rbx rbp r13 r14 r15 r12d
..B2.2:                         # Preds ..B2.1
        movl      $.L_2__STRING.3, %edi                         #55.2
        xorl      %eax, %eax                                    #55.2
        call      printf                                        #55.2
                                # LOE rbx rbp r13 r14 r15 r12d
..B2.3:                         # Preds ..B2.2
        movl      $.L_2__STRING.4, %esi                         #55.2
        lea       8(%rsp), %rdi                                 #55.2
        movl      $7, %edx                                      #55.2
        movb      $0, 15(%rsp)                                  #55.2
        call      strncpy                                       #55.2
                                # LOE rbx rbp r13 r14 r15 r12d
..B2.4:                         # Preds ..B2.3
        movl      $.L_2__STRING.5, %esi                         #55.2
        lea       8(%rsp), %rdi                                 #55.2
        movl      $2, %edx                                      #55.2
        call      strncat                                       #55.2
                                # LOE rbx rbp r13 r14 r15 r12d
..B2.5:                         # Preds ..B2.4
        xorl      %eax, %eax                                    #55.2
        testl     %r12d, %r12d                                  #55.2
        jbe       ..B2.10       # Prob 10%                      #55.2
                                # LOE rax rbx rbp r13 r14 r15 r12d
..B2.6:                         # Preds ..B2.5
        movslq    %r12d, %r12                                   #55.2
        movq      %r14, (%rsp)                                  #55.2
..___tag_value_print_array_float.13:                            #
        movq      %r12, %r14                                    #55.2
        movq      %rax, %r12                                    #55.2
..___tag_value_print_array_float.15:                            #
                                # LOE rbx rbp r12 r13 r14 r15
..B2.7:                         # Preds ..B2.8 ..B2.6
        cvtss2sd  (%r13,%r12,4), %xmm0                          #55.2
        movl      $1, %eax                                      #55.2
        lea       8(%rsp), %rdi                                 #55.2
        call      printf                                        #55.2
                                # LOE rbx rbp r12 r13 r14 r15
..B2.8:                         # Preds ..B2.7
        incq      %r12                                          #55.2
        cmpq      %r14, %r12                                    #55.2
        jb        ..B2.7        # Prob 82%                      #55.2
                                # LOE rbx rbp r12 r13 r14 r15
..B2.9:                         # Preds ..B2.8
        movq      (%rsp), %r14                                  #
..___tag_value_print_array_float.16:                            #
                                # LOE rbx rbp r14 r15
..B2.10:                        # Preds ..B2.9 ..B2.5
        movl      $.L_2__STRING.3, %edi                         #55.2
        xorl      %eax, %eax                                    #55.2
        call      printf                                        #55.2
                                # LOE rbx rbp r14 r15
..B2.11:                        # Preds ..B2.10
        addq      $24, %rsp                                     #56.1
..___tag_value_print_array_float.17:                            #
        popq      %r13                                          #56.1
..___tag_value_print_array_float.19:                            #
        popq      %r12                                          #56.1
..___tag_value_print_array_float.21:                            #
        ret                                                     #56.1
        .align    16,0x90
..___tag_value_print_array_float.22:                            #
                                # LOE
# mark_end;
	.type	print_array_float,@function
	.size	print_array_float,.-print_array_float
	.data
# -- End  print_array_float
	.text
# -- Begin  print_array_double
# mark_begin;
       .align    16,0x90
	.globl print_array_double
print_array_double:
# parameter 1: %rdi
# parameter 2: %rsi
# parameter 3: %edx
..B3.1:                         # Preds ..B3.0
..___tag_value_print_array_double.23:                           #59.1
        pushq     %r12                                          #59.1
..___tag_value_print_array_double.25:                           #
        pushq     %r13                                          #59.1
..___tag_value_print_array_double.27:                           #
        subq      $24, %rsp                                     #59.1
..___tag_value_print_array_double.29:                           #
        movq      %rsi, %r13                                    #59.1
        movq      %rdi, %rsi                                    #59.1
        movl      $.L_2__STRING.2, %edi                         #60.2
        xorl      %eax, %eax                                    #60.2
        movl      %edx, %r12d                                   #59.1
        call      printf                                        #60.2
                                # LOE rbx rbp r13 r14 r15 r12d
..B3.2:                         # Preds ..B3.1
        movl      $.L_2__STRING.3, %edi                         #60.2
        xorl      %eax, %eax                                    #60.2
        call      printf                                        #60.2
                                # LOE rbx rbp r13 r14 r15 r12d
..B3.3:                         # Preds ..B3.2
        movl      $.L_2__STRING.6, %esi                         #60.2
        lea       8(%rsp), %rdi                                 #60.2
        movl      $7, %edx                                      #60.2
        movb      $0, 15(%rsp)                                  #60.2
        call      strncpy                                       #60.2
                                # LOE rbx rbp r13 r14 r15 r12d
..B3.4:                         # Preds ..B3.3
        movl      $.L_2__STRING.5, %esi                         #60.2
        lea       8(%rsp), %rdi                                 #60.2
        movl      $2, %edx                                      #60.2
        call      strncat                                       #60.2
                                # LOE rbx rbp r13 r14 r15 r12d
..B3.5:                         # Preds ..B3.4
        xorl      %eax, %eax                                    #60.2
        testl     %r12d, %r12d                                  #60.2
        jbe       ..B3.10       # Prob 10%                      #60.2
                                # LOE rax rbx rbp r13 r14 r15 r12d
..B3.6:                         # Preds ..B3.5
        movslq    %r12d, %r12                                   #60.2
        movq      %r14, (%rsp)                                  #60.2
..___tag_value_print_array_double.30:                           #
        movq      %r12, %r14                                    #60.2
        movq      %rax, %r12                                    #60.2
..___tag_value_print_array_double.32:                           #
                                # LOE rbx rbp r12 r13 r14 r15
..B3.7:                         # Preds ..B3.8 ..B3.6
        cvtss2sd  (%r13,%r12,4), %xmm0                          #60.2
        movl      $1, %eax                                      #60.2
        lea       8(%rsp), %rdi                                 #60.2
        call      printf                                        #60.2
                                # LOE rbx rbp r12 r13 r14 r15
..B3.8:                         # Preds ..B3.7
        incq      %r12                                          #60.2
        cmpq      %r14, %r12                                    #60.2
        jb        ..B3.7        # Prob 82%                      #60.2
                                # LOE rbx rbp r12 r13 r14 r15
..B3.9:                         # Preds ..B3.8
        movq      (%rsp), %r14                                  #
..___tag_value_print_array_double.33:                           #
                                # LOE rbx rbp r14 r15
..B3.10:                        # Preds ..B3.9 ..B3.5
        movl      $.L_2__STRING.3, %edi                         #60.2
        xorl      %eax, %eax                                    #60.2
        call      printf                                        #60.2
                                # LOE rbx rbp r14 r15
..B3.11:                        # Preds ..B3.10
        addq      $24, %rsp                                     #61.1
..___tag_value_print_array_double.34:                           #
        popq      %r13                                          #61.1
..___tag_value_print_array_double.36:                           #
        popq      %r12                                          #61.1
..___tag_value_print_array_double.38:                           #
        ret                                                     #61.1
        .align    16,0x90
..___tag_value_print_array_double.39:                           #
                                # LOE
# mark_end;
	.type	print_array_double,@function
	.size	print_array_double,.-print_array_double
	.data
# -- End  print_array_double
	.text
# -- Begin  print_array_long
# mark_begin;
       .align    16,0x90
	.globl print_array_long
print_array_long:
# parameter 1: %rdi
# parameter 2: %rsi
# parameter 3: %edx
..B4.1:                         # Preds ..B4.0
..___tag_value_print_array_long.40:                             #64.1
        pushq     %r12                                          #64.1
..___tag_value_print_array_long.42:                             #
        pushq     %r13                                          #64.1
..___tag_value_print_array_long.44:                             #
        subq      $24, %rsp                                     #64.1
..___tag_value_print_array_long.46:                             #
        movq      %rsi, %r13                                    #64.1
        movq      %rdi, %rsi                                    #64.1
        movl      $.L_2__STRING.2, %edi                         #65.2
        xorl      %eax, %eax                                    #65.2
        movl      %edx, %r12d                                   #64.1
        call      printf                                        #65.2
                                # LOE rbx rbp r13 r14 r15 r12d
..B4.2:                         # Preds ..B4.1
        movl      $.L_2__STRING.3, %edi                         #65.2
        xorl      %eax, %eax                                    #65.2
        call      printf                                        #65.2
                                # LOE rbx rbp r13 r14 r15 r12d
..B4.3:                         # Preds ..B4.2
        movl      $.L_2__STRING.6, %esi                         #65.2
        lea       8(%rsp), %rdi                                 #65.2
        movl      $7, %edx                                      #65.2
        movb      $0, 15(%rsp)                                  #65.2
        call      strncpy                                       #65.2
                                # LOE rbx rbp r13 r14 r15 r12d
..B4.4:                         # Preds ..B4.3
        movl      $.L_2__STRING.5, %esi                         #65.2
        lea       8(%rsp), %rdi                                 #65.2
        movl      $2, %edx                                      #65.2
        call      strncat                                       #65.2
                                # LOE rbx rbp r13 r14 r15 r12d
..B4.5:                         # Preds ..B4.4
        xorl      %eax, %eax                                    #65.2
        testl     %r12d, %r12d                                  #65.2
        jbe       ..B4.10       # Prob 10%                      #65.2
                                # LOE rax rbx rbp r13 r14 r15 r12d
..B4.6:                         # Preds ..B4.5
        movslq    %r12d, %r12                                   #65.2
        movq      %r14, (%rsp)                                  #65.2
..___tag_value_print_array_long.47:                             #
        movq      %r12, %r14                                    #65.2
        movq      %rax, %r12                                    #65.2
..___tag_value_print_array_long.49:                             #
                                # LOE rbx rbp r12 r13 r14 r15
..B4.7:                         # Preds ..B4.8 ..B4.6
        xorl      %eax, %eax                                    #65.2
        lea       8(%rsp), %rdi                                 #65.2
        movq      (%r13,%r12,8), %rsi                           #65.2
        call      printf                                        #65.2
                                # LOE rbx rbp r12 r13 r14 r15
..B4.8:                         # Preds ..B4.7
        incq      %r12                                          #65.2
        cmpq      %r14, %r12                                    #65.2
        jb        ..B4.7        # Prob 82%                      #65.2
                                # LOE rbx rbp r12 r13 r14 r15
..B4.9:                         # Preds ..B4.8
        movq      (%rsp), %r14                                  #
..___tag_value_print_array_long.50:                             #
                                # LOE rbx rbp r14 r15
..B4.10:                        # Preds ..B4.9 ..B4.5
        movl      $.L_2__STRING.3, %edi                         #65.2
        xorl      %eax, %eax                                    #65.2
        call      printf                                        #65.2
                                # LOE rbx rbp r14 r15
..B4.11:                        # Preds ..B4.10
        addq      $24, %rsp                                     #66.1
..___tag_value_print_array_long.51:                             #
        popq      %r13                                          #66.1
..___tag_value_print_array_long.53:                             #
        popq      %r12                                          #66.1
..___tag_value_print_array_long.55:                             #
        ret                                                     #66.1
        .align    16,0x90
..___tag_value_print_array_long.56:                             #
                                # LOE
# mark_end;
	.type	print_array_long,@function
	.size	print_array_long,.-print_array_long
	.data
# -- End  print_array_long
	.text
# -- Begin  init_data_long
# mark_begin;
       .align    16,0x90
	.globl init_data_long
init_data_long:
# parameter 1: %rdi
# parameter 2: %esi
..B5.1:                         # Preds ..B5.0
..___tag_value_init_data_long.57:                               #123.1
        pushq     %r12                                          #123.1
..___tag_value_init_data_long.59:                               #
        subq      $32, %rsp                                     #123.1
..___tag_value_init_data_long.61:                               #
        movl      %esi, %esi                                    #123.1
        shlq      $3, %rsi                                      #124.9
        movq      _f_data_file(%rip), %r12                      #124.9
        testq     %r12, %r12                                    #124.9
        je        ..B5.17       # Prob 1%                       #124.9
                                # LOE rbx rbp rsi rdi r12 r13 r14 r15
..B5.2:                         # Preds ..B5.1
        testq     %rsi, %rsi                                    #124.9
        jle       ..B5.13       # Prob 1%                       #124.9
                                # LOE rax rbx rbp rsi rdi r12 r13 r14 r15
..B5.3:                         # Preds ..B5.2
        movq      %r13, 16(%rsp)                                #
..___tag_value_init_data_long.62:                               #
        movq      %r14, 8(%rsp)                                 #
..___tag_value_init_data_long.64:                               #
        movq      %rdi, %r14                                    #
        movq      %r15, (%rsp)                                  #
..___tag_value_init_data_long.66:                               #
        movq      %rsi, %r15                                    #
..___tag_value_init_data_long.68:                               #
                                # LOE rbx rbp r12 r14 r15
..B5.4:                         # Preds ..B5.10 ..B5.3
        movq      %r14, %rdi                                    #124.9
        movl      $1, %esi                                      #124.9
        movq      %r15, %rdx                                    #124.9
        movq      %r12, %rcx                                    #124.9
        call      fread                                         #124.9
                                # LOE rax rbx rbp r12 r14 r15
..B5.21:                        # Preds ..B5.4
        movq      %rax, %r13                                    #124.9
                                # LOE rbx rbp r12 r13 r14 r15
..B5.5:                         # Preds ..B5.21
        testq     %r13, %r13                                    #124.9
        jne       ..B5.7        # Prob 50%                      #124.9
                                # LOE rbx rbp r12 r13 r14 r15
..B5.6:                         # Preds ..B5.5
        movq      %r12, %rdi                                    #124.9
        call      ferror                                        #124.9
                                # LOE rbx rbp r12 r13 r14 r15 eax
..B5.22:                        # Preds ..B5.6
        testl     %eax, %eax                                    #124.9
        jne       ..B5.14       # Prob 20%                      #124.9
                                # LOE rbx rbp r12 r13 r14 r15
..B5.7:                         # Preds ..B5.22 ..B5.5
        cmpq      %r15, %r13                                    #124.9
        jge       ..B5.10       # Prob 78%                      #124.9
                                # LOE rbx rbp r12 r13 r14 r15
..B5.8:                         # Preds ..B5.7
        movq      %r12, %rdi                                    #124.9
        xorl      %esi, %esi                                    #124.9
        xorl      %edx, %edx                                    #124.9
        call      fseek                                         #124.9
                                # LOE rbx rbp r12 r13 r14 r15
..B5.9:                         # Preds ..B5.8
        movq      %r12, %rdi                                    #124.9
        call      fflush                                        #124.9
                                # LOE rbx rbp r12 r13 r14 r15
..B5.10:                        # Preds ..B5.9 ..B5.7
        subq      %r13, %r15                                    #124.9
        addq      %r13, %r14                                    #124.9
        testq     %r15, %r15                                    #124.9
        jg        ..B5.4        # Prob 82%                      #124.9
                                # LOE rbx rbp r12 r13 r14 r15
..B5.11:                        # Preds ..B5.10
        movq      8(%rsp), %r14                                 #
..___tag_value_init_data_long.71:                               #
        movq      %r13, %rax                                    #
        movq      16(%rsp), %r13                                #
..___tag_value_init_data_long.72:                               #
        movq      (%rsp), %r15                                  #
..___tag_value_init_data_long.73:                               #
                                # LOE rax rbx rbp r13 r14 r15
..B5.13:                        # Preds ..B5.2 ..B5.11 ..B5.23
        addq      $32, %rsp                                     #124.9
..___tag_value_init_data_long.74:                               #
        popq      %r12                                          #124.9
..___tag_value_init_data_long.76:                               #
        ret                                                     #124.9
..___tag_value_init_data_long.77:                               #
                                # LOE
..B5.14:                        # Preds ..B5.22                 # Infreq
        movl      $.L_2__STRING.8, %edi                         #124.9
        movq      16(%rsp), %r13                                #
..___tag_value_init_data_long.82:                               #
        movq      8(%rsp), %r14                                 #
..___tag_value_init_data_long.83:                               #
        movq      (%rsp), %r15                                  #
..___tag_value_init_data_long.84:                               #
        call      perror                                        #124.9
..___tag_value_init_data_long.85:                               #
                                # LOE rbx rbp r12 r13 r14 r15
..B5.15:                        # Preds ..B5.14                 # Infreq
        movq      %r12, %rdi                                    #124.9
        call      clearerr                                      #124.9
                                # LOE rbx rbp r13 r14 r15
..B5.16:                        # Preds ..B5.15                 # Infreq
        call      __errno_location                              #124.9
                                # LOE rax rbx rbp r13 r14 r15
..B5.23:                        # Preds ..B5.16                 # Infreq
        movl      (%rax), %eax                                  #124.9
        jmp       ..B5.13       # Prob 100%                     #124.9
..___tag_value_init_data_long.88:                               #
                                # LOE rbx rbp r13 r14 r15 eax
..B5.17:                        # Preds ..B5.1                  # Infreq
        movl      $.L_2__STRING.7, %edi                         #124.9
        xorl      %eax, %eax                                    #124.9
        movq      stderr(%rip), %rsi                            #124.9
        call      fputs                                         #124.9
                                # LOE
..B5.18:                        # Preds ..B5.17                 # Infreq
        movl      $1, %edi                                      #124.9
        call      exit                                          #124.9
        .align    16,0x90
..___tag_value_init_data_long.91:                               #
                                # LOE
# mark_end;
	.type	init_data_long,@function
	.size	init_data_long,.-init_data_long
	.data
# -- End  init_data_long
	.text
# -- Begin  init_data_double
# mark_begin;
       .align    16,0x90
	.globl init_data_double
init_data_double:
# parameter 1: %rdi
# parameter 2: %esi
..B6.1:                         # Preds ..B6.0
..___tag_value_init_data_double.92:                             #118.1
        pushq     %r12                                          #118.1
..___tag_value_init_data_double.94:                             #
        subq      $32, %rsp                                     #118.1
..___tag_value_init_data_double.96:                             #
        movl      %esi, %esi                                    #118.1
        shlq      $3, %rsi                                      #119.9
        movq      _f_data_file(%rip), %r12                      #119.9
        testq     %r12, %r12                                    #119.9
        je        ..B6.17       # Prob 1%                       #119.9
                                # LOE rbx rbp rsi rdi r12 r13 r14 r15
..B6.2:                         # Preds ..B6.1
        testq     %rsi, %rsi                                    #119.9
        jle       ..B6.13       # Prob 1%                       #119.9
                                # LOE rax rbx rbp rsi rdi r12 r13 r14 r15
..B6.3:                         # Preds ..B6.2
        movq      %r13, 16(%rsp)                                #
..___tag_value_init_data_double.97:                             #
        movq      %r14, 8(%rsp)                                 #
..___tag_value_init_data_double.99:                             #
        movq      %rdi, %r14                                    #
        movq      %r15, (%rsp)                                  #
..___tag_value_init_data_double.101:                            #
        movq      %rsi, %r15                                    #
..___tag_value_init_data_double.103:                            #
                                # LOE rbx rbp r12 r14 r15
..B6.4:                         # Preds ..B6.10 ..B6.3
        movq      %r14, %rdi                                    #119.9
        movl      $1, %esi                                      #119.9
        movq      %r15, %rdx                                    #119.9
        movq      %r12, %rcx                                    #119.9
        call      fread                                         #119.9
                                # LOE rax rbx rbp r12 r14 r15
..B6.21:                        # Preds ..B6.4
        movq      %rax, %r13                                    #119.9
                                # LOE rbx rbp r12 r13 r14 r15
..B6.5:                         # Preds ..B6.21
        testq     %r13, %r13                                    #119.9
        jne       ..B6.7        # Prob 50%                      #119.9
                                # LOE rbx rbp r12 r13 r14 r15
..B6.6:                         # Preds ..B6.5
        movq      %r12, %rdi                                    #119.9
        call      ferror                                        #119.9
                                # LOE rbx rbp r12 r13 r14 r15 eax
..B6.22:                        # Preds ..B6.6
        testl     %eax, %eax                                    #119.9
        jne       ..B6.14       # Prob 20%                      #119.9
                                # LOE rbx rbp r12 r13 r14 r15
..B6.7:                         # Preds ..B6.22 ..B6.5
        cmpq      %r15, %r13                                    #119.9
        jge       ..B6.10       # Prob 78%                      #119.9
                                # LOE rbx rbp r12 r13 r14 r15
..B6.8:                         # Preds ..B6.7
        movq      %r12, %rdi                                    #119.9
        xorl      %esi, %esi                                    #119.9
        xorl      %edx, %edx                                    #119.9
        call      fseek                                         #119.9
                                # LOE rbx rbp r12 r13 r14 r15
..B6.9:                         # Preds ..B6.8
        movq      %r12, %rdi                                    #119.9
        call      fflush                                        #119.9
                                # LOE rbx rbp r12 r13 r14 r15
..B6.10:                        # Preds ..B6.9 ..B6.7
        subq      %r13, %r15                                    #119.9
        addq      %r13, %r14                                    #119.9
        testq     %r15, %r15                                    #119.9
        jg        ..B6.4        # Prob 82%                      #119.9
                                # LOE rbx rbp r12 r13 r14 r15
..B6.11:                        # Preds ..B6.10
        movq      8(%rsp), %r14                                 #
..___tag_value_init_data_double.106:                            #
        movq      %r13, %rax                                    #
        movq      16(%rsp), %r13                                #
..___tag_value_init_data_double.107:                            #
        movq      (%rsp), %r15                                  #
..___tag_value_init_data_double.108:                            #
                                # LOE rax rbx rbp r13 r14 r15
..B6.13:                        # Preds ..B6.2 ..B6.11 ..B6.23
        addq      $32, %rsp                                     #119.9
..___tag_value_init_data_double.109:                            #
        popq      %r12                                          #119.9
..___tag_value_init_data_double.111:                            #
        ret                                                     #119.9
..___tag_value_init_data_double.112:                            #
                                # LOE
..B6.14:                        # Preds ..B6.22                 # Infreq
        movl      $.L_2__STRING.8, %edi                         #119.9
        movq      16(%rsp), %r13                                #
..___tag_value_init_data_double.117:                            #
        movq      8(%rsp), %r14                                 #
..___tag_value_init_data_double.118:                            #
        movq      (%rsp), %r15                                  #
..___tag_value_init_data_double.119:                            #
        call      perror                                        #119.9
..___tag_value_init_data_double.120:                            #
                                # LOE rbx rbp r12 r13 r14 r15
..B6.15:                        # Preds ..B6.14                 # Infreq
        movq      %r12, %rdi                                    #119.9
        call      clearerr                                      #119.9
                                # LOE rbx rbp r13 r14 r15
..B6.16:                        # Preds ..B6.15                 # Infreq
        call      __errno_location                              #119.9
                                # LOE rax rbx rbp r13 r14 r15
..B6.23:                        # Preds ..B6.16                 # Infreq
        movl      (%rax), %eax                                  #119.9
        jmp       ..B6.13       # Prob 100%                     #119.9
..___tag_value_init_data_double.123:                            #
                                # LOE rbx rbp r13 r14 r15 eax
..B6.17:                        # Preds ..B6.1                  # Infreq
        movl      $.L_2__STRING.7, %edi                         #119.9
        xorl      %eax, %eax                                    #119.9
        movq      stderr(%rip), %rsi                            #119.9
        call      fputs                                         #119.9
                                # LOE
..B6.18:                        # Preds ..B6.17                 # Infreq
        movl      $1, %edi                                      #119.9
        call      exit                                          #119.9
        .align    16,0x90
..___tag_value_init_data_double.126:                            #
                                # LOE
# mark_end;
	.type	init_data_double,@function
	.size	init_data_double,.-init_data_double
	.data
# -- End  init_data_double
	.text
# -- Begin  init_data_float
# mark_begin;
       .align    16,0x90
	.globl init_data_float
init_data_float:
# parameter 1: %rdi
# parameter 2: %esi
..B7.1:                         # Preds ..B7.0
..___tag_value_init_data_float.127:                             #112.1
        pushq     %r12                                          #112.1
..___tag_value_init_data_float.129:                             #
        subq      $32, %rsp                                     #112.1
..___tag_value_init_data_float.131:                             #
        movl      %esi, %esi                                    #112.1
        shlq      $2, %rsi                                      #113.10
        movq      _f_data_file(%rip), %r12                      #113.10
        testq     %r12, %r12                                    #113.10
        je        ..B7.17       # Prob 1%                       #113.10
                                # LOE rbx rbp rsi rdi r12 r13 r14 r15
..B7.2:                         # Preds ..B7.1
        testq     %rsi, %rsi                                    #113.10
        jle       ..B7.13       # Prob 1%                       #113.10
                                # LOE rax rbx rbp rsi rdi r12 r13 r14 r15
..B7.3:                         # Preds ..B7.2
        movq      %r13, 16(%rsp)                                #
..___tag_value_init_data_float.132:                             #
        movq      %r14, 8(%rsp)                                 #
..___tag_value_init_data_float.134:                             #
        movq      %rdi, %r14                                    #
        movq      %r15, (%rsp)                                  #
..___tag_value_init_data_float.136:                             #
        movq      %rsi, %r15                                    #
..___tag_value_init_data_float.138:                             #
                                # LOE rbx rbp r12 r14 r15
..B7.4:                         # Preds ..B7.10 ..B7.3
        movq      %r14, %rdi                                    #113.10
        movl      $1, %esi                                      #113.10
        movq      %r15, %rdx                                    #113.10
        movq      %r12, %rcx                                    #113.10
        call      fread                                         #113.10
                                # LOE rax rbx rbp r12 r14 r15
..B7.21:                        # Preds ..B7.4
        movq      %rax, %r13                                    #113.10
                                # LOE rbx rbp r12 r13 r14 r15
..B7.5:                         # Preds ..B7.21
        testq     %r13, %r13                                    #113.10
        jne       ..B7.7        # Prob 50%                      #113.10
                                # LOE rbx rbp r12 r13 r14 r15
..B7.6:                         # Preds ..B7.5
        movq      %r12, %rdi                                    #113.10
        call      ferror                                        #113.10
                                # LOE rbx rbp r12 r13 r14 r15 eax
..B7.22:                        # Preds ..B7.6
        testl     %eax, %eax                                    #113.10
        jne       ..B7.14       # Prob 20%                      #113.10
                                # LOE rbx rbp r12 r13 r14 r15
..B7.7:                         # Preds ..B7.22 ..B7.5
        cmpq      %r15, %r13                                    #113.10
        jge       ..B7.10       # Prob 78%                      #113.10
                                # LOE rbx rbp r12 r13 r14 r15
..B7.8:                         # Preds ..B7.7
        movq      %r12, %rdi                                    #113.10
        xorl      %esi, %esi                                    #113.10
        xorl      %edx, %edx                                    #113.10
        call      fseek                                         #113.10
                                # LOE rbx rbp r12 r13 r14 r15
..B7.9:                         # Preds ..B7.8
        movq      %r12, %rdi                                    #113.10
        call      fflush                                        #113.10
                                # LOE rbx rbp r12 r13 r14 r15
..B7.10:                        # Preds ..B7.9 ..B7.7
        subq      %r13, %r15                                    #113.10
        addq      %r13, %r14                                    #113.10
        testq     %r15, %r15                                    #113.10
        jg        ..B7.4        # Prob 82%                      #113.10
                                # LOE rbx rbp r12 r13 r14 r15
..B7.11:                        # Preds ..B7.10
        movq      8(%rsp), %r14                                 #
..___tag_value_init_data_float.141:                             #
        movq      %r13, %rax                                    #
        movq      16(%rsp), %r13                                #
..___tag_value_init_data_float.142:                             #
        movq      (%rsp), %r15                                  #
..___tag_value_init_data_float.143:                             #
                                # LOE rax rbx rbp r13 r14 r15
..B7.13:                        # Preds ..B7.2 ..B7.11 ..B7.23
        addq      $32, %rsp                                     #114.9
..___tag_value_init_data_float.144:                             #
        popq      %r12                                          #114.9
..___tag_value_init_data_float.146:                             #
        ret                                                     #114.9
..___tag_value_init_data_float.147:                             #
                                # LOE
..B7.14:                        # Preds ..B7.22                 # Infreq
        movl      $.L_2__STRING.8, %edi                         #113.10
        movq      16(%rsp), %r13                                #
..___tag_value_init_data_float.152:                             #
        movq      8(%rsp), %r14                                 #
..___tag_value_init_data_float.153:                             #
        movq      (%rsp), %r15                                  #
..___tag_value_init_data_float.154:                             #
        call      perror                                        #113.10
..___tag_value_init_data_float.155:                             #
                                # LOE rbx rbp r12 r13 r14 r15
..B7.15:                        # Preds ..B7.14                 # Infreq
        movq      %r12, %rdi                                    #113.10
        call      clearerr                                      #113.10
                                # LOE rbx rbp r13 r14 r15
..B7.16:                        # Preds ..B7.15                 # Infreq
        call      __errno_location                              #113.10
                                # LOE rax rbx rbp r13 r14 r15
..B7.23:                        # Preds ..B7.16                 # Infreq
        movl      (%rax), %eax                                  #113.10
        jmp       ..B7.13       # Prob 100%                     #113.10
..___tag_value_init_data_float.158:                             #
                                # LOE rbx rbp r13 r14 r15 eax
..B7.17:                        # Preds ..B7.1                  # Infreq
        movl      $.L_2__STRING.7, %edi                         #113.10
        xorl      %eax, %eax                                    #113.10
        movq      stderr(%rip), %rsi                            #113.10
        call      fputs                                         #113.10
                                # LOE
..B7.18:                        # Preds ..B7.17                 # Infreq
        movl      $1, %edi                                      #113.10
        call      exit                                          #113.10
        .align    16,0x90
..___tag_value_init_data_float.161:                             #
                                # LOE
# mark_end;
	.type	init_data_float,@function
	.size	init_data_float,.-init_data_float
	.data
# -- End  init_data_float
	.text
# -- Begin  init_data_gen
# mark_begin;
       .align    16,0x90
	.globl init_data_gen
init_data_gen:
# parameter 1: %rdi
# parameter 2: %esi
# parameter 3: %rdx
..B8.1:                         # Preds ..B8.0
..___tag_value_init_data_gen.162:                               #107.1
        pushq     %r12                                          #107.1
..___tag_value_init_data_gen.164:                               #
        subq      $32, %rsp                                     #107.1
..___tag_value_init_data_gen.166:                               #
        movl      %esi, %esi                                    #107.1
        imulq     %rsi, %rdx                                    #108.38
        movq      _f_data_file(%rip), %r12                      #108.9
        testq     %r12, %r12                                    #108.9
        je        ..B8.17       # Prob 1%                       #108.9
                                # LOE rdx rbx rbp rdi r12 r13 r14 r15
..B8.2:                         # Preds ..B8.1
        testq     %rdx, %rdx                                    #108.9
        jle       ..B8.13       # Prob 1%                       #108.9
                                # LOE rax rdx rbx rbp rdi r12 r13 r14 r15
..B8.3:                         # Preds ..B8.2
        movq      %r13, 16(%rsp)                                #
..___tag_value_init_data_gen.167:                               #
        movq      %r14, 8(%rsp)                                 #
..___tag_value_init_data_gen.169:                               #
        movq      %rdi, %r14                                    #
        movq      %r15, (%rsp)                                  #
..___tag_value_init_data_gen.171:                               #
        movq      %rdx, %r15                                    #
..___tag_value_init_data_gen.173:                               #
                                # LOE rbx rbp r12 r14 r15
..B8.4:                         # Preds ..B8.10 ..B8.3
        movq      %r14, %rdi                                    #108.9
        movl      $1, %esi                                      #108.9
        movq      %r15, %rdx                                    #108.9
        movq      %r12, %rcx                                    #108.9
        call      fread                                         #108.9
                                # LOE rax rbx rbp r12 r14 r15
..B8.21:                        # Preds ..B8.4
        movq      %rax, %r13                                    #108.9
                                # LOE rbx rbp r12 r13 r14 r15
..B8.5:                         # Preds ..B8.21
        testq     %r13, %r13                                    #108.9
        jne       ..B8.7        # Prob 50%                      #108.9
                                # LOE rbx rbp r12 r13 r14 r15
..B8.6:                         # Preds ..B8.5
        movq      %r12, %rdi                                    #108.9
        call      ferror                                        #108.9
                                # LOE rbx rbp r12 r13 r14 r15 eax
..B8.22:                        # Preds ..B8.6
        testl     %eax, %eax                                    #108.9
        jne       ..B8.14       # Prob 20%                      #108.9
                                # LOE rbx rbp r12 r13 r14 r15
..B8.7:                         # Preds ..B8.22 ..B8.5
        cmpq      %r15, %r13                                    #108.9
        jge       ..B8.10       # Prob 78%                      #108.9
                                # LOE rbx rbp r12 r13 r14 r15
..B8.8:                         # Preds ..B8.7
        movq      %r12, %rdi                                    #108.9
        xorl      %esi, %esi                                    #108.9
        xorl      %edx, %edx                                    #108.9
        call      fseek                                         #108.9
                                # LOE rbx rbp r12 r13 r14 r15
..B8.9:                         # Preds ..B8.8
        movq      %r12, %rdi                                    #108.9
        call      fflush                                        #108.9
                                # LOE rbx rbp r12 r13 r14 r15
..B8.10:                        # Preds ..B8.9 ..B8.7
        subq      %r13, %r15                                    #108.9
        addq      %r13, %r14                                    #108.9
        testq     %r15, %r15                                    #108.9
        jg        ..B8.4        # Prob 82%                      #108.9
                                # LOE rbx rbp r12 r13 r14 r15
..B8.11:                        # Preds ..B8.10
        movq      8(%rsp), %r14                                 #
..___tag_value_init_data_gen.176:                               #
        movq      %r13, %rax                                    #
        movq      16(%rsp), %r13                                #
..___tag_value_init_data_gen.177:                               #
        movq      (%rsp), %r15                                  #
..___tag_value_init_data_gen.178:                               #
                                # LOE rax rbx rbp r13 r14 r15
..B8.13:                        # Preds ..B8.2 ..B8.11 ..B8.23
        addq      $32, %rsp                                     #108.9
..___tag_value_init_data_gen.179:                               #
        popq      %r12                                          #108.9
..___tag_value_init_data_gen.181:                               #
        ret                                                     #108.9
..___tag_value_init_data_gen.182:                               #
                                # LOE
..B8.14:                        # Preds ..B8.22                 # Infreq
        movl      $.L_2__STRING.8, %edi                         #108.9
        movq      16(%rsp), %r13                                #
..___tag_value_init_data_gen.187:                               #
        movq      8(%rsp), %r14                                 #
..___tag_value_init_data_gen.188:                               #
        movq      (%rsp), %r15                                  #
..___tag_value_init_data_gen.189:                               #
        call      perror                                        #108.9
..___tag_value_init_data_gen.190:                               #
                                # LOE rbx rbp r12 r13 r14 r15
..B8.15:                        # Preds ..B8.14                 # Infreq
        movq      %r12, %rdi                                    #108.9
        call      clearerr                                      #108.9
                                # LOE rbx rbp r13 r14 r15
..B8.16:                        # Preds ..B8.15                 # Infreq
        call      __errno_location                              #108.9
                                # LOE rax rbx rbp r13 r14 r15
..B8.23:                        # Preds ..B8.16                 # Infreq
        movl      (%rax), %eax                                  #108.9
        jmp       ..B8.13       # Prob 100%                     #108.9
..___tag_value_init_data_gen.193:                               #
                                # LOE rbx rbp r13 r14 r15 eax
..B8.17:                        # Preds ..B8.1                  # Infreq
        movl      $.L_2__STRING.7, %edi                         #108.9
        xorl      %eax, %eax                                    #108.9
        movq      stderr(%rip), %rsi                            #108.9
        call      fputs                                         #108.9
                                # LOE
..B8.18:                        # Preds ..B8.17                 # Infreq
        movl      $1, %edi                                      #108.9
        call      exit                                          #108.9
        .align    16,0x90
..___tag_value_init_data_gen.196:                               #
                                # LOE
# mark_end;
	.type	init_data_gen,@function
	.size	init_data_gen,.-init_data_gen
	.data
# -- End  init_data_gen
	.text
# -- Begin  _init_data
# mark_begin;
       .align    16,0x90
	.globl _init_data
_init_data:
# parameter 1: %rdi
# parameter 2: %rsi
..B9.1:                         # Preds ..B9.0
..___tag_value__init_data.197:                                  #70.1
        pushq     %r12                                          #70.1
..___tag_value__init_data.199:                                  #
        subq      $32, %rsp                                     #70.1
..___tag_value__init_data.201:                                  #
        movq      _f_data_file(%rip), %r12                      #75.6
        testq     %r12, %r12                                    #75.22
        je        ..B9.16       # Prob 1%                       #75.22
                                # LOE rbx rbp rsi rdi r12 r13 r14 r15
..B9.2:                         # Preds ..B9.1
        testq     %rsi, %rsi                                    #80.19
        jle       ..B9.12       # Prob 1%                       #80.19
                                # LOE rax rbx rbp rsi rdi r12 r13 r14 r15
..B9.3:                         # Preds ..B9.2
        movq      %r13, 16(%rsp)                                #
..___tag_value__init_data.202:                                  #
        movq      %r14, 8(%rsp)                                 #
..___tag_value__init_data.204:                                  #
        movq      %rsi, %r14                                    #
        movq      %r15, (%rsp)                                  #
..___tag_value__init_data.206:                                  #
        movq      %rdi, %r15                                    #
..___tag_value__init_data.208:                                  #
                                # LOE rbx rbp r12 r14 r15
..B9.4:                         # Preds ..B9.10 ..B9.3
        movq      %r15, %rdi                                    #82.8
        movl      $1, %esi                                      #82.8
        movq      %r14, %rdx                                    #82.8
        movq      %r12, %rcx                                    #82.8
        call      fread                                         #82.8
                                # LOE rax rbx rbp r12 r14 r15
..B9.20:                        # Preds ..B9.4
        movq      %rax, %r13                                    #82.31
                                # LOE rbx rbp r12 r13 r14 r15
..B9.5:                         # Preds ..B9.20
        testq     %r13, %r13                                    #83.13
        jne       ..B9.7        # Prob 50%                      #83.13
                                # LOE rbx rbp r12 r13 r14 r15
..B9.6:                         # Preds ..B9.5
        movq      %r12, %rdi                                    #83.18
        call      ferror                                        #83.18
                                # LOE rbx rbp r12 r13 r14 r15 eax
..B9.21:                        # Preds ..B9.6
        testl     %eax, %eax                                    #83.18
        jne       ..B9.13       # Prob 20%                      #83.18
                                # LOE rbx rbp r12 r13 r14 r15
..B9.7:                         # Preds ..B9.21 ..B9.5
        cmpq      %r14, %r13                                    #89.12
        jge       ..B9.10       # Prob 78%                      #89.12
                                # LOE rbx rbp r12 r13 r14 r15
..B9.8:                         # Preds ..B9.7
        movq      %r12, %rdi                                    #92.4
        xorl      %esi, %esi                                    #92.4
        xorl      %edx, %edx                                    #92.4
        call      fseek                                         #92.4
                                # LOE rbx rbp r12 r13 r14 r15
..B9.9:                         # Preds ..B9.8
        movq      %r12, %rdi                                    #93.4
        call      fflush                                        #93.4
                                # LOE rbx rbp r12 r13 r14 r15
..B9.10:                        # Preds ..B9.9 ..B9.7
        subq      %r13, %r14                                    #95.3
        addq      %r13, %r15                                    #96.3
        testq     %r14, %r14                                    #80.19
        jg        ..B9.4        # Prob 82%                      #80.19
                                # LOE rbx rbp r12 r13 r14 r15
..B9.11:                        # Preds ..B9.10
        movq      8(%rsp), %r14                                 #
..___tag_value__init_data.211:                                  #
        movq      %r13, %rax                                    #
        movq      16(%rsp), %r13                                #
..___tag_value__init_data.212:                                  #
        movq      (%rsp), %r15                                  #
..___tag_value__init_data.213:                                  #
                                # LOE rax rbx rbp r13 r14 r15
..B9.12:                        # Preds ..B9.2 ..B9.11
        addq      $32, %rsp                                     #103.9
..___tag_value__init_data.214:                                  #
        popq      %r12                                          #103.9
..___tag_value__init_data.216:                                  #
        ret                                                     #103.9
..___tag_value__init_data.217:                                  #
                                # LOE
..B9.13:                        # Preds ..B9.21                 # Infreq
        movl      $.L_2__STRING.8, %edi                         #85.4
        movq      16(%rsp), %r13                                #
..___tag_value__init_data.222:                                  #
        movq      8(%rsp), %r14                                 #
..___tag_value__init_data.223:                                  #
        movq      (%rsp), %r15                                  #
..___tag_value__init_data.224:                                  #
        call      perror                                        #85.4
..___tag_value__init_data.225:                                  #
                                # LOE rbx rbp r12 r13 r14 r15
..B9.14:                        # Preds ..B9.13                 # Infreq
        movq      %r12, %rdi                                    #86.4
        call      clearerr                                      #86.4
                                # LOE rbx rbp r13 r14 r15
..B9.15:                        # Preds ..B9.14                 # Infreq
        call      __errno_location                              #87.11
                                # LOE rax rbx rbp r13 r14 r15
..B9.22:                        # Preds ..B9.15                 # Infreq
        movl      (%rax), %eax                                  #87.11
        addq      $32, %rsp                                     #87.11
..___tag_value__init_data.228:                                  #
        popq      %r12                                          #87.11
..___tag_value__init_data.230:                                  #
        ret                                                     #87.11
..___tag_value__init_data.231:                                  #
                                # LOE
..B9.16:                        # Preds ..B9.1                  # Infreq
        movl      $.L_2__STRING.7, %edi                         #77.3
        xorl      %eax, %eax                                    #77.3
        movq      stderr(%rip), %rsi                            #77.3
        call      fputs                                         #77.3
                                # LOE
..B9.17:                        # Preds ..B9.16                 # Infreq
        movl      $1, %edi                                      #78.3
        call      exit                                          #78.3
        .align    16,0x90
..___tag_value__init_data.236:                                  #
                                # LOE
# mark_end;
	.type	_init_data,@function
	.size	_init_data,.-_init_data
	.data
# -- End  _init_data
	.text
# -- Begin  init_data_file
# mark_begin;
       .align    16,0x90
	.globl init_data_file
init_data_file:
# parameter 1: %rdi
..B10.1:                        # Preds ..B10.0
..___tag_value_init_data_file.237:                              #36.1
        pushq     %rsi                                          #36.1
..___tag_value_init_data_file.239:                              #
        movq      _f_data_file(%rip), %rax                      #37.6
        testq     %rax, %rax                                    #37.22
        je        ..B10.3       # Prob 10%                      #37.22
                                # LOE rbx rbp rdi r12 r13 r14 r15
..B10.2:                        # Preds ..B10.1
        popq      %rcx                                          #38.3
..___tag_value_init_data_file.240:                              #
        ret                                                     #38.3
..___tag_value_init_data_file.241:                              #
                                # LOE
..B10.3:                        # Preds ..B10.1                 # Infreq
        movl      $.L_2__STRING.0, %esi                         #39.17
        call      fopen                                         #39.17
                                # LOE rax rbx rbp r12 r13 r14 r15
..B10.4:                        # Preds ..B10.3                 # Infreq
        movq      %rax, _f_data_file(%rip)                      #39.2
        testq     %rax, %rax                                    #40.22
        je        ..B10.6       # Prob 1%                       #40.22
                                # LOE rbx rbp r12 r13 r14 r15
..B10.5:                        # Preds ..B10.4                 # Infreq
        popq      %rcx                                          #45.1
..___tag_value_init_data_file.242:                              #
        ret                                                     #45.1
..___tag_value_init_data_file.243:                              #
                                # LOE
..B10.6:                        # Preds ..B10.4                 # Infreq
        movl      $.L_2__STRING.1, %edi                         #42.3
        call      perror                                        #42.3
                                # LOE
..B10.7:                        # Preds ..B10.6                 # Infreq
        call      __errno_location                              #43.8
                                # LOE rax
..B10.11:                       # Preds ..B10.7                 # Infreq
        movl      (%rax), %edi                                  #43.3
        call      exit                                          #43.3
        .align    16,0x90
..___tag_value_init_data_file.244:                              #
                                # LOE
# mark_end;
	.type	init_data_file,@function
	.size	init_data_file,.-init_data_file
	.data
# -- End  init_data_file
	.bss
	.align 8
	.align 8
_f_data_file:
	.type	_f_data_file,@object
	.size	_f_data_file,8
	.space 8	# pad
	.section .rodata.str1.4, "aMS",@progbits,1
	.align 4
	.align 4
.L_2__STRING.2:
	.byte	37
	.byte	115
	.byte	32
	.byte	58
	.byte	10
	.byte	0
	.type	.L_2__STRING.2,@object
	.size	.L_2__STRING.2,6
	.space 2	# pad
	.align 4
.L_2__STRING.3:
	.byte	45
	.byte	45
	.byte	45
	.byte	45
	.byte	10
	.byte	0
	.type	.L_2__STRING.3,@object
	.size	.L_2__STRING.3,6
	.space 2	# pad
	.align 4
.L_2__STRING.4:
	.byte	37
	.byte	102
	.byte	0
	.type	.L_2__STRING.4,@object
	.size	.L_2__STRING.4,3
	.space 1	# pad
	.align 4
.L_2__STRING.5:
	.byte	10
	.byte	0
	.type	.L_2__STRING.5,@object
	.size	.L_2__STRING.5,2
	.space 2	# pad
	.align 4
.L_2__STRING.6:
	.byte	37
	.byte	97
	.byte	0
	.type	.L_2__STRING.6,@object
	.size	.L_2__STRING.6,3
	.space 1	# pad
	.align 4
.L_2__STRING.8:
	.byte	114
	.byte	101
	.byte	97
	.byte	100
	.byte	32
	.byte	100
	.byte	97
	.byte	116
	.byte	97
	.byte	32
	.byte	102
	.byte	105
	.byte	108
	.byte	101
	.byte	0
	.type	.L_2__STRING.8,@object
	.size	.L_2__STRING.8,15
	.space 1	# pad
	.align 4
.L_2__STRING.0:
	.byte	114
	.byte	0
	.type	.L_2__STRING.0,@object
	.size	.L_2__STRING.0,2
	.space 2	# pad
	.align 4
.L_2__STRING.1:
	.byte	111
	.byte	112
	.byte	101
	.byte	110
	.byte	32
	.byte	100
	.byte	97
	.byte	116
	.byte	97
	.byte	32
	.byte	102
	.byte	105
	.byte	108
	.byte	101
	.byte	0
	.type	.L_2__STRING.1,@object
	.size	.L_2__STRING.1,15
	.section .rodata.str1.32, "aMS",@progbits,1
	.align 32
	.align 4
.L_2__STRING.7:
	.byte	68
	.byte	97
	.byte	116
	.byte	97
	.byte	32
	.byte	102
	.byte	105
	.byte	108
	.byte	101
	.byte	32
	.byte	109
	.byte	117
	.byte	115
	.byte	116
	.byte	32
	.byte	98
	.byte	101
	.byte	32
	.byte	105
	.byte	110
	.byte	105
	.byte	116
	.byte	105
	.byte	97
	.byte	108
	.byte	105
	.byte	122
	.byte	101
	.byte	100
	.byte	32
	.byte	33
	.byte	10
	.byte	0
	.type	.L_2__STRING.7,@object
	.size	.L_2__STRING.7,33
	.data
	.section .note.GNU-stack, ""
// -- Begin DWARF2 SEGMENT .eh_frame
	.section .eh_frame,"a",@progbits
.eh_frame_seg:
	.align 8
	.4byte 0x00000014
	.8byte 0x7801000100000000
	.8byte 0x0000019008070c10
	.4byte 0x00000000
	.4byte 0x00000024
	.4byte 0x0000001c
	.8byte ..___tag_value_close_data_file.1
	.8byte ..___tag_value_close_data_file.5-..___tag_value_close_data_file.1
	.byte 0x04
	.4byte ..___tag_value_close_data_file.3-..___tag_value_close_data_file.1
	.4byte 0x0410070c
	.4byte ..___tag_value_close_data_file.4-..___tag_value_close_data_file.3
	.2byte 0x070c
	.byte 0x08
	.4byte 0x0000006c
	.4byte 0x00000044
	.8byte ..___tag_value_print_array_float.6
	.8byte ..___tag_value_print_array_float.22-..___tag_value_print_array_float.6
	.byte 0x04
	.4byte ..___tag_value_print_array_float.8-..___tag_value_print_array_float.6
	.4byte 0x070c028c
	.2byte 0x0410
	.4byte ..___tag_value_print_array_float.10-..___tag_value_print_array_float.8
	.4byte 0x070c038d
	.2byte 0x0418
	.4byte ..___tag_value_print_array_float.12-..___tag_value_print_array_float.10
	.4byte 0x0430070c
	.4byte ..___tag_value_print_array_float.13-..___tag_value_print_array_float.12
	.4byte 0x0e09068e
	.2byte 0x040e
	.4byte ..___tag_value_print_array_float.15-..___tag_value_print_array_float.13
	.2byte 0x068e
	.byte 0x04
	.4byte ..___tag_value_print_array_float.16-..___tag_value_print_array_float.15
	.4byte 0x040e0e09
	.4byte ..___tag_value_print_array_float.17-..___tag_value_print_array_float.16
	.4byte 0x0918070c
	.2byte 0x0d0d
	.byte 0x04
	.4byte ..___tag_value_print_array_float.19-..___tag_value_print_array_float.17
	.4byte 0x0910070c
	.2byte 0x0c0c
	.byte 0x04
	.4byte ..___tag_value_print_array_float.21-..___tag_value_print_array_float.19
	.8byte 0x000000000008070c
	.4byte 0x0000006c
	.4byte 0x000000b4
	.8byte ..___tag_value_print_array_double.23
	.8byte ..___tag_value_print_array_double.39-..___tag_value_print_array_double.23
	.byte 0x04
	.4byte ..___tag_value_print_array_double.25-..___tag_value_print_array_double.23
	.4byte 0x070c028c
	.2byte 0x0410
	.4byte ..___tag_value_print_array_double.27-..___tag_value_print_array_double.25
	.4byte 0x070c038d
	.2byte 0x0418
	.4byte ..___tag_value_print_array_double.29-..___tag_value_print_array_double.27
	.4byte 0x0430070c
	.4byte ..___tag_value_print_array_double.30-..___tag_value_print_array_double.29
	.4byte 0x0e09068e
	.2byte 0x040e
	.4byte ..___tag_value_print_array_double.32-..___tag_value_print_array_double.30
	.2byte 0x068e
	.byte 0x04
	.4byte ..___tag_value_print_array_double.33-..___tag_value_print_array_double.32
	.4byte 0x040e0e09
	.4byte ..___tag_value_print_array_double.34-..___tag_value_print_array_double.33
	.4byte 0x0918070c
	.2byte 0x0d0d
	.byte 0x04
	.4byte ..___tag_value_print_array_double.36-..___tag_value_print_array_double.34
	.4byte 0x0910070c
	.2byte 0x0c0c
	.byte 0x04
	.4byte ..___tag_value_print_array_double.38-..___tag_value_print_array_double.36
	.8byte 0x000000000008070c
	.4byte 0x0000006c
	.4byte 0x00000124
	.8byte ..___tag_value_print_array_long.40
	.8byte ..___tag_value_print_array_long.56-..___tag_value_print_array_long.40
	.byte 0x04
	.4byte ..___tag_value_print_array_long.42-..___tag_value_print_array_long.40
	.4byte 0x070c028c
	.2byte 0x0410
	.4byte ..___tag_value_print_array_long.44-..___tag_value_print_array_long.42
	.4byte 0x070c038d
	.2byte 0x0418
	.4byte ..___tag_value_print_array_long.46-..___tag_value_print_array_long.44
	.4byte 0x0430070c
	.4byte ..___tag_value_print_array_long.47-..___tag_value_print_array_long.46
	.4byte 0x0e09068e
	.2byte 0x040e
	.4byte ..___tag_value_print_array_long.49-..___tag_value_print_array_long.47
	.2byte 0x068e
	.byte 0x04
	.4byte ..___tag_value_print_array_long.50-..___tag_value_print_array_long.49
	.4byte 0x040e0e09
	.4byte ..___tag_value_print_array_long.51-..___tag_value_print_array_long.50
	.4byte 0x0918070c
	.2byte 0x0d0d
	.byte 0x04
	.4byte ..___tag_value_print_array_long.53-..___tag_value_print_array_long.51
	.4byte 0x0910070c
	.2byte 0x0c0c
	.byte 0x04
	.4byte ..___tag_value_print_array_long.55-..___tag_value_print_array_long.53
	.8byte 0x000000000008070c
	.4byte 0x000000bc
	.4byte 0x00000194
	.8byte ..___tag_value_init_data_long.57
	.8byte ..___tag_value_init_data_long.91-..___tag_value_init_data_long.57
	.byte 0x04
	.4byte ..___tag_value_init_data_long.59-..___tag_value_init_data_long.57
	.4byte 0x070c028c
	.2byte 0x0410
	.4byte ..___tag_value_init_data_long.61-..___tag_value_init_data_long.59
	.4byte 0x0430070c
	.4byte ..___tag_value_init_data_long.62-..___tag_value_init_data_long.61
	.4byte 0x0d09048d
	.2byte 0x040d
	.4byte ..___tag_value_init_data_long.64-..___tag_value_init_data_long.62
	.4byte 0x0e09058e
	.2byte 0x040e
	.4byte ..___tag_value_init_data_long.66-..___tag_value_init_data_long.64
	.4byte 0x0f09068f
	.2byte 0x040f
	.4byte ..___tag_value_init_data_long.68-..___tag_value_init_data_long.66
	.4byte 0x058e048d
	.2byte 0x068f
	.byte 0x04
	.4byte ..___tag_value_init_data_long.71-..___tag_value_init_data_long.68
	.4byte 0x040e0e09
	.4byte ..___tag_value_init_data_long.72-..___tag_value_init_data_long.71
	.4byte 0x040d0d09
	.4byte ..___tag_value_init_data_long.73-..___tag_value_init_data_long.72
	.4byte 0x040f0f09
	.4byte ..___tag_value_init_data_long.74-..___tag_value_init_data_long.73
	.4byte 0x0910070c
	.2byte 0x0c0c
	.byte 0x04
	.4byte ..___tag_value_init_data_long.76-..___tag_value_init_data_long.74
	.4byte 0x0408070c
	.4byte ..___tag_value_init_data_long.77-..___tag_value_init_data_long.76
	.8byte 0x8e048d028c30070c
	.4byte 0x04068f05
	.4byte ..___tag_value_init_data_long.82-..___tag_value_init_data_long.77
	.4byte 0x040d0d09
	.4byte ..___tag_value_init_data_long.83-..___tag_value_init_data_long.82
	.4byte 0x040e0e09
	.4byte ..___tag_value_init_data_long.84-..___tag_value_init_data_long.83
	.4byte 0x040f0f09
	.4byte ..___tag_value_init_data_long.85-..___tag_value_init_data_long.84
	.4byte 0x058e048d
	.2byte 0x068f
	.byte 0x04
	.4byte ..___tag_value_init_data_long.88-..___tag_value_init_data_long.85
	.8byte 0x0f090e0e090d0d09
	.2byte 0x000f
	.4byte 0x000000bc
	.4byte 0x00000254
	.8byte ..___tag_value_init_data_double.92
	.8byte ..___tag_value_init_data_double.126-..___tag_value_init_data_double.92
	.byte 0x04
	.4byte ..___tag_value_init_data_double.94-..___tag_value_init_data_double.92
	.4byte 0x070c028c
	.2byte 0x0410
	.4byte ..___tag_value_init_data_double.96-..___tag_value_init_data_double.94
	.4byte 0x0430070c
	.4byte ..___tag_value_init_data_double.97-..___tag_value_init_data_double.96
	.4byte 0x0d09048d
	.2byte 0x040d
	.4byte ..___tag_value_init_data_double.99-..___tag_value_init_data_double.97
	.4byte 0x0e09058e
	.2byte 0x040e
	.4byte ..___tag_value_init_data_double.101-..___tag_value_init_data_double.99
	.4byte 0x0f09068f
	.2byte 0x040f
	.4byte ..___tag_value_init_data_double.103-..___tag_value_init_data_double.101
	.4byte 0x058e048d
	.2byte 0x068f
	.byte 0x04
	.4byte ..___tag_value_init_data_double.106-..___tag_value_init_data_double.103
	.4byte 0x040e0e09
	.4byte ..___tag_value_init_data_double.107-..___tag_value_init_data_double.106
	.4byte 0x040d0d09
	.4byte ..___tag_value_init_data_double.108-..___tag_value_init_data_double.107
	.4byte 0x040f0f09
	.4byte ..___tag_value_init_data_double.109-..___tag_value_init_data_double.108
	.4byte 0x0910070c
	.2byte 0x0c0c
	.byte 0x04
	.4byte ..___tag_value_init_data_double.111-..___tag_value_init_data_double.109
	.4byte 0x0408070c
	.4byte ..___tag_value_init_data_double.112-..___tag_value_init_data_double.111
	.8byte 0x8e048d028c30070c
	.4byte 0x04068f05
	.4byte ..___tag_value_init_data_double.117-..___tag_value_init_data_double.112
	.4byte 0x040d0d09
	.4byte ..___tag_value_init_data_double.118-..___tag_value_init_data_double.117
	.4byte 0x040e0e09
	.4byte ..___tag_value_init_data_double.119-..___tag_value_init_data_double.118
	.4byte 0x040f0f09
	.4byte ..___tag_value_init_data_double.120-..___tag_value_init_data_double.119
	.4byte 0x058e048d
	.2byte 0x068f
	.byte 0x04
	.4byte ..___tag_value_init_data_double.123-..___tag_value_init_data_double.120
	.8byte 0x0f090e0e090d0d09
	.2byte 0x000f
	.4byte 0x000000bc
	.4byte 0x00000314
	.8byte ..___tag_value_init_data_float.127
	.8byte ..___tag_value_init_data_float.161-..___tag_value_init_data_float.127
	.byte 0x04
	.4byte ..___tag_value_init_data_float.129-..___tag_value_init_data_float.127
	.4byte 0x070c028c
	.2byte 0x0410
	.4byte ..___tag_value_init_data_float.131-..___tag_value_init_data_float.129
	.4byte 0x0430070c
	.4byte ..___tag_value_init_data_float.132-..___tag_value_init_data_float.131
	.4byte 0x0d09048d
	.2byte 0x040d
	.4byte ..___tag_value_init_data_float.134-..___tag_value_init_data_float.132
	.4byte 0x0e09058e
	.2byte 0x040e
	.4byte ..___tag_value_init_data_float.136-..___tag_value_init_data_float.134
	.4byte 0x0f09068f
	.2byte 0x040f
	.4byte ..___tag_value_init_data_float.138-..___tag_value_init_data_float.136
	.4byte 0x058e048d
	.2byte 0x068f
	.byte 0x04
	.4byte ..___tag_value_init_data_float.141-..___tag_value_init_data_float.138
	.4byte 0x040e0e09
	.4byte ..___tag_value_init_data_float.142-..___tag_value_init_data_float.141
	.4byte 0x040d0d09
	.4byte ..___tag_value_init_data_float.143-..___tag_value_init_data_float.142
	.4byte 0x040f0f09
	.4byte ..___tag_value_init_data_float.144-..___tag_value_init_data_float.143
	.4byte 0x0910070c
	.2byte 0x0c0c
	.byte 0x04
	.4byte ..___tag_value_init_data_float.146-..___tag_value_init_data_float.144
	.4byte 0x0408070c
	.4byte ..___tag_value_init_data_float.147-..___tag_value_init_data_float.146
	.8byte 0x8e048d028c30070c
	.4byte 0x04068f05
	.4byte ..___tag_value_init_data_float.152-..___tag_value_init_data_float.147
	.4byte 0x040d0d09
	.4byte ..___tag_value_init_data_float.153-..___tag_value_init_data_float.152
	.4byte 0x040e0e09
	.4byte ..___tag_value_init_data_float.154-..___tag_value_init_data_float.153
	.4byte 0x040f0f09
	.4byte ..___tag_value_init_data_float.155-..___tag_value_init_data_float.154
	.4byte 0x058e048d
	.2byte 0x068f
	.byte 0x04
	.4byte ..___tag_value_init_data_float.158-..___tag_value_init_data_float.155
	.8byte 0x0f090e0e090d0d09
	.2byte 0x000f
	.4byte 0x000000bc
	.4byte 0x000003d4
	.8byte ..___tag_value_init_data_gen.162
	.8byte ..___tag_value_init_data_gen.196-..___tag_value_init_data_gen.162
	.byte 0x04
	.4byte ..___tag_value_init_data_gen.164-..___tag_value_init_data_gen.162
	.4byte 0x070c028c
	.2byte 0x0410
	.4byte ..___tag_value_init_data_gen.166-..___tag_value_init_data_gen.164
	.4byte 0x0430070c
	.4byte ..___tag_value_init_data_gen.167-..___tag_value_init_data_gen.166
	.4byte 0x0d09048d
	.2byte 0x040d
	.4byte ..___tag_value_init_data_gen.169-..___tag_value_init_data_gen.167
	.4byte 0x0e09058e
	.2byte 0x040e
	.4byte ..___tag_value_init_data_gen.171-..___tag_value_init_data_gen.169
	.4byte 0x0f09068f
	.2byte 0x040f
	.4byte ..___tag_value_init_data_gen.173-..___tag_value_init_data_gen.171
	.4byte 0x058e048d
	.2byte 0x068f
	.byte 0x04
	.4byte ..___tag_value_init_data_gen.176-..___tag_value_init_data_gen.173
	.4byte 0x040e0e09
	.4byte ..___tag_value_init_data_gen.177-..___tag_value_init_data_gen.176
	.4byte 0x040d0d09
	.4byte ..___tag_value_init_data_gen.178-..___tag_value_init_data_gen.177
	.4byte 0x040f0f09
	.4byte ..___tag_value_init_data_gen.179-..___tag_value_init_data_gen.178
	.4byte 0x0910070c
	.2byte 0x0c0c
	.byte 0x04
	.4byte ..___tag_value_init_data_gen.181-..___tag_value_init_data_gen.179
	.4byte 0x0408070c
	.4byte ..___tag_value_init_data_gen.182-..___tag_value_init_data_gen.181
	.8byte 0x8e048d028c30070c
	.4byte 0x04068f05
	.4byte ..___tag_value_init_data_gen.187-..___tag_value_init_data_gen.182
	.4byte 0x040d0d09
	.4byte ..___tag_value_init_data_gen.188-..___tag_value_init_data_gen.187
	.4byte 0x040e0e09
	.4byte ..___tag_value_init_data_gen.189-..___tag_value_init_data_gen.188
	.4byte 0x040f0f09
	.4byte ..___tag_value_init_data_gen.190-..___tag_value_init_data_gen.189
	.4byte 0x058e048d
	.2byte 0x068f
	.byte 0x04
	.4byte ..___tag_value_init_data_gen.193-..___tag_value_init_data_gen.190
	.8byte 0x0f090e0e090d0d09
	.2byte 0x000f
	.4byte 0x000000d4
	.4byte 0x00000494
	.8byte ..___tag_value__init_data.197
	.8byte ..___tag_value__init_data.236-..___tag_value__init_data.197
	.byte 0x04
	.4byte ..___tag_value__init_data.199-..___tag_value__init_data.197
	.4byte 0x070c028c
	.2byte 0x0410
	.4byte ..___tag_value__init_data.201-..___tag_value__init_data.199
	.4byte 0x0430070c
	.4byte ..___tag_value__init_data.202-..___tag_value__init_data.201
	.4byte 0x0d09048d
	.2byte 0x040d
	.4byte ..___tag_value__init_data.204-..___tag_value__init_data.202
	.4byte 0x0e09058e
	.2byte 0x040e
	.4byte ..___tag_value__init_data.206-..___tag_value__init_data.204
	.4byte 0x0f09068f
	.2byte 0x040f
	.4byte ..___tag_value__init_data.208-..___tag_value__init_data.206
	.4byte 0x058e048d
	.2byte 0x068f
	.byte 0x04
	.4byte ..___tag_value__init_data.211-..___tag_value__init_data.208
	.4byte 0x040e0e09
	.4byte ..___tag_value__init_data.212-..___tag_value__init_data.211
	.4byte 0x040d0d09
	.4byte ..___tag_value__init_data.213-..___tag_value__init_data.212
	.4byte 0x040f0f09
	.4byte ..___tag_value__init_data.214-..___tag_value__init_data.213
	.4byte 0x0910070c
	.2byte 0x0c0c
	.byte 0x04
	.4byte ..___tag_value__init_data.216-..___tag_value__init_data.214
	.4byte 0x0408070c
	.4byte ..___tag_value__init_data.217-..___tag_value__init_data.216
	.8byte 0x8e048d028c30070c
	.4byte 0x04068f05
	.4byte ..___tag_value__init_data.222-..___tag_value__init_data.217
	.4byte 0x040d0d09
	.4byte ..___tag_value__init_data.223-..___tag_value__init_data.222
	.4byte 0x040e0e09
	.4byte ..___tag_value__init_data.224-..___tag_value__init_data.223
	.4byte 0x040f0f09
	.4byte ..___tag_value__init_data.225-..___tag_value__init_data.224
	.4byte 0x058e048d
	.2byte 0x068f
	.byte 0x04
	.4byte ..___tag_value__init_data.228-..___tag_value__init_data.225
	.4byte 0x0910070c
	.2byte 0x0c0c
	.byte 0x04
	.4byte ..___tag_value__init_data.230-..___tag_value__init_data.228
	.4byte 0x0408070c
	.4byte ..___tag_value__init_data.231-..___tag_value__init_data.230
	.8byte 0x0d0d09028c30070c
	.4byte 0x090e0e09
	.2byte 0x0f0f
	.byte 0x00
	.4byte 0x0000003c
	.4byte 0x0000056c
	.8byte ..___tag_value_init_data_file.237
	.8byte ..___tag_value_init_data_file.244-..___tag_value_init_data_file.237
	.byte 0x04
	.4byte ..___tag_value_init_data_file.239-..___tag_value_init_data_file.237
	.4byte 0x0410070c
	.4byte ..___tag_value_init_data_file.240-..___tag_value_init_data_file.239
	.4byte 0x0408070c
	.4byte ..___tag_value_init_data_file.241-..___tag_value_init_data_file.240
	.4byte 0x0410070c
	.4byte ..___tag_value_init_data_file.242-..___tag_value_init_data_file.241
	.4byte 0x0408070c
	.4byte ..___tag_value_init_data_file.243-..___tag_value_init_data_file.242
	.2byte 0x070c
	.byte 0x10
# End
