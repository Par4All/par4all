#
#
#   	COMPLEXITY EVALUATION
#   	---------------------
#
# Pierre Berthomier  07-09-90
# Lei Zhou	     22-02-91
#
####### The source files directly involved in complexity are:
#	
# in $INCLUDEDIR:
#	complexity_ri.f.tex	describing `complexity' data structures
#	complexity_ri.newgen	  |
#	complexity_ri.spec	  | three files generated with NewGen
#	complexity_ri.h		  |
#
# in ~pips/Pips/Development/Lib/complexity:
#      (complexity-local.h	my local header)
#	complexity.h		automatically generated header
#				  the local one + subroutines decl.
#	comp_scan.c		subroutines that scan the RI to
#                                 count operations
#	expr_to_pnome.c		subroutines that walk the RI expressions
#				  to try to give them a polynomial form
#	comp_unstr.c		subroutines that cope with unstructured
#				  graphs of statements
#	comp_util.c		useful subroutines for evaluation
#				  of complexity
#	comp_math.c		"mathematical" operations on complexities:
#				  addition, integration, ...
#	comp_matrice.c	        matrice inversion for floating point
#
#	comp_prettyprint.c	routines for prettyprinting complexities
#				  with Fortran source code
#	polynome_ri.c		interface polynomial library / RI
#
#	main.c		        main(), to test complexity routines.
#
#
####### The usable files created are:
#
#	complexity		contains all routines and the main
#				  to become a pass of PIPS
#	libcomplexity.a		contains all but main.c
#
#######
#
# The following macros define the value of commands that are used to
# compile source code.
#
# you can add your own options behind pips default values.
# 
# example: CFLAGS= $(PIPS_CFLAGS) -DSYSTEM=BSD4.2
#
AR=		$(PIPS_AR)
ARFLAGS=	$(PIPS_ARFLAGS)
CC=		$(PIPS_CC)
CFLAGS=		$(PIPS_CFLAGS)
CPPFLAGS=	$(PIPS_CPPFLAGS)
LD=		$(PIPS_LD)
LDFLAGS=	$(PIPS_LDFLAGS)
LEX=		$(PIPS_LEX)
LFLAGS=		$(PIPS_LFLAGS)
LINT=		$(PIPS_LINT)
LINTFLAGS=	$(PIPS_LINTFLAGS)
YACC=		$(PIPS_YACC)
YFLAGS=		$(PIPS_YFLAGS)

# The following macros define your library.

# Name of the library without the .a suffix.
TARGET= 	complexity

# Name of the main program to test or use the library
MAIN=		main

# Generated from 'order_libraries' 07-09-90


# Source, header and object files used to build the library.
# Do not include the main program source file.
# Do include the .c, .h and .o extensions.

LIB_CFILES=	comp_scan.c comp_expr_to_pnome.c comp_unstr.c\
		comp_util.c comp_math.c comp_prettyprint.c polynome_ri.c\
		comp_matrice.c
LIB_HEADERS=	complexity-local.h
LIB_OBJECTS=	comp_scan.o comp_expr_to_pnome.o comp_unstr.o\
		comp_util.o comp_math.o comp_prettyprint.o polynome_ri.o\
		comp_matrice.o
### End of config.makefile

