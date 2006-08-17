# $Id$

AR	= ar
ARFLAGS	= -rv
CC	= cc -T cray-t3d
CFLAGS	= -O 3
CMKDEP	= -M
CPPFLAGS= -I$(NEWGEN_ROOT)/Include -I$(LINEAR_ROOT)/Include \
		-I$(PIPS_ROOT)/Include -I$(EXTERN_ROOT)/Include
LD	= $(CC)
LDFLAGS	= -g -L./$(ARCH) -L$(PIPS_ROOT)/Lib/$(ARCH) \
		-L$(NEWGEN_ROOT)/Lib/$(ARCH) -L$(LINEAR_ROOT)/Lib/$(ARCH) \
		-L$(EXTERN_ROOT)/Lib/$(ARCH)
RANLIB	= :
LEX	= lex
LFLAGS	=
FC	= cf77 -C cray-t3d
# -g suppresses all optimizations!
FFLAGS	= -O 1 -O scalar3
LINT	= lint
LINTFLAGS= -habxu
YACC	= yacc
YFLAGS	=
ETAGS	= etags
TAR	= tar
ZIP	= compress
DIFF	= diff
# the cray m4 results in some internal stack overflow on hpfc runtime...
M4	= gm4
M4FLAGS	=
LX2HTML	= latex2html
L2HFLAGS= -link 8 -split 5
LATEX	= latex
BIBTEX	= bibtex
MAKEIDX	= makeindex
DVIPS	= dvips
RMAN	= rman
