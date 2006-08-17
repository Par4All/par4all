# $Id$

GLIBC	= /usr5/gnu/SPARC/glibc
GCCDIR	= $(GCC_EXEC_PREFIX)sparc-sun-sunos4.1.4/2.7.2.f.1

AR	= gar
ARFLAGS	= rv
CC	= gcc
CFLAGS	= -O2 -g -Wall -msupersparc -pipe
CMKDEP	= -M
CPPFLAGS= -DLINEAR_VALUE_IS_CHARS -DLINEAR_NO_OVERFLOW_CONTROL \
		-D__USE_FIXED_PROTOTYPES__ -D__USE_GNU -D__USE_BSD -nostdinc \
		-I$(NEWGEN_ROOT)/Include -I$(LINEAR_ROOT)/Include \
		-I$(PIPS_ROOT)/Include -I$(PIPS_EXTEDIR) \
		-I$(GLIBC)/include -I$(GCCDIR)/include -I/usr/include
LD	= $(CC) 
LDFLAGS	= -g -nostdlib $(GLIBC)/lib/crt0.o -L./$(ARCH) \
		-L$(PIPS_ROOT)/Lib/$(ARCH) -L$(NEWGEN_ROOT)/Lib/$(ARCH) \
		-L$(LINEAR_ROOT)/Lib/$(ARCH) -L$(PIPS_EXTEDIR)/$(ARCH) \
		-L$(PIPS_EXTEDIR) -L$(GLIBC)/lib -L$(GCCDIR)/lib \
		-L/usr/lib -lc -lgcc
RANLIB	= granlib
LEX	= flex
LFLAGS	= -l
FC	= g77
FFLAGS	= -O2 -g -Wimplicit -pipe
LINT	= lint
LINTFLAGS= -habxu
YACC	= bison
YFLAGS	= -y
PROTO	= cproto
PRFLAGS	= -evcf2
ETAGS	= etags
TAR	= gtar
ZIP	= gzip -v9
DIFF	= gdiff
M4	= gm4
M4FLAGS	=
LX2HTML	= :
LATEX	= :
MAKEIDX	= :
DVIPS	= :
RMAN	= :
