#
# $RCSfile: makefile_macros.GNU,v $ (version $Revision$)
# $Date: 1996/08/09 10:58:18 $, 
#
# this file contains macro definitions 

GLIBC	= /usr5/gnu/SPARC/glibc
GCCDIR	= $(GCC_EXEC_PREFIX)sparc-sun-sunos4.1.4/2.7.2.f.1

AR	= gar
ARFLAGS	= rv
CC	= gcc
CFLAGS	= -O2 -ansi -g -pedantic -Wall -msupersparc -pipe
CPPFLAGS= -D__USE_FIXED_PROTOTYPES__ -nostdinc -I$(NEWGEN_INCLUDEDIR) -I$(LINEAR_INCLUDEDIR) -I$(PIPS_INCLUDEDIR) -I$(PIPS_EXTEDIR) -I$(GLIBC)/include -I$(GCCDIR)/include -I/usr/include 
LD	= $(CC) 
LDFLAGS	= -g -nostdlib $(GLIBC)/lib/crt0.o -L./$(ARCH) -L$(PIPS_LIBDIR)/$(ARCH) -L$(NEWGEN_LIBDIR)/$(ARCH) -L$(LINEAR_LIBDIR)/$(ARCH) -L$(PIPS_EXTEDIR)/$(ARCH) -L$(PIPS_EXTEDIR) -L$(GLIBC)/lib -L$(GCCDIR)/lib -L/usr/lib -lc -lgcc
RANLIB	= granlib
LEX	= flex
LFLAGS	= 
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

# end of $RCSfile: makefile_macros.GNU,v $
#
