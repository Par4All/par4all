#
# $RCSfile: makefile_macros.GNU,v $ (version $Revision$)
# $Date: 1996/07/15 14:01:55 $, 
#
# this file contains macro definitions 

AR	= gar
ARFLAGS	= rv
CC	= gcc
CFLAGS	= -O2 -g -ansi -pedantic -Wall -msupersparc -pipe
CPPFLAGS= -D__USE_FIXED_PROTOTYPES__ -I/usr5/gnu/SPARC/include -I$(NEWGEN_INCLUDEDIR) -I$(LINEAR_INCLUDEDIR) -I$(PIPS_INCLUDEDIR) -I$(PIPS_EXTEDIR)
LD	= $(CC) 
LDFLAGS	= -g -nostdlib /usr5/gnu/SPARC/lib/crt0.o -L/usr5/gnu/SPARC/lib -L./$(ARCH) -L$(PIPS_LIBDIR)/$(ARCH) -L$(NEWGEN_LIBDIR)/$(ARCH) -L$(LINEAR_LIBDIR)/$(ARCH) -L$(PIPS_EXTEDIR)/$(ARCH) -lc -lgcc
RANLIB	= granlib
LEX	= flex
LFLAGS	= -l
FC	= g77
FFLAGS	= -O2 -g -Wimplicit -pipe
LINT	= lint
LINTFLAGS= -habxu
YACC	= bison
YFLAGS	= -y
PROTOIZE= cproto -vcf2
ETAGS	= etags
TAR	= gtar
ZIP	= gzip -v9
DIFF	= gdiff
M4	= gm4

# end of $RCSfile: makefile_macros.GNU,v $
#
