#
# $RCSfile: makefile_macros.GNU,v $ (version $Revision$)
# $Date: 1996/08/10 15:46:07 $, 
#
# this file contains macro definitions 

AR	= gar
ARFLAGS	= rv
CC	= gcc
CFLAGS	= -O2 -ansi -g -pedantic -Wall -msupersparc -pipe
CPPFLAGS= -D__USE_FIXED_PROTOTYPES__ -I$(NEWGEN_INCLUDEDIR) -I$(LINEAR_INCLUDEDIR) -I$(PIPS_INCLUDEDIR) -I$(PIPS_EXTEDIR) 
LD	= $(CC) 
LDFLAGS	= -g -L./$(ARCH) -L$(PIPS_LIBDIR)/$(ARCH) -L$(NEWGEN_LIBDIR)/$(ARCH) -L$(LINEAR_LIBDIR)/$(ARCH) -L$(PIPS_EXTEDIR)/$(ARCH) -L$(PIPS_EXTEDIR)
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
