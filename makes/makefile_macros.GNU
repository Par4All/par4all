#
# $Id$
#
# $Log: makefile_macros.GNU,v $
# Revision 1.31  1997/05/14 15:54:31  coelho
# *** empty log message ***
#
# Revision 1.30  1997/05/14 08:33:54  coelho
# use bsd files...
#
# Revision 1.29  1997/04/15 12:10:22  coelho
# -pedantic-errors...
#
# Revision 1.28  1997/03/26 17:25:59  coelho
# simply GNU softwares.
#
#
# this file contains macro definitions 

AR	= gar
ARFLAGS	= rv
CC	= gcc
CANSI	= -ansi -pedantic-errors
CFLAGS	= -g -O2 -Wall -pipe
CMKDEP	= -M
CPPFLAGS= \
	-D__USE_BSD \
	-D__USE_FIXED_PROTOTYPES__ \
	-I. \
	-I$(NEWGEN_ROOT)/Include \
	-I$(LINEAR_ROOT)/Include \
	-I$(PIPS_ROOT)/Include \
	-I$(EXTERN_ROOT)/Include 
LD	= $(CC) 
LDFLAGS	= \
	-g \
	-L./$(ARCH) \
	-L$(PIPS_ROOT)/Lib/$(ARCH) \
	-L$(NEWGEN_ROOT)/Lib/$(ARCH) \
	-L$(LINEAR_ROOT)/Lib/$(ARCH) \
	-L$(EXTERN_ROOT)/Lib/$(ARCH)
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
M4FLAGS	=
LX2HTML	= latex2html
L2HFLAGS= -link 8 -split 5
LATEX	= latex
BIBTEX	= bibtex
RMAN	= rman
MAKEIDX	= makeindex
DVIPS	= dvips

# end of it!
#
