#
# $Id$
#

AR	= ar
ARFLAGS	= rv
CC	= cc
CFLAGS	= -O -g
CMKDEP	= -M
LD	= $(CC)
RANLIB	= ranlib
LEX	= flex
LFLAGS	=
FC	= f77
FFLAGS	= -O -g
LINT	= lint
LINTFLAGS= -habxu

# may need to be overwritten
CC_VERSION	= $(CC) --version | head -1

# The parser can no longer be compiled with yacc...
YACC	= bison
YFLAGS	= -y

PROTO   = cproto
PRFLAGS    = -evcf2

# A dummy target for the flymake-mode in Emacs:
check-syntax:
	$(COMPILE) -o nul.o -S ${CHK_SOURCES}

# end of it!
#
