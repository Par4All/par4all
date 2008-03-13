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

# The parser can no longer be compiled with yacc...
YACC	= bison
YFLAGS	= -y

PROTO   = cproto
PRFLAGS    = -evcf2

# A dummy target for the flymake-mode in Emacs:
check-syntax:
	gcc -o nul -S ${CHK_SOURCES}

# end of it!
#
