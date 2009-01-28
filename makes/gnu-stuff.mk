#
# $Id$
#
# macros related to gnu compilation.
#

AR	= gar
ARFLAGS	= rv

CC	= gcc
CANSI	= -ansi -pedantic-errors
CFLAGS	= -g -O2 -Wall -W -pipe
# ??? -MG
CMKDEP	= -MM

LD	= $(CC)
RANLIB	= granlib

ifdef PIPS_F77
	FC	= $(PIPS_F77)
else
	FC	= f77
endif

FFLAGS	= -O2 -g -Wimplicit -pipe

LDFLAGS += -g

# putenv() => svid
# getwd()  => bsd
# getopt() => posix2

CPPFLAGS += \
	-D__USE_FIXED_PROTOTYPES__

LEX	= flex
LFLAGS	= 

LINT	= lint
LINTFLAGS= -habxu
