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
CMKDEP	= -M
LD	= $(CC) 
RANLIB	= granlib
FC	= g77
FFLAGS	= -O2 -g -Wimplicit -pipe

LDFLAGS += -g

# putenv() => svid
# getwd()  => bsd
# getopt() => posix2

CPPFLAGS += \
	-D__USE_FIXED_PROTOTYPES__

