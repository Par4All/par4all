#
# $Id$
#
# macros related to gnu compilation.
#

AR	= gar
ARFLAGS	= rv
CC	= gcc
CANSI	= -ansi -pedantic-errors
CFLAGS	= -g -O2 -Wall -pipe
CMKDEP	= -M
LD	= $(CC) 
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

LDFLAGS += -g

# putenv() => svid
# getwd()  => bsd
# getopt() => posix2

CPPFLAGS += \
	-D_POSIX_C_SOURCE=2 \
	-D_BSD_SOURCE \
	-D_SVID_SOURCE \
	-D__USE_FIXED_PROTOTYPES__

