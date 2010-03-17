# $Id$

include $(ROOT)/makes/longlong.mk
include $(ROOT)/makes/DEFAULT.mk

CFLAGS	= -fast -xtarget=ultra1/140 -xarch=v8plusa -g
CMKDEP	= -xM

# mouais.

FFLAGS	= -fast -u
LEX	= flex
LFLAGS	= 
