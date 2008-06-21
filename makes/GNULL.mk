# $Id$

include $(ROOT)/makes/DEFAULT.mk
include $(ROOT)/makes/gnu.mk
include $(ROOT)/makes/longlong.mk

# -ansi -petantic-errors and long long int is not a good idea.
CANSI=

# must force definition of long long int constants?
# CPPFLAGS+= -D__GNU_LIBRARY__ -D__USE_GNU
