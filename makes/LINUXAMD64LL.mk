# $Id$

include $(ROOT)/makes/GNULL.mk
include $(ROOT)/makes/linux.mk
CFLAGS	+= -m64 -mtune=generic
# temporary fix
LDFLAGS	+= -static
include $(ROOT)/makes/no_wpips.mk
