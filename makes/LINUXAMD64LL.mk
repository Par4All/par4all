# $Id$

include $(ROOT)/makes/GNULL.mk
include $(ROOT)/makes/linux.mk
CFLAGS	+= -m64 -mtune=generic
include $(ROOT)/makes/no_wpips.mk
