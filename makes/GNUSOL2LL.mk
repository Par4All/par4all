# $Id$

include $(ROOT)/makes/GNULL.mk

AR	= ar
RANLIB	= ranlib
CFLAGS  += -msupersparc
CANSI	=
INSTALL	= INSTALL=cp $(ROOT)/makes/install.sh
