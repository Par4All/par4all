# $Id$

include $(ROOT)/makes/GNU.mk

CANSI	= -ansi -pedantic
CFLAGS	+= -msupersparc
L2HFLAGS= -link 8 -split 5
DIFF	= diff
