# $Id$

include $(ROOT)/makes/GNU.mk

# The option -pipe does not work on OSF1:
CFLAGS = -g -O2 -Wall
