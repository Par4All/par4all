# $Id$

include $(ROOT)/makes/CRAY-T3D.mk

FC	= TARGET=cray-t3d f90
FFLAGS	= -O 3 -O unroll2 -e I
CC	= TARGET=cray-t3d cc
