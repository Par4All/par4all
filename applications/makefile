#
# $Id: makefile,v 1.1 2002/10/15 07:38:40 kienhuis Exp $
#

ROOT  = ..

# C files in this directory
SRCS = \
	testlib.c \
	polytest.c \
	c2p.c \
	r2p.c \
	findv.c \
	pp.c \
	disjoint_union_sep.c \
	disjoint_union_adj.c \
	testehrhart.c \
	verif_ehrhart.c \
	Zpolytest.c \
	example.c

POLY_EXEC= $(SRCS:%.c=%.exec)

EXTRA_DEFINES=-DEMAIN -DEP_EVALUATION

# Include the shared compilation rules
include $(ROOT)/vars.mk
include $(ROOT)/default_functions.mk

