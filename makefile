#
# @author Bart Kienhuis
# $Id: makefile,v 1.2 2002/10/18 14:03:06 loechner Exp $
#
# Top-level makefile for polylib. 

ROOT	= .

SUBDIRS = source

APPDIRS = applications

TESTDIRS = Test

# default: build the executables
.PHONY: execs

include $(ROOT)/default_functions.mk
