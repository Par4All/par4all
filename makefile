#
# @author Bart Kienhuis
# $Id: makefile,v 1.1 2002/10/14 14:08:18 olaru Exp $
#
# Top-level makefile for polylib. 

ROOT	= .

SUBDIRS = source

APPDIRS = applications

TESTDIRS = Test

package: vars.mk

include $(ROOT)/default_functions.mk
