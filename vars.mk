
# @Author: Bart Kienhuis
# $Id: vars.mk,v 1.24 2002/10/16 13:57:29 olaru Exp $

VERSION = 5.11.1

# NOTE: Don't edit this file if it is called vars.mk, instead
# edit vars.mk.in, which is read by configure

# Default top-level directory.
prefix =	/.amd/terre/export/home/terre/d03/r2d2/olaru/TEMP/Polylib/Polylib

# Usually the same as prefix. 
# exec_prefix is part of the autoconf standard.
exec_prefix =	${prefix}

# Source directory we are building from.
srcdir =	.

# Directory in which to install scripts
BIN_INSTALL_DIR =	$(exec_prefix)/bin

# c compiler flags and defines
CC	             	= gcc
CFLAGS                  = -g -O2
EXTRA_FLAGS             = 
DEFINES         	=  -DSTDC_HEADERS=1 -DHAVE_LIMITS_H=1 -DHAVE_UNISTD_H=1 -DSIZEOF_INT=4 -DSIZEOF_LONG_INT=4 -DSIZEOF_LONG_LONG_INT=8  

# Linker flags and defines, both static and shared lib
LD 	        	= gcc
LDFLAGS                 = 

# Additional libraries that need to be linked
LIBS                    = 

# Additional tools needed when making libraries
LN_S			= ln -s
RANLIB			= ranlib

SHEXT  			= so

# defines needed for arithmetic lib
INT_AFLAGS = -DLINEAR_VALUE_IS_INT
LONG_AFLAGS = -DLINEAR_VALUE_IS_LONGLONG -DLINEAR_VALUE_PROTECT_MULTIPLY			-DLINEAR_VALUE_ASSUME_SOFTWARE_IDIV
GMP_AFLAGS = -DGNUMP
INT_BITS = 32
LONG_BITS = 64
GMP_BITS = gmp

# Library type to install
INSTALL_LIB = install-static install-shared

# Commands used to install scripts and data
INSTALL =		/usr/bin/install -c
INSTALL_PROGRAM =	${INSTALL}
INSTALL_DATA =		${INSTALL} -m 644

## GNU-MP stuff
EXTRA_INCLUDES=
EXTRA_LIBS=-lgmp 

# Platform specific variables
OSTYPE	= linux-gnu
HOST    = pc
BUILD   = i686


EXEC_EXTRA_SUFFIX = 

# When compiling the tests, we need to link additional libraries
# include polylib
SHAREDLIB_FLAG          = -shared
LDCONFIG = ldconfig

LIBS_TO_BUILD = 64
EXEC_TO_BUILD = 64
BITS=64
AFLAGS=-DLINEAR_VALUE_IS_LONGLONG -DLINEAR_VALUE_PROTECT_MULTIPLY			-DLINEAR_VALUE_ASSUME_SOFTWARE_IDIV


OBJ_DIR = Obj.$(BITS).$(BUILD)-$(HOST)-$(OSTYPE)
LIB = $(OBJ_DIR)/$(PSTATIC)
EXEC_EXTRA_LIBS = $(LIB)
CFLAGS +=  $(EXTRA_INCLUDES) $(AFLAGS) $(EXTRA_FLAGS) -I $(ROOT)/include


############################################################
# Documentation Generation 
############################################################

DOXYGEN = test

############################################################
# Library Definitions
############################################################

STATIC_LIB = libpolylib$(BITS).a
SHARED_LIB = libpolylib$(BITS).$(SHEXT)

PSTATIC = $(SHARED_LIB).$(VERSION)
PSHARED = $(SHARED_LIB).$(VERSION)

############################################################
# Install Definitions
############################################################

INSTALLDIR = ${prefix}
BINDIR = $(INSTALLDIR)/bin
LIBDIR = $(INSTALLDIR)/lib
INCLUDEDIR = $(INSTALLDIR)/include
MANDIR = $(INSTALLDIR)/man
DOCSDIR = $(INSTALLDIR)/doc/packages/polylib-$(VERSION)

############################################################
# Shell script to create required directories
############################################################

mkinstalldirs = $(SHELL) ./mkinstalldirs
