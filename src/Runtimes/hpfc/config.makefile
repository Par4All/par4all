#
# $RCSfile: config.makefile,v $ version $Revision$
# ($Date: 1996/08/22 14:51:03 $, )
#
# depends on 
# + PVM_ARCH 
# + PVM_ROOT
# + USE_PVMe
# + USE_GNU

# this settings may be edited
# I prefer to use free gnu compilers...
USE_GNU=1

# additional defs for m4
M4FLAGS	+= -D DEMO
M4FLAGS	+= -D DIRECT
M4FLAGS	+= -D USE_GNU
# M4FLAGS	+= -D DEBUG
# M4FLAGS	+= -D USE_PVMe

#############################################################################

SCRIPTS =	hpfc_llcmd \
		hpfc_add_warning \
		hpfc_generate_h \
		hpfc_generate_init

#
# Default compilers and options

CC		= gcc
CFLAGS		= -O2 -pipe -ansi -pedantic -Wall
CPPFLAGS	= -D__USE_FIXED_PROTOTYPES__
FC 		= f77
FFLAGS		= -O2 -u

#
# others

ifeq ($(PVM_ARCH),CM5)
#
# Thinking Machine CM5
#
FFLAGS	+= -Nx1000

CMMD_INDIR	= /net/cm5/CMSW/CMMD/cmmd-3.2/include
CMMD_F77_H	= cmmd_fort.h
#
endif

ifeq ($(PVM_ARCH),SUN4)
#
# SUN - SOLARIS 1 (SUNOS 4)
#
FFLAGS		= -fast -u
#
endif

ifeq ($(PVM_ARCH),SUNMP)
#
# SUN - SOLARIS 2
#
CC	= cc
CFLAGS	= -O2
FC	= f77
FFLAGS	= -fast -u
#
endif

ifeq ($(PVM_ARCH),SUN4SOL2)
#
# SUN - SOLARIS 2
#
CC	= cc
CFLAGS	= -O2
FC	= f77
FFLAGS	= -fast -u
#
endif

ifeq ($(PVM_ARCH),RS6K)
#
# IBM compilers on RS6K/SPx...
#
FC	= xlf
FFLAGS	= -O2 -u
CC	= xlc
CFLAGS	= -O2
#
endif

ifeq ($(PVM_ARCH),ALPHA)
#
# DEC alpha
#
FC	= f77
FFLAGS	= -fast -u
CC	= cc
CFLAGS	= -O4
#
endif

# ??? this env. dependence is not very convincing...
ifdef USE_GNU
#
# GNU Compilers
# if set, overwrites the architecture dependent defaults...
#
FC	= g77
# -Wall -pedantic
FFLAGS	= -O2 -Wimplicit -pipe
CC	= gcc
CFLAGS	= -O2 -pipe -ansi -Wall -pedantic
CPPFLAGS= -D__USE_FIXED_PROTOTYPES__
#
endif

ifdef USE_PVMe
#
# PVM 3.2 compatibility
#
M4FLAGS += -D PVMDATARAW=PVMRAW \
	   -D PvmDataRaw=PVMRAW \
	   -D pvmtaskinfo=taskinfo \
	   -D pvmhostinfo=hostinfo

M4FLAGS	+= -D SYNC_EXIT
#
endif

M4FLAGS += -D ARCHITECTURE=$(PVM_ARCH) hpfc_lib_m4_macros

COPY		= cp
MOVE 		= mv

#
# I distinguish between PVM{3,e}_ROOT...

PVM_INC		= $(PVM_ROOT)/include
PVM_CONF	= $(PVM_ROOT)/conf

ifdef USE_PVMe
#
# if another PVM is used, I still need PVM 3 m4 macros...
#
PVM_INC		= $(PVM_ROOT)/lib
PVM_CONF	= $(PVM3_ROOT)/conf
#
endif

#
# pvm3 portability macros for Fortran calls to C functions:

M4_CONF_FILE	= $(PVM_CONF)/$(PVM_ARCH).m4

M4FLAGS	+=	$(M4_CONF_FILE)

PVM_HEADERS	= pvm3.h fpvm3.h
LIB_M4FFILES = 	hpfc_packing.m4f \
		hpfc_reductions.m4f \
		hpfc_rtsupport.m4f \
		hpfc_shift.m4f \
		hpfc_bufmgr.m4f \
		hpfc_broadcast.m4f
LIB_M4CFILES =	hpfc_misc.m4c
LIB_FFILES =	hpfc_check.f \
		hpfc_fake.f

M4_HEADERS 	= hpfc_procs.m4h \
		  hpfc_buffers.m4h 
CORE_HEADERS	= hpfc_commons.h \
		  hpfc_parameters.h \
		  hpfc_param.h \
		  hpfc_globs.h \
		  hpfc_misc.h

DDC_FFILES 	= $(LIB_M4FFILES:.m4f=.f)
DDC_CFILES	= $(LIB_M4CFILES:.m4c=.c)
DDC_HEADERS 	= $(LIB_M4FFILES:.m4f=.h) \
		  $(M4_HEADERS:.m4h=.h)\
		  hpfc_includes.h

LIB_HEADERS	= $(CORE_HEADERS) \
		  $(DDC_HEADERS)

LIB_OBJECTS= $(addprefix $(PVM_ARCH)/, $(DDC_FFILES:.f=.o) $(DDC_CFILES:.c=.o))

M4_MACROS 	= hpfc_lib_m4_macros hpfc_architecture_m4_macros
HPFC_MAKEFILES 	= hpfc_Makefile hpfc_Makefile_init 
DOCS		= hpfc_runtime_library.README

#
# the files to install

SOURCES = 	$(M4_MACROS) \
		$(M4_HEADERS) \
		$(LIB_FFILES) \
		$(LIB_M4FFILES) \
		$(LIB_M4CFILES) \
		$(HPFC_MAKEFILES) \
		$(CORE_HEADERS) \
		$(DOCS) \
		$(SCRIPTS)

LIB_TARGET = $(PVM_ARCH)/libhpfcruntime.a

# $(LIB_OBJECTS) $(LIB_TARGET): $(PVM_ARCH)

$(PVM_ARCH):; mkdir $@

#
# Installation:

INSTALL_INC_DIR:=$(INSTALL_RTM_DIR)/hpfc
INSTALL_LIB_DIR:=$(INSTALL_RTM_DIR)/hpfc/$(PVM_ARCH)

INSTALL_INC =	$(CORE_HEADERS) \
		$(DDC_HEADERS) \
		$(HPFC_MAKEFILES) \
		$(M4_MACROS) \
		$(SCRIPTS) \
		$(LIB_FFILES)

INSTALL_LIB=	$(LIB_TARGET)

#
# rules
#

ifeq ($(PVM_ARCH),CM5)
all: $(CMMD_F77_H) 
endif

all: $(PVM_ARCH) $(PVM_HEADERS) $(DDC_HEADERS) $(DDC_CFILES) $(DDC_FFILES) \
		$(LIB_OBJECTS) $(LIB_TARGET) 

#
# get pvm headers
#

pvm3.h:	$(PVM_INC)/pvm3.h
	$(COPY) $(PVM_INC)/pvm3.h .

fpvm3.h:$(PVM_INC)/fpvm3.h
	$(COPY) $(PVM_INC)/fpvm3.h .

ifeq ($(PVM_ARCH),CM5)
#
$(CMMD_F77_H):	$(CMMD_INDIR)/cm/$(CMMD_F77_H)
	$(COPY) $(CMMD_INDIR)/cm/$(CMMD_F77_H) .
#
endif

#

$(LIB_TARGET):	$(PVM_HEADERS) $(LIB_HEADERS) $(LIB_OBJECTS) 
	$(RM) $(LIB_TARGET) 
	$(AR) $(ARFLAGS) $(LIB_TARGET) $(LIB_OBJECTS) 
	$(RANLIB) $(LIB_TARGET) 

%.h: %.f
	# building $@ from $<
	./hpfc_generate_h < $< > $@
	./hpfc_add_warning $@

$(PVM_ARCH)/%.o: %.c
	$(COMPILE) $< -o $@

$(PVM_ARCH)/%.o: %.f
	$(F77COMPILE) $< -o $@

hpfc_includes.h: $(LIB_M4FFILES:.m4f=.h) 
	#
	# building $@
	#
	{ for i in $(LIB_M4FFILES:.m4f=.h) ; do \
	  echo "      include \"$$i\"" ; done; } > $@
	./hpfc_add_warning $@

clean: local-clean
local-clean: 
	$(RM) *~ $(LIB_OBJECTS) $(PVM_HEADERS) \
		$(DDC_HEADERS) 	$(DDC_FFILES) $(DDC_CFILES) $(LIB_TARGET)
	test ! -d $(PVM_ARCH) || rmdir $(PVM_ARCH)

# that is all
#
