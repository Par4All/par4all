#
# $RCSfile: config.makefile,v $ version $Revision$
# ($Date: 1997/01/07 11:15:56 $, )
#
# depends on 
# + PVM_ARCH 
# + PVM_ROOT
# + _HPFC_USE_PVMe_

ifeq ($(PIPS_ARCH),.)
RT_ARCH=$(PVM_ARCH)
else
RT_ARCH=$(PVM_ARCH)-$(PIPS_ARCH)
endif

#
# additional defs for m4
ifeq ($(FC),g77)
M4FLAGS	+=	-D _HPFC_NO_BYTE1_ \
		-D _HPFC_NO_INTEGER2_
_HPFC_USE_GNU_ = 1
endif

PVM_ENCODING_OPTION =	PvmDataInPlace

M4FLAGS+= -D _HPFC_DIMENSIONS_=3
M4FLAGS+= -D _HPFC_ENCODING_=$(PVM_ENCODING_OPTION)

M4FLAGS	+= -D _HPFC_DEMO_
M4FLAGS	+= -D _HPFC_DIRECT_
# M4FLAGS	+= -D _HPFC_DEBUG_

# the default on IBM is to use PVMe
ifeq ($(PVM_ARCH),RS6K)
_HPFC_USE_PVMe_ = 1
endif

ifdef _HPFC_USE_PVMe_
M4FLAGS	+= -D _HPFC_USE_PVMe_
endif


#############################################################################

SCRIPTS =	hpfc_llcmd \
		hpfc_add_warning \
		hpfc_generate_h \
		hpfc_generate_init

#
# Default compilers and options

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
# FFLAGS	= -O2 -u
FFLAGS	= -O2 -qarch=pwr2 -qtune=pwr2 -qhot -u
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
ifdef _HPFC_USE_GNU_
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

M4FLAGS += -D PVM_ARCH=$(PVM_ARCH) hpfc_lib_m4_macros

COPY	= cp
MOVE 	= mv

#
# I distinguish between PVM{3,e}_ROOT...

pvminc	= $(PVM_ROOT)/include
pvmconf	= $(PVM_ROOT)/conf

ifdef _HPFC_USE_PVMe_
#
# if another PVM is used, I still need PVM 3 m4 macros...
# IBM puts includes in lib:-(
pvminc	= $(PVM_ROOT)/lib
pvmconf	= $(PVM3_ROOT)/conf
#
endif

ifeq ($(PVM_ARCH),CRAY)
#
# CRAY PVM is does not have pvm_version
# also no need to link to the pvm library
#
pvminc	= /usr/include/mpp
M4FLAGS	+= 	-D _HPFC_NO_PVM_VERSION_ \
		-D _HPFC_NO_BYTE1_ \
		-D _HPFC_NO_INTEGER2_ \
		-D _HPFC_NO_REAL4_ \
		-D _HPFC_NO_COMPLEX8_
PVM_ENCODING_OPTION =	PvmDataRaw
endif

#
# pvm3 portability macros for Fortran calls to C functions:

M4COPT	+=	$(PVM_ARCH).m4

PVM_HEADERS  =	pvm3.h fpvm3.h
LIB_M4FFILES = 	hpfc_packing.m4f \
		hpfc_reductions.m4f \
		hpfc_rtsupport.m4f \
		hpfc_shift.m4f \
		hpfc_bufmgr.m4f \
		hpfc_broadcast.m4f
LIB_M4CFILES =	hpfc_misc.m4c
LIB_FFILES =	hpfc_check.f \
		hpfc_main.f \
		hpfc_main_host.f \
		hpfc_main_node.f

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

$(DDC_FFILES) $(DDC_CFILES) $(DDC_HEADERS): $(PVM_ARCH).m4

LIB_HEADERS	= $(CORE_HEADERS) \
		  $(DDC_HEADERS)

LIBOBJECTS:= $(addprefix $(RT_ARCH)/, $(DDC_FFILES:.f=.o) $(DDC_CFILES:.c=.o))

M4_MACROS 	= hpfc_lib_m4_macros hpfc_architecture_m4_macros
HPFC_MAKEFILES 	= hpfc_Makefile_init 
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

LIB_TARGET = $(RT_ARCH)/libhpfcruntime.a
MKI_TARGET = $(RT_ARCH)/compilers.make

# $(LIBOBJECTS) $(LIB_TARGET): $(RT_ARCH)

$(RT_ARCH):; mkdir $@

#
# Installation:

INSTALL_INC_DIR:=$(INSTALL_RTM_DIR)/hpfc
INSTALL_LIB_DIR:=$(INSTALL_RTM_DIR)/hpfc/$(RT_ARCH)

INSTALL_INC =	$(CORE_HEADERS) \
		$(DDC_HEADERS) \
		$(HPFC_MAKEFILES) \
		$(M4_MACROS) \
		$(SCRIPTS) \
		$(LIB_FFILES) 

INSTALL_LIB=	$(LIB_TARGET) $(MKI_TARGET)

#
# rules
#

ifeq ($(PVM_ARCH),CM5)
all: $(CMMD_F77_H) 
#
$(CMMD_F77_H):	$(CMMD_INDIR)/cm/$(CMMD_F77_H)
	$(COPY) $(CMMD_INDIR)/cm/$(CMMD_F77_H) .
endif

all: $(RT_ARCH) $(PVM_HEADERS) $(DDC_HEADERS) $(DDC_CFILES) $(DDC_FFILES) \
		$(LIBOBJECTS) $(LIB_TARGET) $(MKI_TARGET)

#
# get pvm headers
#

pvm3.h:	$(pvminc)/pvm3.h
	$(COPY) $< $@

fpvm3.h:$(pvminc)/fpvm3.h
	$(COPY) $< $@

$(PVM_ARCH).m4:
	$(COPY) $(pvmconf)/$(PVM_ARCH).m4 $@

#

$(LIB_TARGET):	$(PVM_HEADERS) $(LIB_HEADERS) $(LIBOBJECTS) 
	$(RM) $(LIB_TARGET) 
	$(ARCHIVE) $(LIB_TARGET) $(LIBOBJECTS) 
	$(RANLIB) $(LIB_TARGET) 

%.h: %.f
	# building $@ from $<
	sh ./hpfc_generate_h < $< > $@ ; \
	sh ./hpfc_add_warning $@

$(RT_ARCH)/%.o: %.c
	$(COMPILE) $< -o $@

ifeq ($(PVM_ARCH),CRAY)
$(RT_ARCH)/%.o: %.f
	$(F77COMPILE) $<
	mv $*.o $@
else
$(RT_ARCH)/%.o: %.f
	$(F77COMPILE) $< -o $@
endif

$(RT_ARCH)/compilers.make:
	#
	# building $@
	#
	{ echo "FC=$(FC)"; \
	  echo "FFLAGS=$(FFLAGS)";\
	  echo "CC=$(CC)";\
	  echo "CFLAGS=$(CFLAGS)";\
	  echo "CPPFLAGS=$(CPPFLAGS)";\
	  echo "M4=$(M4)";\
	  echo "_HPFC_USE_PVMe_=$(_HPFC_USE_PVMe_)";} > $@

hpfc_includes.h: $(LIB_M4FFILES:.m4f=.h) 
	#
	# building $@
	#
	{ for i in $(LIB_M4FFILES:.m4f=.h) ; do \
	  echo "      include \"$$i\"" ; done; } > $@
	sh ./hpfc_add_warning $@

clean: local-clean
local-clean: 
	$(RM) *~ $(LIBOBJECTS) $(PVM_HEADERS) \
		$(DDC_HEADERS) 	$(DDC_FFILES) $(DDC_CFILES) \
		$(LIB_TARGET)   $(MKI_TARGET)
	test ! -d $(RT_ARCH) || rmdir $(RT_ARCH)

# that is all
#
