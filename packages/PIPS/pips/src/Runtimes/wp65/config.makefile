# $Id$
#
# Copyright 1989-2014 MINES ParisTech
#
# This file is part of PIPS.
#
# PIPS is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIPS.  If not, see <http://www.gnu.org/licenses/>.
#

CPPFLAGS+=	-D COMPILE_FOR_G77

SCRIPTS = 	compile_wp65
CFILES=		lance_wp65.c
MFILE=		Makefile.compile_wp65

SOURCES=	$(CFILES) $(SCRIPTS) $(MFILE) model.rc

LOCAL_LIB=	$(PVM_ARCH)/libwp65runtime.a

OFILES=	$(addprefix $(PVM_ARCH)/, $(CFILES:.c=.o))

#
# installation

INSTALL_RTM_DIR:= $(INSTALL_RTM_DIR)/wp65
INSTALL_LIB_DIR= $(INSTALL_RTM_DIR)/$(PVM_ARCH)

INSTALL_LIB=	$(LOCAL_LIB)
INSTALL_RTM=	$(MFILE) $(SCRIPTS)
INSTALL_SHR=	$(SCRIPTS) model.rc

#
# pvm headers:

CPPFLAGS+=	-I$(PVM_ROOT)/include

# 
# compilation and so.

$(PVM_ARCH)/%.o: %.c
	$(COMPILE) $< -o $@

all: $(PVM_ARCH) $(LOCAL_LIB) .runable

$(PVM_ARCH):; mkdir $@

$(LOCAL_LIB):	$(OFILES)
	$(AR) $(ARFLAGS) $(LOCAL_LIB) $(OFILES)
	$(RANLIB) $(LOCAL_LIB)

clean-compiled: clean
clean: local-clean
local-clean:; -$(RM) *~ $(OFILES) $(LOCAL_LIB)

# that is all
#
