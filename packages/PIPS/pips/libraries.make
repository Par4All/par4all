# $Id$
#
# Copyright 1989-2010 MINES ParisTech
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

#debug_output := $(shell echo libraries.mk  > /dev/tty)

######################################################################## NEWGEN

# newgen.libs definition
ifdef WITH_DYNAMIC_LIBRARIES
newgen.libs=newgenlibs
else
include $(NEWGEN_ROOT)/libraries.make
endif

######################################################################## LINEAR

# linear.libs definition
ifdef WITH_DYNAMIC_LIBRARIES
linear.libs=linearlibs
else
include $(LINEAR_ROOT)/libraries.make
endif

######################################################################## OTHERS

other.libs	= m
ifdef LINEAR_DEPEND_GMP
other.libs += gmp
endif

##################################################################### EXTERNALS

ifdef WITH_DYNAMIC_LIBRARIES
# only needed for pips
extern.libs	= polylib
else
# maybe must create a link to libpolylib64.a
extern.libs	= polylib
endif

################################################################### PIPS COMMON

# old stuff: 
# prgm_mapping scheduling reindexing array_dfg paf-util static_controlize pip

pipslibs.libs	= \
	top-level pipsmake static_controlize paf-util \
	wp65 hpfc hyperplane to_begin_with \
	instrumentation locality expressions transformations accel-util hwac \
	movements bootstrap callgraph icfg chains complexity \
	conversion prettyprint atomizer syntax c_syntax pointer_values \
	alias-classes effects-simple effects-convex effects-generic \
	comp_sections semantics control continuation rice ricedg \
	pipsdbm transformer preprocessor effects-util ri-util step properties \
	text-util misc reductions flint sac safescale phrase \
	gpu scalopes newgen pocc-interface taskify rstream_interface \
	regions_to_loops task_parallelization

ifdef WITH_DYNAMIC_LIBRARIES
pips.libs	= \
	pipslibs $(newgen.libs) $(linear.libs) \
	$(extern.libs) $(other.libs)
else
pips.libs	= \
	$(pipslibs.libs) $(newgen.libs) $(linear.libs) \
	$(extern.libs) $(other.libs)
endif

########################################################################## PIPS

PIPS_MAIN	= main_pips.o

######################################################################### TPIPS

tpips_add.libs	= readline ncurses
tpips.libs	= $(pips.libs) $(tpips_add.libs)

TPIPS_MAIN	= main_tpips.o

######################################################################### WPIPS

# The following locations should be parameterized somewhere else
# or à la autoconf
X11_ROOT=/usr/X11R6
OPENWINHOME=$(X11_ROOT)

WPIPS_CPPFLAGS 	= -I$(OPENWINHOME)/include -I$(X11_ROOT)/include -Iicons
WPIPS_LDFLAGS 	= -L$(OPENWINHOME)/lib -L$(X11_ROOT)/lib

wpips_add.libs	= xview olgx X11
wpips.libs	= $(pips.libs) $(wpips_add.libs)
WPIPS_MAIN 	= main_wpips.o

######################################################################### GPIPS

include $(MAKE.d)/has_gtk2.mk

# The following locations should be parameterized somewhere else
# or à la autoconf

ifeq ($(has_gtk2),ok)

GPIPS_CPPFLAGS 	:= $(shell pkg-config --cflags gtk+-2.0)
GPIPS_LDFLAGS 	:= $(shell pkg-config --libs gtk+-2.0)

gpips_add.libs	=
gpips.libs	= $(pips.libs) $(gpips_add.libs)
GPIPS_MAIN 	= main_gpips.o

endif # gtk2 availibility through pkg-config

######################################################################### FPIPS

ifndef PIPS_NO_WPIPS
	FPIPS_LDFLAGS	+= $(WPIPS_LDFLAGS)
	# By default, compile with wpips:
	fpips_add.libs	+= wpips $(wpips_add.libs)
endif

ifndef PIPS_NO_GPIPS
	FPIPS_LDFLAGS	+= $(GPIPS_LDFLAGS)
	# By default, compile with gpips:
	fpips_add.libs	+= gpips $(gpips_add.libs)
endif

fpips.libs	= pips tpips $(pips.libs) $(fpips_add.libs) $(tpips_add.libs)
FPIPS_MAIN	= main_fpips.o

VERSION=0.1
