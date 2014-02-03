# $Id$
#
# Copyright 1989-2014 MINES ParisTech
#
# This file is part of PIPS.
#
# PIPS is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with PIPS.  If not, see <http://www.gnu.org/licenses/>.
#

ifndef ROOT
$(error "expected ROOT macro not found!")
endif

# is this one really needed?
ifndef PIPS_ROOT
ifeq ($(PROJECT),pips)
PIPS_ROOT	= $(ROOT)
else
PIPS_ROOT       = $(ROOT)/../pips
endif
endif # PIPS_ROOT

ifndef NEWGEN_ROOT
ifeq ($(PROJECT),newgen)
NEWGEN_ROOT	= $(ROOT)
else
NEWGEN_ROOT     = $(ROOT)/../newgen
endif
endif # NEWGEN_ROOT

ifndef LINEAR_ROOT
ifeq ($(PROJECT),linear)
LINEAR_ROOT	= $(ROOT)
else
LINEAR_ROOT     = $(ROOT)/../linear
endif
endif # LINEAR_ROOT

ifndef EXTERN_ROOT
EXTERN_ROOT     = $(ROOT)/../extern
endif # EXTERN_ROOT

