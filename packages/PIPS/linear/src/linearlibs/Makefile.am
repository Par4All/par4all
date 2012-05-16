#
# Copyright 1989-2009 MINES ParisTech
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


ifdef WITH_DYNAMIC_LIBRARIES
TARGET=linearlibs
DYNLIB_TARGET=lib$(TARGET).so
DYNLIB_ARCHIVES=$(addprefix $(LIB.d)/lib, $(addsuffix .a, $(linear.libs)))
endif

ROOT    = ../..
PROJECT = linear
nodist_include $(ROOT)/makes/main.mk

.build_inc:

build-header-file:

