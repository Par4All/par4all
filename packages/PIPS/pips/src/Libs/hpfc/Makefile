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

TARGET	= hpfc

LIB_CFILES=	directives.c \
		dynamic.c \
		hpfc.c \
		debug-util.c \
		hpfc-util.c \
		declarations.c \
		compiler-util.c \
		compiler.c \
		compile.c \
		run-time.c \
		generate.c \
		local-ri-util.c \
		inits.c \
		o-analysis.c \
		align-checker.c \
		messages.c \
		message-utils.c \
		build-system.c \
		io-util.c \
		io-compile.c \
		generate-util.c \
		remapping.c \
		host_node_entities.c \
		special_cases.c \
		lattice_extraction.c

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a

VALIDATE_TARGET	= Hpfc

# common stuff
ROOT    = ../../..
PROJECT = pips
include $(ROOT)/makes/main.mk
