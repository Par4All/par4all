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

OBSOLETE_FILES= elarg.c elim_red.c env.c incl_p_h.c inter_poly.c \
	poly.c sc_elarg.c inter_demi.c inter_hyp.c inter_obj.c saturation.c \
	saturation.h sc_to_sg.c syst_convert.c liste-table.h GENPGM_TAGS.h

TARGET	= polyedre

LIB_CFILES= 	sc_enveloppe.c sc_enumerate.c chernikova_fixprec.c chernikova_mulprec.c chernikova.c

LIB_TARGET	= lib$(TARGET).a

INC_TARGET	= $(TARGET).h

################################################################## COMMON STUFF

ROOT    = ../..
PROJECT = linear
include $(ROOT)/makes/main.mk
