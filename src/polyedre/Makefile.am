# $Id$
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

TARGET=polyedre
BUILT_SOURCES=$(TARGET).h
nodist_include_HEADERS=$(TARGET).h
AM_CPPFLAGS=-I$(srcdir)/../arithmetique -I../arithmetique -I../vecteur -I../contrainte -I../sc -I../sommet -I../ray_dte -I../sg -I../polynome $(POLYLIB64_CFLAGS) 

noinst_LTLIBRARIES=libpolyedre.la

obsolete_files= elarg.c elim_red.c env.c incl_p_h.c inter_poly.c \
	poly.c sc_elarg.c inter_demi.c inter_hyp.c inter_obj.c saturation.c \
	saturation.h sc_to_sg.c syst_convert.c liste-table.h GENPGM_TAGS.h

libpolyedre_la_SOURCES= 	sc_enveloppe.c sc_enumerate.c chernikova_fixprec.c chernikova_mulprec.c chernikova.c
include $(top_srcdir)/makes/cproto.mk

