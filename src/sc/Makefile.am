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

noinst_LTLIBRARIES=libsc.la

TARGET=sc

other_headers= sc-private.h

BUILT_SOURCES=$(TARGET).h sc_gram.h
nodist_include_HEADERS=$(TARGET).h
AM_CPPFLAGS=-I$(srcdir)/../arithmetique -I../arithmetique -I../vecteur -I../contrainte -I$(top_srcdir)

AM_YFLAGS=-p syst_ -d
AM_LFLAGS=-p syst_

libsc_la_SOURCES = 	sc_alloc.c \
	        sc_projection.c \
	        sc_read.c \
	        sc.c \
	        sc_integer_analyze.c \
	        sc_feasibility.c \
	        sc_intersection.c \
	        sc_integer_projection.c \
	        sc_normalize.c \
	        sc_build_sc_nredund.c \
	        sc_oppose.c \
	        sc_triang_elim_redond.c \
	        sc_elim_redund.c \
	        sc_elim_simple_redund.c \
		sc_insert_eq.c \
		sc_transformation.c\
		sc_var.c \
	        sc_eval.c \
		sc_unaires.c \
		sc_error.c \
		sc_io.c \
		sc_new_loop_bound.c \
		sc_simplex_feasibility_fixprec.c \
		sc_simplex_feasibility_mulprec.c \
		sc_simplex_feasibility.c \
		sc_debug.c \
		sc_janus_feasibility.c \
		sc_gram.y \
		sc_lex.l \
		sc_belong.c \
		$(other_headers)

CLEANFILES=sc_gram.[ch] sc_lex.c

include $(top_srcdir)/makes/cproto.mk
