# $Id$
#
# Copyright 1989-2012 MINES ParisTech
#
# This file is part of Linear/C3 Library.
#
# Linear/C3 Library is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# Linear/C3 Library is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Linear/C3 Library.  If not, see <http://www.gnu.org/licenses/>.
#

TARGET	= sc

DERIVED_CFILES = sc_gram.c sc_lex.c

INC_CFILES = 	sc_alloc.c \
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
		sc_belong.c

LIB_CFILES= $(INC_CFILES) $(DERIVED_CFILES)

OTHER_HEADERS=	sc-local.h \
		sc-private.h \
		sc_gram.y \
		sc_lex.l

DERIVED_HEADERS = sc_gram.h

LIB_TARGET	= lib$(TARGET).a

INC_TARGET	= $(TARGET).h

################################################################## COMMON STUFF

ROOT    = ../..
PROJECT = linear
include $(ROOT)/makes/main.mk


YY2SYST	= sed '/extern char \*malloc/d;/^FILE *\*/s,=[^,;]*,,g;\
	s/YY/SYST_/g;s/yy/syst_/g;' 

sc_lex.c: sc_lex.l sc_gram.h
	$(SCAN) $< | $(YY2SYST)	> $@

sc_gram.c: sc_gram.y
	$(PARSE) -d $<
	$(YY2SYST) y.tab.c > sc_gram.c
	$(YY2SYST) y.tab.h > sc_gram.h
	$(RM) y.tab.c y.tab.h

sc_gram.h: sc_gram.c

clean: local-clean

local-clean:
	$(RM) $(DERIVED_HEADERS) $(DERIVED_CFILES)
