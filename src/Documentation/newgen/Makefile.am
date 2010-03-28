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

F_c=ri.c \
	dg.c \
	compsec.c \
	word_attachment.c \
	reductions_private.c \
	interval_graph.c \
	pipsdbm_private.c \
	eole_private.c \
	abc_private.c \
	alias_private.c \
	ubs_private.c \
	c_parser_private.c \
	complexity_ri.c \
	database.c \
	paf_ri.c \
	tiling.c \
	graph.c \
	parser_private.c \
	hpf_private.c \
	property.c \
	makefile.c \
	reduction.c \
	message.c \
	text.c \
	hpf.c \
	sac_private.c \
	ri_C.c \
	cloning.c \
	outlining_private.c \
	step_private.c \
	points_to_private.c \
	freia_spoc_private.c
# this duplication cannot be avoided
# for bsd-make compatibility ...
F_h=ri.h \
	dg.h \
	compsec.h \
	word_attachment.h \
	reductions_private.h \
	interval_graph.h \
	pipsdbm_private.h \
	eole_private.h \
	abc_private.h \
	alias_private.h \
	ubs_private.h \
	c_parser_private.h \
	complexity_ri.h \
	database.h \
	paf_ri.h \
	tiling.h \
	graph.h \
	parser_private.h \
	hpf_private.h \
	property.h \
	makefile.h \
	reduction.h \
	message.h \
	text.h \
	hpf.h \
	sac_private.h \
	ri_C.h \
	cloning.h \
	outlining_private.h \
	step_private.h \
	points_to_private.h \
	freia_spoc_private.h
# this duplication cannot be avoided
# for bsd-make compatibility ...
F_tex=ri.tex \
	dg.tex \
	compsec.tex \
	word_attachment.tex \
	reductions_private.tex \
	interval_graph.tex \
	pipsdbm_private.tex \
	eole_private.tex \
	abc_private.tex \
	alias_private.tex \
	ubs_private.tex \
	c_parser_private.tex \
	complexity_ri.tex \
	database.tex \
	paf_ri.tex \
	tiling.tex \
	graph.tex \
	parser_private.tex \
	hpf_private.tex \
	property.tex \
	makefile.tex \
	reduction.tex \
	message.tex \
	text.tex \
	hpf.tex \
	sac_private.tex \
	ri_C.tex \
	cloning.tex \
	outlining_private.tex \
	step_private.tex \
	points_to_private.tex \
	freia_spoc_private.tex
# this duplication cannot be avoided
# for bsd-make compatibility ...
F_pdf=ri.pdf \
	dg.pdf \
	compsec.pdf \
	word_attachment.pdf \
	reductions_private.pdf \
	interval_graph.pdf \
	pipsdbm_private.pdf \
	eole_private.pdf \
	abc_private.pdf \
	alias_private.pdf \
	ubs_private.pdf \
	c_parser_private.pdf \
	complexity_ri.pdf \
	database.pdf \
	paf_ri.pdf \
	tiling.pdf \
	graph.pdf \
	parser_private.pdf \
	hpf_private.pdf \
	property.pdf \
	makefile.pdf \
	reduction.pdf \
	message.pdf \
	text.pdf \
	hpf.pdf \
	sac_private.pdf \
	ri_C.pdf \
	cloning.pdf \
	outlining_private.pdf \
	step_private.pdf \
	points_to_private.pdf \
	freia_spoc_private.pdf

if WITH_DOC
dist_noinst_DATA=$(F_tex) unstructured.idraw newgen_domain.sty
doc_DATA=$(F_pdf)
endif

dist_noinst_SCRIPTS=remove-latex-comments
noinst_LTLIBRARIES=libnewgenaux.la
libnewgenaux_la_SOURCES=$(F_c)
BUILT_SOURCES=specs.h $(F_c)
include $(top_srcdir)/makes/latex.mk
SUFFIXES+=.newgen 

AM_CPPFLAGS=$(NEWGENLIBS_CFLAGS)

include_HEADERS=all_newgen_headers.h specs.h $(F_h)

all_newgen_headers.h: specs.h Makefile
	sed -n 's,^\(.*\)_spec.*,#include "\1.h",p' specs.h > $@

specs.h:Makefile $(F_tex)
	$(MAKE) `echo $(F_c) | sed -e 's/\.c/.newgen /g'`
	PATH=`dirname $(NEWGEN)`:$$PATH $(NEWGEN) -c `echo $(F_c) | sed -e 's/\.c/.newgen/g'` | $(MAKE_ALL_SPECS) > specs.h

$(F_spec):specs.h
$(F_c):specs.h

.tex.newgen:
	$(srcdir)/remove-latex-comments $<

# Convert idraw/eps files.
ri.pdf:unstructured.pdf


CLEANFILES=$(F_c) $(include_HEADERS) *.newgen $(F_h) *.spec $(F_pdf) unstructured.pdf

