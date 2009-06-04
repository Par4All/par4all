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

# unused: dcdg.tex

use_pdflatex	= 1

F.tex =	ri.tex \
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
	cloning.tex

SOURCES = $(F.tex) unstructured.idraw newgen_domain.sty optimizer.tex

# Install in a newgen subdirectory:
HTM.subd = newgen
DOC.subd = newgen

F.newgen	= $(F.tex:%.tex=%.newgen)
F.h		= $(F.newgen:%.newgen=%.h)
F.c		= $(F.newgen:%.newgen=%.c)
F.spec		= $(F.newgen:%.newgen=%.spec)

ALLHS =	all_newgen_headers.h specs.h

INSTALL_INC=	$(F.h) $(F.newgen) $(F.c) $(F.spec) $(ALLHS)
INSTALL_DOC=	$(F.newgen:%.newgen=%.pdf)
INSTALL_HTM=	$(F.tex:%.tex=%.htdoc)

all: $(ALLHS) $(INSTALL_DOC)

dvi: $(F.newgen:%.newgen=%.dvi)
pdf: $(F.newgen:%.newgen=%.pdf)
newgen: $(F.newgen)
allhs: $(ALLHS)

%.newgen: %.tex
	$(RM) $@
	./remove-latex-comments $<
	chmod a-w $@

ri.pdf: unstructured.pdf

all_newgen_headers.h: specs.h
	#
	# building $@ (ordered as specs.h!)
	#
	$(RM) $@
	sed -n 's,^\(.*\)_spec.*,#include "\1.h",p' $< > $@
	chmod a+r-w $@

specs.h $(F.spec): $(F.newgen)
	#
	# building specs.h
	#
	newgen -c $(F.newgen) | make_all_specs > specs.h

lisp_internal_representation: $(F.newgen)
	#
	# building $@
	#
	newgen -lisp $(F.newgen)
	touch $@

clean: local-clean

local-clean:
	$(RM) $(INSTALL_INC) $(F.newgen)
	$(RM) *.dvi *.ind *.log *.ps *.pdf *.aux *.idx *.ilg *.toc *.out *.brf
	$(RM) -r $(INSTALL_HTM) ri.html dg.html


# what to do depending on phases
phase1: allhs

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif
include $(ROOT)/makes/main.mk
