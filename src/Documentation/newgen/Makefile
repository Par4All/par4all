#
# $Id$
# 

# unused: dcdg.tex 

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
	ri_C.tex

SOURCES = $(F.tex) unstructured.idraw newgen_domain.sty optimizer.tex

F.newgen	= $(F.tex:%.tex=%.newgen)
F.h		= $(F.newgen:%.newgen=%.h)
F.c		= $(F.newgen:%.newgen=%.c)
F.spec		= $(F.newgen:%.newgen=%.spec)

ALLHS =	all_newgen_headers.h specs.h

INSTALL_INC=	$(F.h) $(F.newgen) $(F.c) $(F.spec) $(ALLHS)
INSTALL_DOC=	$(F.newgen:%.newgen=%.pdf)
INSTALL_HTM=	$(F.tex:%.tex=%.htdoc)
# Install in a newgen subdirectory:
HTM.subd = newgen

all: $(ALLHS) $(INSTALL_DOC)

dvi: $(F.newgen:%.newgen=%.dvi)
pdf: $(F.newgen:%.newgen=%.pdf)
newgen: $(F.newgen)
allhs: $(ALLHS)

%.newgen: %.tex
	$(RM) $@
	./remove-latex-comments $<
	chmod a-w $@

all_newgen_headers.h: specs.h
	#
	# building $@ (ordered as specs.h!)
	#
	$(RM) $@
	sed -n 's,^\(.*\)_spec.*,#include "\1.h",p' $< > $@
	chmod a+r-w $@

specs.h: $(F.newgen)
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
	$(RM) *.dvi *.ind *.log *.ps *.pdf *.aux *.idx *.ilg *.toc
	$(RM) -r $(INSTALL_HTM) ri.html dg.html


# what to do depending on phases
phase2: allhs

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif
include $(ROOT)/makes/main.mk
