#
# $Id$
# 

# unused: dcdg.tex 

ETEXF =	ri.tex \
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
	sac_private.tex

SOURCES = $(ETEXF) unstructured.idraw newgen_domain.sty \
	ri_C.tex optimizer.tex

NGENS =	$(ETEXF:%.tex=%.newgen)
HEADS = $(NGENS:%.newgen=%.h)
CGENS = $(NGENS:%.newgen=%.c)
SPECS =	$(NGENS:%.newgen=%.spec)

ALLHS =	all_newgen_headers.h specs.h

INSTALL_INC=	$(HEADS) $(NGENS) $(CGENS) $(SPECS) $(ALLHS)
INSTALL_DOC=	$(NGENS:%.newgen=%.ps)
INSTALL_HTM=	ri dg

all: $(ALLHS) $(INSTALL_DOC) ri.html dg.html

dvi: $(NGENS:%.newgen=%.dvi)
ps: $(NGENS:%.newgen=%.ps)
newgen: $(NGENS)
allhs: $(ALLHS)

all_newgen_headers.h: specs.h
	#
	# building $@ (ordered as specs.h!)
	#
	$(RM) $@
	sed -n 's,^\(.*\)_spec.*,#include "\1.h",p' $< > $@
	chmod a+r-w $@

specs.h: $(NGENS)
	#
	# building specs.h
	#
	newgen -c $(NGENS) | make_all_specs > specs.h

lisp_internal_representation: $(NGENS)
	#
	# building $@
	#
	newgen -lisp $(NGENS)
	touch $@

clean: local-clean

local-clean:
	$(RM) $(INSTALL_INC) $(NGENS) *.dvi *.ps
	$(RM) -r $(INSTALL_HTM) ri.html dg.html

# common stuff

ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif
include $(ROOT)/makes/main.mk
