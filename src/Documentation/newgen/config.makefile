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
	pipsdbm_private.tex

FTEXF =	complexity_ri.ftex \
	database.ftex \
	paf_ri.ftex \
	tiling.ftex \
	graph.ftex \
	parser_private.ftex \
	hpf_private.ftex \
	property.ftex \
	makefile.ftex \
	reduction.ftex \
	message.ftex \
	text.ftex \
	hpf.ftex

SOURCES = $(ETEXF) $(FTEXF) unstructured.idraw newgen_domain.sty

NGENS =	$(ETEXF:.tex=.newgen) $(FTEXF:.ftex=.newgen)
HEADS = $(NGENS:.newgen=.h)
SPECS =	$(NGENS:.newgen=.spec)

ALLHS =	all_newgen_headers.h specs.h

INSTALL_INC=	$(HEADS) $(SPECS) $(ALLHS)
INSTALL_DOC=	$(NGENS:.newgen=.ps)
INSTALL_HTM=	ri dg

all: $(ALLHS) $(INSTALL_DOC) ri.html dg.html

dvi: $(NGENS:.newgen=.dvi)
ps: $(NGENS:.newgen=.ps)
newgen: $(NGENS)

all_newgen_headers.h: $(NGENS)
	#
	# building $@
	#
	$(RM) $@
	for f in $(HEADS) ; do echo "#include \"$$f\"" ; done > $@
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
	$(RM) $(INSTALL_INC) $(NGENS) $(FTEXF:.ftex=.tex) *.dvi *.ps
	$(RM) -r $(INSTALL_HTM) ri.html dg.html

# end of $RCSfile: config.makefile,v $
#
