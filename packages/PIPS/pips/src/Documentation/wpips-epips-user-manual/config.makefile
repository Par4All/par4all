#
# config.makefile (version 1.19)
# 96/12/17, 16:58:18

PIPSMAKERC=$(PIPS_ROOT)/Src/Documentation/pipsmake/pipsmake-rc.tex

SOURCES=	wpips-epips-user-manual.tex \
	A-279-modified.tex \
	Commandes-beatrice.tex \
	mybib-beatrice.bib \
	presentation_WP65.tex


INSTALL_DOC=	wpips-epips-user-manual.ps
INSTALL_HTM=	wpips-epips-user-manual

DERIVED_FILES=	$(INSTALL_DOC) wpips-epips-user-manual.html

all: $(DERIVED_FILES)

wpips-epips-user-manual.dvi: wpips-epips-user-manual.tex \
		wpips-epips-declarations.tex \
		wpips-epips-options-menu.tex \
		wpips-epips-transform-menu.tex

wpips-epips-declarations.tex \
wpips-epips-view-menu.tex \
wpips-epips-transform-menu.tex \
wpips-epips-compile-menu.tex \
wpips-epips-options-menu.tex : $(PIPSMAKERC)
	perl generate_all_menu_documentation < $(PIPSMAKERC)


clean: local-clean

local-clean:
	$(RM) -r $(DERIVED_FILES) *.aux *.log *.ind *.idx *.toc *.ilg *.dvi

# end of config.makefile
#
