#
# config.makefile (version 1.19)
# 96/12/17, 16:58:18

PIPSMAKERC=$(PIPS_ROOT)/Src/Documentation/pipsmake/pipsmake-rc.tex

SOURCES=	tpips-user-manual.tex tpips.html

INSTALL_DOC=	tpips-user-manual.ps
INSTALL_HTM=	tpips-user-manual tpips.html

DERIVED_FILES=	$(INSTALL_DOC) tpips-user-manual.html

all: $(DERIVED_FILES)

clean: local-clean

local-clean:
	$(RM) -r $(DERIVED_FILES) *.aux *.log *.ind *.idx *.toc *.ilg *.dvi

# end of config.makefile
#
