#
# config.makefile (version 1.4)
# 96/09/02, 11:13:24

G= faq

SOURCES=	$(G).tex \
		$(G).bib

INSTALL_DOC=	$(G).ps

INSTALL_HTM=	$(G)

all: $(INSTALL_DOC) $(G).html

clean: local-clean
local-clean:
	$(RM) -r $(INSTALL_DOC) $(INSTALL_HTM) *.dvi $(G).html

# end of config.makefile
#
