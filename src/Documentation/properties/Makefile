# $Id$

SOURCES	= properties-rc.tex

INSTALL_DOC	= properties-rc.pdf
INSTALL_HTM	= properties-rc.htdoc
ETC_TARGET	=  properties.rc

all: $(ETC_TARGET) $(INSTALL_DOC)

properties.rc: properties-rc.tex
	#
	# building properties.rc
	#
	sed 's,	,    ,g;s/ *$$//;/^alias /d' $< | $(FLTVERB) > $@

clean: local-clean
local-clean:
	$(RM) -r $(INSTALL_HTM) $(INSTALL_DOC) $(ETC_TARGET)
	$(RM) properties-rc.html *.dvi *.ind *.log *.aux *.idx *.ilg \
		*.toc *~ *.brf *.out *.ps

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif
include $(ROOT)/makes/main.mk

# this is needed by phase0 of Passes/tpips
phase0: .build_etc
