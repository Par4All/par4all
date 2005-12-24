# $Id$

SOURCES	= properties-rc.tex

INSTALL_DOC	= properties-rc.ps
INSTALL_HTM	= properties-rc
ETC_TARGET	=  properties.rc

all: $(ETC_TARGET) $(INSTALL_DOC) properties-rc.html

properties.rc: properties-rc.tex
	#
	# building properties.rc
	#
	sed 's,	,    ,g;s/ *$$//;/^alias /d' $< | $(FLTVERB) > $@

clean: local-clean
local-clean:
	$(RM) -r $(INSTALL_HTM) $(INSTALL_DOC) $(ETC_TARGET) \
		properties-rc.html *.dvi

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif
include $(ROOT)/makes/main.mk
