# $Id$

SOURCES=list_of_reports \
	old_documentation_makefile \
	dret103.ftex \
	dret104.ftex \
	dret105.ftex \
	dret109.ftex \
	dret110.ftex \
	dret116.ftex \
	dret117.ftex \
	dret124.ftex \
	dret127.ftex \
	dret128.ftex \
	dret133.ftex \
	dret134.ftex \
	dret136.ftex \
	dret137.ftex \
	dret138.ftex \
	dret139.ftex \
	dret140.ftex \
	dret141.ftex \
	dret142.ftex \
	dret143.ftex \
	dret144.1.1.ftex \
	dret144.ftex \
	dret145.ftex \
	dret146.ftex \
	dret151.ftex \
	dret152.ftex \
	dret161.ftex \
	dret163.ftex \
	dret174.ftex \
	dret175.ftex \
	dret184.ftex \
	dret189.ftex \
	pips-2.ftex \
	pips-org.fig \
	pips-org.tex

INSTALL_DOC=	dret133.ps dret105.ps
INSTALL_HTM= 	dret133 dret105

all: $(INSTALL_DOC) $(INSTALL_DOC:.ps=.html)

clean: local-clean
local-clean:
	$(RM) -r $(INSTALL_DOC) $(INSTALL_HTM) \
		$(INSTALL_DOC:.ps=.html) $(INSTALL_DOC:.ps=.dvi)

# end of it
#
