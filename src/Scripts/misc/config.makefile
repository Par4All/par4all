#
# $RCSfile: config.makefile,v $ for misc
#

INSTALL_SHR=	pips-deal-with-include \
		pips-process-module \
		pips-split \
		pips-unsplit-workspace 

UTL_SCRIPTS = \
		filter_verbatim \
		job-make \
		job-receive \
		unjustify

FILES =	 \
		extract-doc.awk \
		accent.sed

HFI =	handle_fortran_includes

INSTALL_UTL=	$(UTL_SCRIPTS) $(FILES)
SCRIPTS=	$(INSTALL_SHR) $(UTL_SCRIPTS)
SOURCES=	$(SCRIPTS) $(FILES) $(HFI).c

$(ARCH)/$(HFI): $(ARCH)/$(HFI).o
	$(RM) $@
	$(LD) $(LDFLAGS) -o $@ $< -lrx
	chmod a+rx-w $@

clean: local-clean
local-clean:
	$(RM) $(ARCH)/$(HFI) $(ARCH)/$(HFI).o

all: .runable

# that is all
#
