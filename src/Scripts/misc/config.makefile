#
# $Id$
#

INSTALL_SHR =	pips-deal-with-include \
		pips-process-module \
		pips-split \
		pips-unsplit-workspace \
		subroutine_callgraph_level.pl \
		normalisation.pl \
		replace_file_name.pl \
		insert_alias_checks_in_source.pl \
		remove_useless_includes.pl \
		check_for_recursions.pl


UTL_SCRIPTS = 	filter_verbatim \
		job-make \
		job-receive \
		unjustify \
		multi_sccs.sh

FILES =	 	extract-doc.awk \
		accent.sed

#HFI =	handle_fortran_includes

INSTALL_UTL=	$(UTL_SCRIPTS) $(FILES)
SCRIPTS=	$(INSTALL_SHR) $(UTL_SCRIPTS)
SOURCES=	$(SCRIPTS) $(FILES)
# $(HFI).c

#$(ARCH)/$(HFI): $(ARCH)/$(HFI).o
#	$(RM) $@
#	$(LD) $(LDFLAGS) -o $@ $< -lrx
#	chmod a+rx-w $@

clean: local-clean
local-clean:; $(RM) $(ARCH)/$(HFI) $(ARCH)/$(HFI).o

all: .runable

# that is all
#
