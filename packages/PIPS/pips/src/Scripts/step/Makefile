# $Id$

F.sh = run_step.script step_install step_filtre_include

step_install:step_install.in
	sed -e 's,@datadir@,$(CURDIR)/../../Runtimes,g' $< > $@
	chmod +x $@

clean: local-clean
local-clean:
	$(RM) step_install


INSTALL_EXE = $(F.sh)

# common stuff
ROOT    = ../../..
PROJECT = pips
include $(ROOT)/makes/main.mk
