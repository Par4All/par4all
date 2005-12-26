# $Id$

F.sh = 	hpfc \
	hpfc_directives \
	hpfc_compile \
	hpfc_delete \
	hpfc_install

BIN_TARGET = hpfc_interactive

INSTALL_EXE = $(F.sh)

INSTALL_SHR = hpfc_stubs.f hpfc_stubs.direct

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

PROJECT = pips

include $(ROOT)/makes/main.mk

# local stuff
$(ARCH)/hpfc_interactive.o: $(ARCH)

$(ARCH)/hpfc_interactive: $(ARCH)/hpfc_interactive.o
	$(LINK) $(ARCH)/hpfc_interactive \
		$(ARCH)/hpfc_interactive.o -lreadline -ltermcap

# the direct version of the stubs need not be filtered by hpfc_directives.
hpfc_stubs.direct: hpfc_stubs.f
	# building $@ from $<
	sed 's,^!fcd\$$ fake,      call hpfc9,' $< > $@

clean: local-clean

local-clean:
	$(RM) hpfc_stubs.direct
