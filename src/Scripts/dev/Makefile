#
# $Id$
#

UTILS=		install_pips_sources \
		pips_install_file \
		pips_set_directory_for_binaries

SCRIPTS = 	$(UTILS) \
		analyze_libraries \
		grep_libraries \
		order_libraries \
		pips-experiment \
		pips_at_night \
		pips_at_night_at_cri \
		build_so_files

SRC_MACROS =	define_libraries.sh

DDC_MACROS = 	define_libraries.make

MACROS	=	$(DDC_MACROS) $(SRC_MACROS) 

COPY	=	cp

INSTALL_UTL=	$(SCRIPTS)
INSTALL_INC=	$(MACROS)

SOURCES	=	$(SCRIPTS) $(SRC_MACROS)

quick-install: install_utils
all: $(DDC_MACROS)

clean: local-clean
local-clean:; $(RM) $(DDC_MACROS)

$(PIPS_ROOT)/Include/define_libraries.make:; touch $@

install_utils: 
	#
	# installing shared utils
	#
	$(INSTALL) $(PIPS_ROOT)/Utils $(UTILS)
	$(INSTALL) $(NEWGEN_ROOT)/Utils $(UTILS)
	$(INSTALL) $(LINEAR_ROOT)/Utils $(UTILS)

#
# where to install forward makefiles

SUBDIRS = . Src Libs Passes Scripts Runtimes Documentation \
	Src/Libs Src/Passes Src/Scripts Src/Runtimes Src/Documentation

DIRS= 	$(PIPS_ROOT) $(NEWGEN_ROOT) $(LINEAR_ROOT) \
	$(PIPS_DEVEDIR) $(NEWGEN_DEVEDIR) $(LINEAR_DEVEDIR)
