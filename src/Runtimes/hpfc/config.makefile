#
# $RCSfile: config.makefile,v $ version $Revision$
# ($Date: 1996/07/19 11:04:11 $, )
#

M4_PARAMS = $(HPFC_RUNTIME_M4_PARAMS)
M4_PARAMS += -D ARCHITECTURE=$(PVM_ARCH) hpfc_lib_m4_macros

Makefile.include: Makefile.m4 
	$(M4) $(M4_PARAMS) $< > $@

all clean clean-compiled recompile: Makefile.include

include ./Makefile.include

# that is all
#
