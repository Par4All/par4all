#
# $RCSfile: config.makefile,v $ version $Revision$
# ($Date: 1995/09/25 17:45:39 $, )
#

M4_PARAMS = $(HPFC_RUNTIME_M4_PARAMS)
M4_PARAMS += -D ARCHITECTURE=$(PVM_ARCH) hpfc_lib_m4_macros

.INIT: Makefile.include

Makefile.include: Makefile.m4 
	$(M4) $(M4_PARAMS) $< > $@

include ./Makefile.include

# that is all
#
