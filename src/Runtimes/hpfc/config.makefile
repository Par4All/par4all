#
# $RCSfile: config.makefile,v $ version $Revision$
# ($Date: 1995/09/15 16:28:21 $, )
#

.INIT: Makefile.include

Makefile.include: Makefile.m4 
	$(M4) -DARCHITECTURE=$(PVM_ARCH) hpfc_lib_m4_macros $< > $@

include ./Makefile.include

# that is all
#
