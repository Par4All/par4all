#
# $RCSfile: config.makefile,v $ version $Revision$
# ($Date: 1995/09/12 13:49:09 $, )
#

.INIT: Makefile.include

Makefile.include: Makefile.m4 
	m4 -DARCHITECTURE=$(PVM_ARCH) \
		hpfc_architecture_m4_macros hpfc_lib_m4_macros $< > $@

include ./Makefile.include

# that is all
#
