#
# $RCSfile: config.makefile,v $ version $Revision$
# ($Date: 1995/09/12 13:51:31 $, )
#

.INIT: Makefile.include

Makefile.include: Makefile.m4 
	m4 -DARCHITECTURE=$(PVM_ARCH) hpfc_lib_m4_macros $< > $@

include ./Makefile.include

# that is all
#
