#
# $RCSfile: config.makefile,v $ version $Revision$
# ($Date: 1995/08/04 13:57:27 $, )
#

.INIT: Makefile.include

Makefile.include: Makefile.m4 
	m4 -DARCHITECTURE=$(PVM_ARCH) Makefile.m4 > Makefile.include

include ./Makefile.include

# that is all
#
