#
# $RCSfile: config.makefile,v $ version $Revision$
# ($Date: 1995/08/03 14:03:49 $, )
#

.INIT: Makefile

Makefile: $RCSfile: config.makefile,v $ Makefile.m4 
	{ cat $RCSfile: config.makefile,v $
	  m4 -DARCHITECTURE=$(PVM_ARCH) Makefile.m4 ;\
	} > Makefile.new
	mv Makefile.new Makefile

# that is all
#
