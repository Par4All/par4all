#
# $RCSfile: config.makefile,v $ for env
#

SCRIPTS =	Xressources \
		xinitrc

SFILES =	pipsrc.ref \
		make-pipsrc.sh \
		make-pipsrc.csh

RFILES = 	pipsrc.sh \
		pipsrc.csh

all: pipsrc.sh pipsrc.csh

pipsrc.sh: pipsrc.ref
	$(RM) pipsrc.sh
	$(SHELL) make-pipsrc.sh 
	chmod a-w pipsrc.sh

pipsrc.csh: pipsrc.sh
	$(RM) pipsrc.csh
	$(SHELL) make-pipsrc.csh 
	chmod a-w pipsrc.csh

clean-compiled:
	$(RM) pipsrc.sh pipsrc.csh

clean: clean-compiled

# that is all
#
