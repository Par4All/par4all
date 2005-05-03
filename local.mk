# $Id$

FWD_DIRS	= src

install:
	$(MAKE) -C src phase0
	$(MAKE) -C src phase1
	$(MAKE) -C src phase2
	$(MAKE) -C src phase3


uninstall:
	$(RM) -r ./Include ./Lib ./Bin ./Share ./Utils ./Documentation
