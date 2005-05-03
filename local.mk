# $Id$

FWD_DIRS	= src

install:
	$(MAKE) -C src phase0
	$(MAKE) -C src phase1
	$(MAKE) -C src phase2
	$(MAKE) -C src phase3
	$(MAKE) -C src phase4
	$(MAKE) -C src phase5

uninstall:
	$(RM) -r ./Bin ./Include ./Lib ./Share ./Utils ./Doc ./Runtime
