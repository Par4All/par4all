# $Id$

FWD_DIRS	= src

# multi-phase compilation...
install:
	$(MAKE) -C src phase0
	$(MAKE) -C src phase1
	$(MAKE) -C src phase2
	$(MAKE) -C src phase3
	$(MAKE) -C src phase4
	$(MAKE) -C src phase5

# coldly clean everything
uninstall:
	$(RM) -r ./Include ./Lib ./Bin ./Share ./Utils ./Documentation
