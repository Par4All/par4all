# $Id$

TPIPS	= tpips

F.tpips	= $(wildcard *.tpips)
F.test	= $(F.tpips:%.tpips=%.result/test)

clean:
	$(RM) *~ *.result/out out err
	$(RM) -r *.database

test: $(F.test)

force-test:
	$(RM) $(F.test)
	$(MAKE) test

%.result/test: %.tpips
	$(TPIPS) $< > $@
