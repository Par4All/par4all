# $Id$

TPIPS	= tpips
PIPS	= pips

# source files
F.c	= $(wildcard *.c)
F.f	= $(wildcard *.f)
F.F	= $(wildcard *.F)
F.result= $(wildcard *.result)

# validation scripts
F.tpips	= $(wildcard *.tpips)
F.test	= $(wildcard *.test)

# validation targets
F.valid	= $(F.result:%=%/test)

here	:= $(shell pwd)
FLT	= sed -e 's,$(here),$$VDIR,g'

# default target is to clean
clean:
	$(RM) *~ *.result/out out err $(F.c:%.c=%.o) a.out *.tmp
	$(RM) -r *.database

validate: $(F.valid)

force-validate:
	$(RM) $(F.valid)
	$(MAKE) validate

# shell script
%.result/test: %.test
	$< | $(FLT)  > $@ ; exit 0

# tpips scripts
%.result/test: %.tpips
	$(TPIPS) $< | $(FLT) > $@ ; exit 0

%.result/test: %.tpips2
	$(TPIPS) $< 2<&1 | $(FLT) > $@ ; exit 0

# default_tpips
# FILE could be $<
%.result/test: %.c default_tpips
	WSPACE=$* FILE=$(here)/$< VDIR=$(here) $(TPIPS) default_tpips \
	| $(FLT) > $@ ; exit 0

%.result/test: %.f default_tpips
	WSPACE=$* FILE=$(here)/$< VDIR=$(here) $(TPIPS) default_tpips \
	| $(FLT) > $@ ; exit 0

%.result/test: %.F default_tpips
	WSPACE=$* FILE=$(here)/$< VDIR=$(here) $(TPIPS) default_tpips \
	| $(FLT) > $@ ; exit 0

# default_test relies on substitutions
DEFTEST	= default_test2
%.result/test: %.c $(DEFTEST)
	WSPACE=$* FILE=$(here)/$< sh $(DEFTEST) | $(FLT) > $@ ; exit 0

%.result/test: %.f default_test
	WSPACE=$* FILE=$(here)/$< sh $(DEFTEST) | $(FLT) > $@ ; exit 0

%.result/test: %.F default_test
	WSPACE=$* FILE=$(here)/$< sh $(DEFTEST) | $(FLT) > $@ ; exit 0

# what about nothing?
