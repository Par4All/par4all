# $Id$

# pips exes
TPIPS	= tpips
PIPS	= pips

# default output file
TEST	= test

# source files
F.c	= $(wildcard *.c)
F.f	= $(wildcard *.f)
F.F	= $(wildcard *.F)

# result directory
F.result= $(wildcard *.result)

# validation scripts
F.tpips	= $(wildcard *.tpips)
F.test	= $(wildcard *.test)

# validation output
F.valid	= $(F.result:%=%/$(TEST))

here	:= $(shell pwd)
FLT	= sed -e 's,$(here),$$VDIR,g'
OK	= exit 0

# default target is to clean
clean:
	$(RM) *~ *.result/out out err $(F.c:%.c=%.o) a.out *.tmp
	$(RM) -r *.database

# regenerate "test" files: svn diff show the diffs!
validate:
	$(RM) $(F.valid)
	$(MAKE) $(F.valid)

# generate "out" files
validate-out:
	$(MAKE) TEST=out validate

# shell script
%.result/$(TEST): %.test
	$< | $(FLT)  > $@ ; $(OK)

# tpips scripts
%.result/$(TEST): %.tpips
	$(TPIPS) $< | $(FLT) > $@ ; $(OK)

%.result/$(TEST): %.tpips2
	$(TPIPS) $< 2<&1 | $(FLT) > $@ ; $(OK)

# default_tpips
# FILE could be $<
%.result/$(TEST): %.c default_tpips
	WSPACE=$* FILE=$(here)/$< VDIR=$(here) $(TPIPS) default_tpips \
	| $(FLT) > $@ ; $(OK)

%.result/$(TEST): %.f default_tpips
	WSPACE=$* FILE=$(here)/$< VDIR=$(here) $(TPIPS) default_tpips \
	| $(FLT) > $@ ; $(OK)

%.result/$(TEST): %.F default_tpips
	WSPACE=$* FILE=$(here)/$< VDIR=$(here) $(TPIPS) default_tpips \
	| $(FLT) > $@ ; $(OK)

# default_test relies on substitutions...
DEFTEST	= default_test2
%.result/$(TEST): %.c $(DEFTEST)
	WSPACE=$* FILE=$(here)/$< sh $(DEFTEST) \
	| $(FLT) > $@ ; $(OK)

%.result/$(TEST): %.f default_test
	WSPACE=$* FILE=$(here)/$< sh $(DEFTEST) \
	| $(FLT) > $@ ; $(OK)

%.result/$(TEST): %.F default_test
	WSPACE=$* FILE=$(here)/$< sh $(DEFTEST) \
	| $(FLT) > $@ ; $(OK)

# what about nothing?
